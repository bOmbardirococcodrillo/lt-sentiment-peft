# Generate synthetic Lithuanian e-shop reviews with an open-weight LLM (run on a Kaggle GPU),
# then make them look like our dataset: lowercase, no punctuation, lemmatized, no stopwords.
#
# We use an open-weight model that runs on Kaggle, so no data is sent to an external LLM service.
#
# Run:  python generate_synthetic.py
# Output:
#   data/synthetic_raw.csv        the reviews exactly as the LLM wrote them
#   data/synthetic.csv            processed reviews (same style as train.csv) -> used for training
#   data/synthetic_spotcheck.csv  50 random reviews for a native speaker to check by hand

import argparse
import random
import re
from collections import Counter

import pandas as pd
import simplemma
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="utter-project/EuroLLM-1.7B-Instruct")  # open-weight, knows Lithuanian
parser.add_argument("--n_pos", type=int, default=2600)  # we need more positive reviews (~79% of our data)
parser.add_argument("--n_neg", type=int, default=900)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

# what the reviews are about (similar to the shops in our dataset) - gives variety
PRODUCTS = ["telefoną", "nešiojamą kompiuterį", "televizorių", "ausines", "batus", "striukę", "suknelę",
            "kosmetiką", "kvepalus", "sofą", "spintą", "kilimėlius automobiliui", "automobilio dalis",
            "sporto prekes", "dviratį", "žaislus vaikams", "maistą katei", "virtuvės indus",
            "dulkių siurblį", "kelionės draudimą", "lagaminą", "knygas"]
ASPECTS = {
    "positive": ["greitą pristatymą", "gerą prekės kokybę", "malonų aptarnavimą", "gerą kainą",
                 "tvarkingą pakuotę", "tai, kad prekė atitiko aprašymą", "greitą atsakymą į klausimą"],
    "negative": ["vėluojantį pristatymą", "prastą prekės kokybę", "nemandagų aptarnavimą",
                 "sugadintą pakuotę", "tai, kad prekė neatitiko aprašymo", "sudėtingą prekės grąžinimą",
                 "tai, kad niekas neatsako į laiškus"],
}
WORD = {"positive": "teigiamą", "negative": "neigiamą"}
PROMPT = ("Parašyk vieną trumpą {word} internetinės parduotuvės atsiliepimą lietuvių kalba. "
          "Pirkėjas nusipirko {product} ir paminėjo {aspect}. Rašyk kaip tikras pirkėjas, "
          "1–3 sakiniai, be pavadinimo. Pateik tik atsiliepimo tekstą.")
# (English: "Write one short positive/negative online-shop review in Lithuanian. The customer bought
#  {product} and mentions {aspect}. Write like a real customer, 1-3 sentences, no title. Only the review text.")

# ---------------- 1. generate
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"  # needed for batch generation with decoder models
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
llm = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()

jobs = []
for sentiment, n in [("positive", args.n_pos), ("negative", args.n_neg)]:
    for _ in range(n):
        jobs.append((sentiment, random.choice(PRODUCTS), random.choice(ASPECTS[sentiment])))
random.shuffle(jobs)

rows = []
for i in range(0, len(jobs), args.batch_size):
    batch = jobs[i:i + args.batch_size]
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(word=WORD[s], product=p, aspect=a)}],
                                       tokenize=False, add_generation_prompt=True) for s, p, a in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=80, do_sample=True, temperature=0.9, top_p=0.95,
                           pad_token_id=tok.pad_token_id)
    answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for (s, p, a), text in zip(batch, answers):
        text = " ".join(text.strip().split("\n\n")[0].split())  # first paragraph, one line
        rows.append({"raw_text": text.strip('"„“ '), "sentiment": s, "product": p, "aspect": a})
    if (i // args.batch_size) % 10 == 0:
        print(f"generated {len(rows)} / {len(jobs)}  e.g.: {rows[-1]['raw_text'][:90]}")

raw = pd.DataFrame(rows)
raw.to_csv("data/synthetic_raw.csv", index=False)
print(f"\n{len(raw)} reviews generated -> data/synthetic_raw.csv")

# ---------------- 2. simple quality filters
n_start = len(raw)
raw = raw[raw["raw_text"].str.split().str.len().between(3, 80)]  # not empty, not endless
english = {"the", "and", "is", "it", "this", "was", "very", "good", "product", "review"}
eng_share = raw["raw_text"].str.lower().str.split().apply(lambda ws: sum(w in english for w in ws) / len(ws))
raw = raw[eng_share < 0.2]  # drop answers written in English
raw = raw.drop_duplicates(subset="raw_text")
print(f"after quality filters: {len(raw)} of {n_start} kept")

# ---------------- 3. make them look like our dataset
# our reviews are lemmatized and stopwords are removed by the dataset author. We copy that style:
#  - words that already appear in our real training data are kept as they are
#  - unknown word forms are lemmatized with simplemma
#  - stopwords are removed, but only the ones that (almost) never appear in the real data,
#    so we remove exactly what the dataset author removed
train = pd.read_csv("data/train.csv")
counts = Counter(" ".join(train["text"]).split())
LT_STOPWORDS = ["ir", "kad", "bet", "o", "tai", "jau", "dar", "net", "tik", "gal", "su", "be", "iš", "į",
                "per", "prie", "nuo", "apie", "už", "po", "pas", "ant", "iki", "dėl", "tarp", "kaip", "kur",
                "kas", "ką", "kuris", "kuri", "kurie", "šis", "ši", "šie", "tas", "ta", "tie", "tą", "jis",
                "ji", "jie", "jos", "jo", "jų", "jam", "jai", "jiems", "aš", "man", "mane", "mano", "tu",
                "tau", "tavo", "mes", "mus", "mums", "mūsų", "jūs", "jums", "jūsų", "sau", "savo", "yra",
                "ar", "nes", "todėl", "taip", "labai", "viskas", "kiek", "čia", "ten", "vis", "dabar",
                "kai", "jei", "jeigu", "nei", "ne", "irgi", "buvo", "bus"]
stopwords = {w for w in LT_STOPWORDS if counts[w] < 5}
print("stopwords removed (not in real data):", sorted(stopwords))
print("kept (they appear in real data):", sorted(set(LT_STOPWORDS) - stopwords))


def to_dataset_style(text):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    words = []
    for w in text.split():
        if w in stopwords:
            continue
        if counts[w] < 3:  # word form never/rarely seen in real data -> lemmatize it
            w = simplemma.lemmatize(w, lang="lt").lower().strip(".")
        if w:
            words.append(w)
    return " ".join(words)


raw["text"] = raw["raw_text"].apply(to_dataset_style)
raw = raw[raw["text"].str.split().str.len() >= 2].drop_duplicates(subset="text")
# nonsense / non-Lithuanian filter: at least half of the words must exist somewhere in the real data
known_share = raw["text"].str.split().apply(lambda ws: sum(counts[w] > 0 for w in ws) / len(ws))
print(f"dropped {(known_share < 0.5).sum()} reviews where most words are unknown (nonsense or not Lithuanian)")
raw = raw[known_share >= 0.5]
# never allow a synthetic review that is identical to a real review (especially val/test)
real_texts = set(train["text"]) | set(pd.read_csv("data/val.csv")["text"]) | set(pd.read_csv("data/test.csv")["text"])
raw = raw[~raw["text"].isin(real_texts)]
raw["label"] = (raw["sentiment"] == "positive").astype(int)
raw["source"] = "synthetic"
raw[["text", "label", "sentiment", "source", "raw_text", "product", "aspect"]].to_csv("data/synthetic.csv", index=False)
print(f"final synthetic reviews: {len(raw)} ({(raw['label'] == 1).sum()} positive, {(raw['label'] == 0).sum()} negative)")

# ---------------- 4. compare with real data (put this in the report)
def describe(df, name):
    words = df["text"].str.split()
    known = words.apply(lambda ws: sum(counts[w] > 0 for w in ws) / max(len(ws), 1)).mean() * 100
    neg = df["text"].str.contains(r"\bne\b|\bnėra\b|\bne\w{3,}").mean() * 100
    print(f"{name:10} avg words {words.str.len().mean():5.1f} | median {words.str.len().median():4.0f} "
          f"| words known in real data {known:5.1f}% | reviews with negation {neg:5.1f}%")


print()
describe(train, "real")
describe(raw, "synthetic")
print("\nexamples (raw -> processed):")
for _, r in raw.sample(min(6, len(raw)), random_state=1).iterrows():
    print(f"[{r['sentiment']}] {r['raw_text']}\n   -> {r['text']}")

# ---------------- 5. sample for a manual check by a native speaker
check = raw.sample(min(50, len(raw)), random_state=1)[["raw_text", "text", "sentiment"]].copy()
check["grammar_ok_1_0"] = ""
check["label_ok_1_0"] = ""
check["comment"] = ""
check.to_csv("data/synthetic_spotcheck.csv", index=False)
print("\n50 reviews for manual checking -> data/synthetic_spotcheck.csv")