# Error analysis, round 2: compare the mistakes of TF-IDF and of our best LoRA model (validation set)
#
# Uses files that already exist in results/:
#   tfidf_errors.csv                       (from baseline_tfidf.py)
#   errors_lora_r8_all_nfull_s1/2/3.csv    (from train_xlmr.py)
# Runs on CPU in a few seconds:  python compare_errors.py
# Output:
#   results/error_comparison.md       tables for the report
#   results/lora_errors_sample20.csv  20 LoRA errors to read and categorize by hand

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--lora_run", default="lora_r8_all_nfull")  # best LoRA setting (3 seeds)
args = parser.parse_args()

val = pd.read_csv("data/val.csv")
tfidf_wrong = set(pd.read_csv("results/tfidf_errors.csv")["id"])

# a review counts as a LoRA error if LoRA got it wrong in at least 2 of the 3 seeds
# (so we look at stable mistakes, not random ones)
lora_counts = pd.Series(dtype=int)
for seed in [1, 2, 3]:
    ids = pd.read_csv(f"results/errors_{args.lora_run}_s{seed}.csv")["id"]
    lora_counts = lora_counts.add(pd.Series(1, index=ids), fill_value=0)
lora_wrong = set(lora_counts[lora_counts >= 2].index)

val["tfidf_wrong"] = val["id"].isin(tfidf_wrong)
val["lora_wrong"] = val["id"].isin(lora_wrong)
groups = {
    "both wrong": val[val["tfidf_wrong"] & val["lora_wrong"]],
    "only TF-IDF wrong (LoRA fixed it)": val[val["tfidf_wrong"] & ~val["lora_wrong"]],
    "only LoRA wrong (LoRA broke it)": val[~val["tfidf_wrong"] & val["lora_wrong"]],
    "both correct": val[~val["tfidf_wrong"] & ~val["lora_wrong"]],
}

# simple measurable properties of the reviews in each group
LT_LETTERS = "[ąčęėįšųūž]"


def properties(df):
    words = df["text"].str.split().str.len()
    return {
        "reviews": len(df),
        "% true label negative": 100 * (df["label"] == 0).mean(),
        "median words": words.median(),
        "% very short (≤3 words)": 100 * (words <= 3).mean(),
        "% long (>40 words)": 100 * (words > 40).mean(),
        "% no Lithuanian letters (5+ words)": 100 * ((words >= 5) & ~df["text"].str.contains(LT_LETTERS)).mean(),
        "% with negation": 100 * df["text"].str.contains(r"\bne\b|\bnėra\b|\bne\w{3,}").mean(),
        "% from atsiliepimai.lt": 100 * (df["source"] == "atsiliepimai.lt").mean(),
    }


table = pd.DataFrame({name: properties(df) for name, df in groups.items()}).round(1)

lines = ["# Error comparison: TF-IDF vs LoRA (validation set)", "",
         f"LoRA = `{args.lora_run}`, a review counts as a LoRA error if it was wrong in at least 2 of 3 seeds.", "",
         f"- TF-IDF errors: {len(tfidf_wrong)}",
         f"- LoRA errors (stable): {len(lora_wrong)}",
         f"- both wrong: {len(groups['both wrong'])}",
         f"- only TF-IDF wrong (LoRA fixed): {len(groups['only TF-IDF wrong (LoRA fixed it)'])}",
         f"- only LoRA wrong (new LoRA errors): {len(groups['only LoRA wrong (LoRA broke it)'])}", "",
         "## Properties of the reviews in each group", "",
         "| Property | " + " | ".join(table.columns) + " |", "|---|" + "---|" * len(table.columns)]
for prop, row in table.iterrows():
    lines.append(f"| {prop} | " + " | ".join(f"{v:g}" for v in row) + " |")
open("results/error_comparison.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))

# 20 LoRA errors to categorize by hand (same categories as for TF-IDF)
lora_err = val[val["lora_wrong"]]
sample = lora_err.sample(min(20, len(lora_err)), random_state=0).copy()
sample["also_tfidf_wrong"] = sample["tfidf_wrong"]
sample["reason"] = ""
sample[["id", "text", "sentiment", "source", "also_tfidf_wrong", "reason"]].to_csv(
    "results/lora_errors_sample20.csv", index=False)

print("\n20 LoRA errors (the model predicted the OPPOSITE of TRUE):\n")
for _, r in sample.iterrows():
    tag = "TF-IDF also wrong" if r["tfidf_wrong"] else "TF-IDF was right"
    print(f"[{r['id']}] TRUE: {r['sentiment']} | {tag}\n{r['text']}\n")
print("saved results/error_comparison.md and results/lora_errors_sample20.csv")