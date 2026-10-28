# Data preparation for our sentiment project
# Dataset: engineerai/lithuanian-estore-sentiment-binary (Hugging Face)
# Output: data/train.csv, data/val.csv, data/test.csv, data/subsets.json, data/stats.txt

import os
import re
import json
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

# if you downloaded the csv yourself, put its path here instead
DATA_PATH = "hf://datasets/engineerai/lithuanian-estore-sentiment-binary/dataset.csv"
SEED = 42

os.makedirs("data", exist_ok=True)
stats = []  # we save the number of rows after every step for the report


def save_stat(step, df):
    pos = (df["sentiment"] == "positive").mean() * 100
    line = f"{step}: {len(df)} rows, {pos:.1f}% positive"
    print(line)
    stats.append(line)


# 1. load the data
df = pd.read_csv(DATA_PATH, encoding="utf-8")
df["id"] = range(len(df))
save_stat("original", df)
print(df["sentiment"].value_counts())

# 2. encoding: make Lithuanian letters always the same unicode form
df["review_text"] = df["review_text"].astype(str).apply(lambda t: unicodedata.normalize("NFC", t))
broken = df["review_text"].str.contains("Ä|Å|Ã").sum()  # this is how ą, š, ė look when encoding is wrong
print("reviews with broken letters:", broken)
stats.append(f"reviews with broken letters: {broken}")

# 3. drop neutral reviews (our task is positive vs negative)
df = df[df["sentiment"] != "neutral"]
save_stat("after dropping neutral", df)

# 4. clean the text: lowercase + remove extra spaces, then drop empty reviews
df["text"] = df["review_text"].str.lower().str.strip()
df["text"] = df["text"].apply(lambda t: re.sub(r"\s+", " ", t))
df = df[df["text"] != ""]
save_stat("after dropping empty reviews", df)

# 5. duplicates
# same text with BOTH labels -> we don't know the right label, delete all copies
n_labels = df.groupby("text")["sentiment"].nunique()
conflicts = n_labels[n_labels > 1].index
df = df[~df["text"].isin(conflicts)]
save_stat(f"after removing conflicting duplicates ({len(conflicts)} texts)", df)
# same text with the same label -> keep only one copy
df = df.drop_duplicates(subset="text")
save_stat("after removing duplicates", df)

# 6. hide personal data (emails, phone numbers, links)
emails = df["text"].str.contains(r"\S+@\S+\.\w+").sum()
phones = df["text"].str.contains(r"\+370\s?\d{3}\s?\d{5}|\b8\s?6\d{2}\s?\d{5}\b").sum()
df["text"] = df["text"].str.replace(r"\S+@\S+\.\w+", "<email>", regex=True)
df["text"] = df["text"].str.replace(r"\+370\s?\d{3}\s?\d{5}|\b8\s?6\d{2}\s?\d{5}\b", "<phone>", regex=True)
df["text"] = df["text"].str.replace(r"https?://\S+|www\.\S+", "<url>", regex=True)
print(f"masked {emails} emails and {phones} phone numbers")
stats.append(f"masked {emails} emails and {phones} phone numbers")

# 7. negation check: did "ne" survive the stopword removal?
df["has_negation"] = df["text"].str.contains(r"\bne\b|\bnėra\b|\bne\w{3,}")
neg_table = df.groupby("sentiment")["has_negation"].mean() * 100
print("% reviews with negation:\n", neg_table.round(1))
stats.append("% reviews with negation: " + str(neg_table.round(1).to_dict()))

# 8. keep only the columns we need
# rating_normalized is NOT used - it is the label in another form (leakage)
df["label"] = (df["sentiment"] == "positive").astype(int)  # 1 = positive, 0 = negative
df = df[["id", "text", "label", "sentiment", "source", "Store"]]

# 9. split 80 / 10 / 10, stratified by label and source
group = df["label"].astype(str) + df["source"]
train, rest = train_test_split(df, test_size=0.2, random_state=SEED, stratify=group)
group_rest = rest["label"].astype(str) + rest["source"]
val, test = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=group_rest)

# check that no review is in two splits
assert len(set(train["text"]) & set(test["text"])) == 0
assert len(set(train["text"]) & set(val["text"])) == 0
assert len(set(val["text"]) & set(test["text"])) == 0

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)
save_stat("train", train)
save_stat("val", val)
save_stat("test", test)
for name, part in [("train", train), ("val", val), ("test", test)]:
    line = f"{name} sources: {part['source'].value_counts().to_dict()}"
    print(line)
    stats.append(line)

# 10. small training subsets for the data-size experiment (3 different random samples)
subsets = {}
for seed in [1, 2, 3]:
    subsets[seed] = {}
    for size in [500, 1000, 2000, 5000]:
        if size < len(train):
            sample, _ = train_test_split(train, train_size=size, random_state=seed, stratify=train["label"])
            subsets[seed][size] = sample["id"].tolist()
with open("data/subsets.json", "w") as f:
    json.dump(subsets, f)

with open("data/stats.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(stats))
print("done, files saved in data/")
