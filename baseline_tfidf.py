# Baselines: majority class and TF-IDF + logistic regression
# Run prepare_data.py first.
#
#   python baseline_tfidf.py          -> validation set (all experiments, incl. the data-size curve)
#   python baseline_tfidf.py --test   -> FINAL results on the test set (only at the very end!)
#
# C is ALWAYS chosen on the validation set, also in --test mode (the test set is never used for choices).

import argparse
import os
import json
import time
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
evalset = pd.read_csv("data/test.csv") if args.test else val
suffix = "_test" if args.test else ""   # test results go to separate files, val results are not overwritten
os.makedirs("results", exist_ok=True)
results = []


def make_model(C=1.0):
    # word features + character features (3-5 letters, helps with Lithuanian word endings)
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
    )
    # class_weight="balanced" because only ~19% of reviews are negative
    return make_pipeline(features, LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def evaluate(name, size, pred):
    acc = accuracy_score(evalset["label"], pred)
    f1 = f1_score(evalset["label"], pred, average="macro")
    f1_neg = f1_score(evalset["label"], pred, pos_label=0)
    print(f"{name:15} train size {size:6}  accuracy {acc:.3f}  macro-F1 {f1:.3f}  F1(negative) {f1_neg:.3f}")
    results.append({"model": name, "train_size": size, "split": "test" if args.test else "val",
                    "accuracy": acc, "macro_f1": f1, "f1_negative": f1_neg})


# 1. majority class: always says "positive"
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(train["text"], train["label"])
evaluate("majority", len(train), dummy.predict(evalset["text"]))

# 2. TF-IDF + logistic regression, choose C on the VALIDATION set
best_C, best_f1 = 1.0, 0
for C in [1, 3, 10, 30, 100]:
    model = make_model(C).fit(train["text"], train["label"])
    f1 = f1_score(val["label"], model.predict(val["text"]), average="macro")
    print(f"C = {C}: val macro-F1 {f1:.3f}")
    if f1 > best_f1:
        best_C, best_f1 = C, f1
print("best C:", best_C)

start = time.time()
model = make_model(best_C).fit(train["text"], train["label"])
print(f"training time: {time.time() - start:.1f} s")
pred = model.predict(evalset["text"])
evaluate("tfidf", len(train), pred)

# results for each website separately
for source in evalset["source"].unique():
    part = evalset["source"] == source
    print(f"  {source}: macro-F1 {f1_score(evalset['label'][part], pred[part], average='macro'):.3f} ({part.sum()} reviews)")

# save the wrong predictions for the error analysis
errors = evalset[evalset["label"] != pred]
errors.to_csv(f"results/tfidf_errors{suffix}.csv", index=False)
print(len(errors), f"errors saved to results/tfidf_errors{suffix}.csv")

# 3. data-size experiment: train on 500 / 1000 / 2000 / 5000 reviews (validation only)
if not args.test:
    subsets = json.load(open("data/subsets.json"))
    for seed in subsets:
        for size, ids in subsets[seed].items():
            small_train = train[train["id"].isin(ids)]
            m = make_model(best_C).fit(small_train["text"], small_train["label"])
            evaluate("tfidf", int(size), m.predict(evalset["text"]))

# save all results + average over the 3 seeds
df = pd.DataFrame(results)
df.to_csv(f"results/baselines{suffix}.csv", index=False)
print(f"\nsaved results/baselines{suffix}.csv")
print(df.groupby(["model", "train_size"])[["accuracy", "macro_f1", "f1_negative"]].agg(["mean", "std"]).round(3))
