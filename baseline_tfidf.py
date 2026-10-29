# Baselines: majority class and TF-IDF + logistic regression
# Run prepare_data.py first.
# We evaluate on the VALIDATION set. The test set is only used at the very end.

import os
import json
import time
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score, f1_score

USE_TEST = False  # change to True only for the final results!

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/test.csv" if USE_TEST else "data/val.csv")
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
    acc = accuracy_score(val["label"], pred)
    f1 = f1_score(val["label"], pred, average="macro")
    f1_neg = f1_score(val["label"], pred, pos_label=0)
    print(f"{name:15} train size {size:6}  accuracy {acc:.3f}  macro-F1 {f1:.3f}  F1(negative) {f1_neg:.3f}")
    results.append({"model": name, "train_size": size, "accuracy": acc, "macro_f1": f1, "f1_negative": f1_neg})


# 1. majority class: always says "positive"
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(train["text"], train["label"])
evaluate("majority", len(train), dummy.predict(val["text"]))

# 2. TF-IDF + logistic regression, try a few values of C on the validation set
best_C, best_f1 = 1.0, 0
for C in [1, 3, 10, 30, 100]:
    model = make_model(C).fit(train["text"], train["label"])
    f1 = f1_score(val["label"], model.predict(val["text"]), average="macro")
    print(f"C = {C}: macro-F1 {f1:.3f}")
    if f1 > best_f1:
        best_C, best_f1 = C, f1
print("best C:", best_C)

start = time.time()
model = make_model(best_C).fit(train["text"], train["label"])
print(f"training time: {time.time() - start:.1f} s")
pred = model.predict(val["text"])
evaluate("tfidf", len(train), pred)

# results for each website separately
for source in val["source"].unique():
    part = val["source"] == source
    print(f"  {source}: macro-F1 {f1_score(val['label'][part], pred[part], average='macro'):.3f} ({part.sum()} reviews)")

# save the wrong predictions for the error analysis
errors = val[val["label"] != pred]
errors.to_csv("results/tfidf_errors.csv", index=False)
print(len(errors), "errors saved to results/tfidf_errors.csv")

# 3. data-size experiment: train on 500 / 1000 / 2000 / 5000 reviews
subsets = json.load(open("data/subsets.json"))
for seed in subsets:
    for size, ids in subsets[seed].items():
        small_train = train[train["id"].isin(ids)]
        m = make_model(best_C).fit(small_train["text"], small_train["label"])
        evaluate("tfidf", int(size), m.predict(val["text"]))

# save all results + average over the 3 seeds
df = pd.DataFrame(results)
df.to_csv("results/baselines.csv", index=False)
print("\nAverage over seeds:")
print(df.groupby(["model", "train_size"])[["accuracy", "macro_f1", "f1_negative"]].agg(["mean", "std"]).round(3))
