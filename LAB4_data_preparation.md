# Lab 4: Data preparation and first baseline

Goal: make clean train / val / test files that everyone uses, and get the first baseline numbers.
Where: Kaggle notebook with **Internet on** (no GPU needed). Time: about 1 hour.

**Rule:** we never look at the test set until the end. All decisions are made on val.

## 1. Get the code

```python
!git clone https://github.com/bOmbardirococcodrillo/lt-sentiment-peft.git
%cd lt-sentiment-peft
!pip install -q -r requirements.txt
```

## 2. Look at the data

```python
import pandas as pd
df = pd.read_csv("hf://datasets/engineerai/lithuanian-estore-sentiment-binary/dataset.csv")
print(df.shape)
print(df["sentiment"].value_counts())
print(df["source"].value_counts())
df.sample(10)
```

Expected: 49,792 rows; positive 36,889, negative 8,759, neutral 4,144; evertink.lt 46,312, atsiliepimai.lt 3,480.

Write down 2–3 reviews where the label looks wrong (for the "label noise" part of the report).

## 3. Run the preparation

```python
!python prepare_data.py
```

What it does, step by step:

1. **Encoding:** puts Lithuanian letters (ą č ę ė į š ų ū ž) into one standard form and counts broken ones.
2. **Drop neutral:** our task is positive vs negative only.
3. **Clean the text:** lowercase, remove extra spaces, drop empty reviews.
4. **Duplicates:** if the same text has both labels, delete all copies; otherwise keep one copy. This way the same review can't be in both train and test.
5. **Personal data:** replace emails, phone numbers and links with `<email>`, `<phone>` and `<url>`.
6. **Negation check:** how many reviews still contain "ne" / "nėra" / *ne-* words. This is important for sentiment.
7. **Columns:** keep only text, label, source and store. We don't use `rating_normalized`, because it *is* the label (leakage).
8. **Split 80/10/10:** the same % of positive reviews and of each website in every split. Seed 42 means everyone gets the same split.
9. **Subsets:** 500 / 1000 / 2000 / 5000 training reviews, 3 random samples each, for the data-size experiment.

All numbers are saved in `data/stats.txt`. We use them in the report.

## 4. Check the result

```python
tr = pd.read_csv("data/train.csv")
va = pd.read_csv("data/val.csv")
te = pd.read_csv("data/test.csv")
print(len(tr), len(va), len(te))                   # about 80 / 10 / 10
print(tr.label.mean(), va.label.mean(), te.label.mean())   # about the same
```

## 5. Baselines

```python
!python baseline_tfidf.py
```

- **Majority:** always says "positive". It gets high accuracy (~0.81) but macro-F1 only ~0.45, because it never finds a negative review. That's why **macro-F1** is our main metric.
- **TF-IDF + logistic regression:** the number our PEFT models have to beat.
- It also trains on the 500 / 1000 / 2000 / 5000 subsets. These are the first points of the data-size plot.

## 6. Look at the errors

```python
err = pd.read_csv("results/tfidf_errors.csv")
err.sample(20)[["text", "sentiment"]]
```

For each error, write a short reason: wrong label, mixed review, negation, typo / no diacritics, too short, sarcasm. Count them. This is the start of our error analysis.

## 7. Save and share

Download the `data/` and `results/` folders from Kaggle (Output panel) and upload them to GitHub, so everyone uses the same files.

## Checklist

- [ ] train / val / test files on GitHub
- [ ] `data/stats.txt` numbers copied to our notes
- [ ] baseline results on val
- [ ] 20 errors looked at
