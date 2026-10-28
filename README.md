# Lithuanian review sentiment with PEFT (NLP Project 5)

Dataset: engineerai/lithuanian-estore-sentiment-binary (Hugging Face)

## How to run

We run everything on Kaggle (Internet on).

```
pip install -r requirements.txt
python prepare_data.py     # makes data/train.csv, val.csv, test.csv
python baseline_tfidf.py   # majority + TF-IDF baselines on the validation set
```

## Files

- `prepare_data.py`: cleaning and the 80/10/10 split
- `baseline_tfidf.py`: first baselines
- `data/`: the splits (everyone uses the same files)
- `results/`: results of the runs
