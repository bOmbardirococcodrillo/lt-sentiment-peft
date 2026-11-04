# Lithuanian review sentiment with PEFT (NLP Project 5)

Dataset: engineerai/lithuanian-estore-sentiment-binary (Hugging Face)
Model: XLM-RoBERTa-base (FacebookAI/xlm-roberta-base)

## How to run

We run everything on Kaggle (Internet on, GPU T4 for the XLM-R part).

```
pip install -r requirements.txt
python prepare_data.py                              # data/train.csv, val.csv, test.csv
python baseline_tfidf.py                            # majority + TF-IDF baselines (CPU)
python train_xlmr.py --method head                  # frozen XLM-R + head (unadapted baseline)
python train_xlmr.py --method lora --rank 8 --targets qv
python train_xlmr.py --method ia3
python generate_synthetic.py                        # synthetic reviews -> data/synthetic.csv (GPU)
python train_xlmr.py --method lora --train_size 1000 --synthetic 1000
python make_table.py                                # main_table.md, data_size_plot.png, synthetic_table.md
```

Options of `train_xlmr.py`: `--method head|lora|ia3|full`, `--rank`, `--targets qv|attn|all`,
`--train_size 500|1000|2000|5000|full`, `--synthetic N`, `--seed`, `--epochs`, `--lr`, `--test` (final results only).

## Files

- `prepare_data.py`: cleaning and the 80/10/10 split
- `baseline_tfidf.py`: majority and TF-IDF baselines
- `train_xlmr.py`: XLM-R training with different adaptation methods
- `generate_synthetic.py`: synthetic reviews from an open-weight LLM (EuroLLM-1.7B-Instruct), processed like the dataset
- `make_table.py`: main results table, data-size plot, synthetic-data table and plot
- `data/`: the splits (everyone uses the same files)
- `results/`: all runs (`xlmr_runs.csv` is our experiment log)