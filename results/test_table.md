# FINAL results (test set, full training data)

| Method | Trainable params | Runs | Macro-F1 | Accuracy | F1 (neg) |
|---|---|---|---|---|---|
| Majority class | - | 1 | 0.441 | 0.789 | 0.000 |
| TF-IDF + LR | - | 1 | 0.908 | 0.938 | 0.855 |
| XLM-R frozen + head (unadapted) | 592,130 | 3 | 0.847 ± 0.010 | 0.893 | 0.764 |
| XLM-R (IA)³ | 647,426 | 3 | 0.894 ± 0.003 | 0.925 | 0.836 |
| XLM-R LoRA r=8 all | 1,919,234 | 3 | 0.906 ± 0.003 | 0.936 | 0.854 |
| XLM-R LoRA r=8 qv | 887,042 | 3 | 0.904 ± 0.001 | 0.933 | 0.852 |
