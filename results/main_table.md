# Main results (validation set, full training data)

| Method | Trainable params | % of model | Macro-F1 | F1 (neg) | GPU mem (GB) | Train time (min) | Latency (ms/review) | F1 gain vs head | Gain per M params |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LR | - | - | 0.912 | 0.862 | CPU | - | - | - | - |
| head (lr 0.001, 6 ep) | 592,130 | 0.21% | 0.849 ± 0.003 | 0.766 | 1.24 | 7.2 | 8.9 | +0.000 | - |
| ia3 (lr 0.001, 6 ep) | 647,426 | 0.23% | 0.892 ± 0.004 | 0.834 | 3.03 | 18.2 | 11.7 | +0.043 | 0.0671 |
| lora r=8 qv (lr 0.0005, 6 ep) | 887,042 | 0.32% | 0.898 ± 0.001 | 0.842 | 2.47 | 15.4 | 8.9 | +0.049 | 0.0551 |
