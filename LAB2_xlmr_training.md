# Lab 2: Train XLM-R (frozen head, LoRA, (IA)³)

Goal: train XLM-RoBERTa-base on our data with 3 adaptation methods and get the first rows of the **parameter-efficiency table**.
Where: Kaggle notebook with a **GPU**. Time: about 1–1.5 hours, most of it waiting for training.

New files: `train_xlmr.py`, `make_table.py` (put them in the repo next to the other scripts).

## The 3 methods in plain words

| Method | What is trained | Why we need it |
|---|---|---|
| `head` | Only a small classification layer on top. XLM-R itself stays **frozen** | This is our **"unadapted model"** baseline. XLM-R can't classify without a head, so the fairest "no adaptation" is: don't change XLM-R, only learn the head |
| `lora` | Small extra matrices inside XLM-R (+ head) | Our **main PEFT method** |
| `ia3` | Tiny scaling vectors inside XLM-R (+ head), even fewer parameters than LoRA | Our **second PEFT method** (the brief asks for at least two) |
| `full` *(optional)* | All 278M parameters | Expensive upper reference, only if we have time |

## 1. Kaggle setup

1. Open your notebook. In the right panel: **Accelerator → GPU T4 x2** (the script uses only one), **Internet → On**.
2. Get the latest code and data:

```python
import os
if os.path.exists("/kaggle/working/lt-sentiment-peft"):
    %cd /kaggle/working/lt-sentiment-peft
    !git pull
else:
    %cd /kaggle/working
    !git clone https://github.com/bOmbardirococcodrillo/lt-sentiment-peft.git
    %cd lt-sentiment-peft
!pip install -q peft
!nvidia-smi --query-gpu=name,memory.total --format=csv
!ls data
```

You should see `Tesla T4` and the files `train.csv val.csv test.csv subsets.json`.

## 2. Quick test run (2–3 min)

Before the long runs, check that everything works on a small run:

```python
!python train_xlmr.py --method lora --train_size 500 --epochs 1
```

Check the output:
- `device: Tesla T4` (if it says CPU, the GPU is not on)
- `trainable params: ... (0.xx%)`: LoRA trains well under 1% of the model
- `latency: ...` and `saved to results/xlmr_runs.csv` at the end

The F1 of this test doesn't matter (1 epoch, 500 reviews). Then delete this test row so it doesn't end up in the table:

```python
import pandas as pd
pd.read_csv("results/xlmr_runs.csv").iloc[0:0].to_csv("results/xlmr_runs.csv", index=False)
```

## 3. The three main runs (full training data)

Run them one after another, each in its own cell:

```python
!python train_xlmr.py --method head
```
```python
!python train_xlmr.py --method lora --rank 8 --targets qv
```
```python
!python train_xlmr.py --method ia3
```

Each run trains for 3 epochs on 33,028 reviews and prints the validation macro-F1 after each epoch. It keeps the **best epoch**.

**Write down the training time of the first run.** It tells us how long the ablations will take. (We expect roughly 5–20 minutes per run on a T4, but measure it.)

> [!warning] Save after every session
> Kaggle deletes `/kaggle/working` when the session ends. After the runs: `!zip -r results.zip results`, download it, and push it to GitHub (same as in Lab 1).

## 4. Make the table

```python
!python make_table.py
from IPython.display import Image
Image("results/data_size_plot.png")
```

This creates `results/main_table.md`:

| Column | Meaning |
|---|---|
| Trainable params / % of model | How many parameters we actually train |
| Macro-F1, F1 (neg) | Quality on the validation set |
| GPU mem (GB) | Peak GPU memory during training |
| Train time (min) | Training time |
| Latency (ms/review) | Prediction time for one review |
| F1 gain vs head | How much better than the unadapted model |
| Gain per M params | F1 gain divided by trainable parameters in millions (the brief's "performance per million trainable parameters") |

## 5. What to look at

- **Is LoRA better than TF-IDF (0.912)?** If not, that's also a finding (lemmatized text helps TF-IDF). Write it down.
- **Head vs LoRA vs (IA)³:** how much does adapting the inside of XLM-R help, compared with only training the head?
- **Memory:** head-only should need less GPU memory than LoRA (no gradients through XLM-R).
- **Per website:** printed after each run.

> [!note] If LoRA looks weak
> The learning rates are common default values, not tuned. If LoRA gets a clearly lower F1 than TF-IDF, try 2 other values **on the validation set**: `--lr 2e-4` and `--lr 1e-3`. Keep the best one for all later runs, and write in the report that we tuned it on validation.

## 6. Next (Lab 3): the ablations, same script

**LoRA rank and target modules** (full data):
```python
for r in [4, 8, 16, 32]:
    !python train_xlmr.py --method lora --rank {r} --targets qv
for t in ["attn", "all"]:
    !python train_xlmr.py --method lora --rank 8 --targets {t}
```

**Training-data size** (3 seeds each):
```python
for n in ["500", "1000", "2000", "5000"]:
    for s in [1, 2, 3]:
        for m in ["head", "lora", "ia3"]:
            !python train_xlmr.py --method {m} --train_size {n} --seed {s}
```
That's 36 small runs, but small data trains fast. Then `!python make_table.py` draws the **data-size plot**: TF-IDF vs head vs LoRA vs (IA)³.

> [!tip] Long runs without keeping the browser open
> Put all commands in one cell, then click **Save Version → Save & Run All (Commit)**. Kaggle runs the whole notebook in the background (up to 12 h). The `results/` folder is then in the version's **Output** tab.

## Common problems

| Problem | Fix |
|---|---|
| `device: CPU (slow!)` | Accelerator is not set to GPU, or GPU quota used up (Kaggle gives ~30 h/week) |
| `CUDA out of memory` | Add `--batch_size 16` |
| `No module named 'peft'` | Run `!pip install -q peft` |
| Loss becomes `nan` | Learning rate too high, try `--lr 2e-4` |
| `KeyError` with `--train_size` | Use exactly `500`, `1000`, `2000` or `5000` |
| Results disappeared | Session ended. Always download/push `results/` after running |

## Checklist

- [ ] Test run worked on GPU
- [ ] 3 main runs done: head, LoRA (r=8, qv), (IA)³
- [ ] `results/main_table.md` created
- [ ] Training time per run written down
- [ ] `results/` pushed to GitHub
