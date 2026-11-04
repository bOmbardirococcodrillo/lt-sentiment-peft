# Lab 3: Human vs human + synthetic data

Goal: the last required ablation. Does adding **synthetic** (AI-written) Lithuanian reviews help when we have only a little real data?
Where: Kaggle notebook with **GPU** and **Internet on**. Time: about 1 hour (≈30 min generating, ≈30 min training).

New file: `generate_synthetic.py`. Updated: `train_xlmr.py` (new `--synthetic` option), `make_table.py`, `requirements.txt`, `README.md`.

## The idea in plain words

1. An open-weight language model (**EuroLLM-1.7B-Instruct**, made for European languages) writes ~3,500 short positive and negative shop reviews in Lithuanian. It runs on Kaggle, so no data goes to an external AI service (brief, point E5).
2. Our real reviews are lemmatized and have no stopwords. So we **process the synthetic reviews the same way**. Otherwise the model could learn "real looks like X, synthetic looks like Y" instead of learning sentiment.
3. We train LoRA (r=8, q+v, the same setting as in the data-size plot) on:

| Setting | Real reviews | Synthetic reviews |
|---|---|---|
| Human only | 1,000 | 0 (we already have these runs from Lab 2) |
| Human + synthetic 1× | 1,000 | 1,000 |
| Human + synthetic 2× | 1,000 | 2,000 |

We use **1,000 real reviews**, because synthetic data is meant to help when real data is scarce. With 33k real reviews, it would make no visible difference. Validation and test stay **100% real**.

## 1. Setup

```python
import os
if os.path.exists("/kaggle/working/lt-sentiment-peft"):
    %cd /kaggle/working/lt-sentiment-peft
    !git pull
else:
    %cd /kaggle/working
    !git clone https://github.com/bOmbardirococcodrillo/lt-sentiment-peft.git
    %cd lt-sentiment-peft
!pip install -q peft simplemma
!pip uninstall -y -q torchao
!wc -l results/xlmr_runs.csv
```
The last line should show **53** (all runs so far, including the "all linear" seeds). If it's lower, push your latest `results/` first.

## 2. Generate the synthetic reviews (≈20–30 min)

```python
!python generate_synthetic.py
```

What it does:
1. **Generates** 2,600 positive + 900 negative reviews about different products (phones, shoes, car mats, insurance…) and aspects (delivery, quality, service, packaging…), so they aren't all the same.
2. **Filters:** removes too short/too long reviews, English answers, duplicates, and nonsense (reviews where most words don't exist anywhere in our real data).
3. **Processes them like our dataset:** lowercase, no punctuation, stopwords removed, unknown word forms lemmatized with `simplemma`. The stopwords it removes are chosen **from our data**: only words the dataset author clearly removed (they never appear in the real reviews).
4. **Never** keeps a synthetic review identical to a real one (train, val or test).

**Look at the output and write it down for the report:**
- how many reviews survived each filter
- the comparison line: real vs synthetic (average length, % of words known in real data, % with negation)
- the 6 examples (raw → processed). Do they look like real reviews?

> [!warning] If something looks wrong
> - Fewer than ~1,600 positive or ~430 negative reviews left: training with 2,000 synthetic reviews will fail. Run again with more, e.g. `--n_pos 3500 --n_neg 1200`.
> - Many reviews are in English or nonsense: tell us, we'll change the prompt or the model.

## 3. Manual check by a native speaker (≈20 min, no GPU)

Download `data/synthetic_spotcheck.csv` (50 random reviews). For each one, fill in:
- `grammar_ok_1_0`: is the *original* text (`raw_text`) correct, natural Lithuanian? 1 or 0
- `label_ok_1_0`: is it really positive/negative, as the `sentiment` column says? 1 or 0
- `comment`: what's wrong, e.g. "wrong case ending", "sounds like a translation", "too generic", "it's actually neutral"

Then count: *"X% grammatical, Y% correct label"*. This goes into the report and the limitations section.

## 4. Train with synthetic data (≈30 min)

```python
for s in [1, 2, 3]:
    for syn in [1000, 2000]:
        !python train_xlmr.py --method lora --rank 8 --targets qv --train_size 1000 --synthetic {syn} --seed {s}
!python make_table.py
```

- Each seed uses the **same 1,000 real reviews** as in the data-size runs, plus a different random sample of synthetic reviews.
- Synthetic reviews are added with the **same positive/negative ratio** as the real ones (~79/21).
- The best epoch is still chosen on the **real** validation set.

`make_table.py` now also creates:
- `results/synthetic_table.md`: 0 / 1,000 / 2,000 synthetic, mean ± std over 3 seeds
- `results/synthetic_plot.png`: the same as a figure

Synthetic runs are **not** mixed into the main table or the data-size plot.

## 5. Save

```python
!zip -r results.zip results data/synthetic.csv data/synthetic_raw.csv data/synthetic_spotcheck.csv
```
Download, unzip into your repo folder, and push. The synthetic files must be in the repo too, so the experiment can be repeated exactly.

## 6. How to read the result

Baseline from Lab 2: LoRA with 1,000 real reviews = **0.857 ± 0.017**.

| If… | It means… |
|---|---|
| F1 goes up clearly (more than the ± spread) | The synthetic reviews add useful examples when real data is scarce |
| F1 stays about the same | They don't add new information. The model already learns these simple patterns from 1,000 real reviews |
| F1 goes down | The synthetic reviews are too different from real ones (too clean, too generic, label noise). Check the spot-check results |

**Any of these is a valid result.** What matters is explaining it with your numbers: the spot-check %, the real-vs-synthetic comparison, and examples.

Good things to check for the error analysis: does synthetic data help on **negative** reviews (F1 neg)? Real negative reviews are often long stories ("ordered, waited, called, nobody answered…"), while synthetic ones are usually short and obvious.

## Common problems

| Problem | Fix |
|---|---|
| `No module named 'simplemma'` | `!pip install -q simplemma` |
| `ValueError: Cannot take a larger sample than population` | Not enough synthetic reviews of one class. Generate more (`--n_pos`, `--n_neg`) |
| `CUDA out of memory` during generation | `!python generate_synthetic.py --batch_size 16` |
| Generation is very slow / `device` is CPU | GPU is not on |
| `FileNotFoundError: data/synthetic.csv` in training | Run `generate_synthetic.py` first (in the same session), or pull it from GitHub |

## Checklist

- [ ] Synthetic reviews generated, filter numbers and comparison written down
- [ ] 50 reviews checked by a native speaker
- [ ] 6 training runs done (1,000 / 2,000 synthetic × 3 seeds)
- [ ] `synthetic_table.md` + `synthetic_plot.png` created
- [ ] Results and synthetic files pushed to GitHub