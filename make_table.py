# Makes the main results table and the data-size plot from all runs
#   results/main_table.md      parameter-efficiency table (full training data)
#   results/data_size_plot.png macro-F1 vs number of training reviews
#
# Run after the training runs:  python make_table.py
# If your best LoRA setting is not rank 8 / qv:  python make_table.py --lora_rank 16 --lora_targets all

import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--lora_targets", default="qv")
args = parser.parse_args()

runs = pd.read_csv("results/xlmr_runs.csv")
runs = runs[runs["split"] == "val"]
tfidf = pd.read_csv("results/baselines.csv")
full_size = runs["train_size"].max()

# ---------- 1. parameter-efficiency table (full training data, mean over seeds)
full = runs[runs["train_size"] == full_size].fillna({"rank": "", "targets": ""})
table = full.groupby(["method", "rank", "targets", "lr", "epochs"]).agg(
    runs=("seed", "count"), macro_f1=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
    accuracy=("accuracy", "mean"), f1_negative=("f1_negative", "mean"),
    trainable_params=("trainable_params", "first"), trainable_pct=("trainable_pct", "first"),
    peak_mem_gb=("peak_mem_gb", "mean"), train_time_min=("train_time_min", "mean"),
    latency_ms=("latency_ms_bs1", "mean"),
).reset_index()

# gain over the unadapted model (frozen XLM-R + head) per million trainable parameters
head_f1 = table.loc[table["method"] == "head", "macro_f1"]
if len(head_f1):
    table["gain_vs_head"] = table["macro_f1"] - head_f1.iloc[0]
    table["gain_per_M_params"] = table["gain_vs_head"] / (table["trainable_params"] / 1e6)

tfidf_full = tfidf[(tfidf["model"] == "tfidf") & (tfidf["train_size"] == tfidf["train_size"].max())]
lines = ["# Main results (validation set, full training data)", "",
         "| Method | Trainable params | % of model | Macro-F1 | F1 (neg) | GPU mem (GB) | Train time (min) | Latency (ms/review) | F1 gain vs head | Gain per M params |",
         "|---|---|---|---|---|---|---|---|---|---|"]
if len(tfidf_full):
    lines.append(f"| TF-IDF + LR | - | - | {tfidf_full['macro_f1'].iloc[0]:.3f} | {tfidf_full['f1_negative'].iloc[0]:.3f} | CPU | - | - | - | - |")
for _, r in table.iterrows():
    name = r["method"] + (f" r={int(r['rank'])} {r['targets']}" if r["method"] == "lora" else "")
    name += f" (lr {r['lr']:g}, {int(r['epochs'])} ep)"
    f1 = f"{r['macro_f1']:.3f}" + (f" ± {r['f1_std']:.3f}" if r["runs"] > 1 else "")
    gain = f"{r['gain_vs_head']:+.3f}" if "gain_vs_head" in r else "-"
    per_m = f"{r['gain_per_M_params']:.4f}" if "gain_per_M_params" in r and r["method"] != "head" else "-"
    lines.append(f"| {name} | {int(r['trainable_params']):,} | {r['trainable_pct']:.2f}% | {f1} | {r['f1_negative']:.3f} "
                 f"| {r['peak_mem_gb']:.2f} | {r['train_time_min']:.1f} | {r['latency_ms']:.1f} | {gain} | {per_m} |")
open("results/main_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------- 2. data-size plot
plt.figure(figsize=(7, 4.5))


def plot_line(df, label, style):
    g = df.groupby("train_size")["macro_f1"].agg(["mean", "std"]).reset_index().fillna(0)
    plt.errorbar(g["train_size"], g["mean"], yerr=g["std"], fmt=style, capsize=3, label=label)


plot_line(tfidf[tfidf["model"] == "tfidf"], "TF-IDF + LR", "o--")
plot_line(runs[runs["method"] == "head"], "XLM-R frozen + head", "s-")
lora = runs[(runs["method"] == "lora") & (runs["rank"] == args.lora_rank) & (runs["targets"] == args.lora_targets)]
plot_line(lora, f"XLM-R LoRA (r={args.lora_rank}, {args.lora_targets})", "^-")
plot_line(runs[runs["method"] == "ia3"], "XLM-R (IA)³", "d-")

plt.xscale("log")
sizes = sorted(set(tfidf["train_size"]) | set(runs["train_size"]))
plt.xticks(sizes, [f"{s:,}" for s in sizes])
plt.minorticks_off()
plt.xlabel("Number of training reviews (log scale)")
plt.ylabel("Macro-F1 (validation)")
plt.title("Performance vs training-data size")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/data_size_plot.png", dpi=200)
print("\nsaved results/main_table.md and results/data_size_plot.png")
