# Train XLM-RoBERTa for Lithuanian sentiment with different adaptation methods
#   head : frozen XLM-R, only the classification head is trained ("unadapted model" baseline)
#   lora : LoRA adapters + head
#   ia3  : (IA)3 + head (our second PEFT method)
#   full : full fine-tuning (optional, expensive reference)
#
# Every run adds one row to results/xlmr_runs.csv with:
# macro-F1, accuracy, trainable params, GPU memory, training time, latency
#
# Examples:
#   python train_xlmr.py --method head
#   python train_xlmr.py --method lora --rank 8 --targets qv
#   python train_xlmr.py --method ia3
#   python train_xlmr.py --method lora --train_size 500 --seed 2

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--method", default="lora", choices=["head", "lora", "ia3", "full"])
parser.add_argument("--rank", type=int, default=8)                       # LoRA rank
parser.add_argument("--targets", default="qv", choices=["qv", "attn", "all"])  # where LoRA goes
parser.add_argument("--train_size", default="full")                      # full, 500, 1000, 2000, 5000
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
parser.add_argument("--test", action="store_true")                       # only for the final results!
args = parser.parse_args()

# default settings per method (we tuned nothing yet, these are common values)
DEFAULT_LR = {"head": 1e-3, "lora": 5e-4, "ia3": 3e-3, "full": 2e-5}
lr = args.lr or DEFAULT_LR[args.method]
epochs = args.epochs or (3 if args.train_size == "full" else 10)  # small data needs more epochs

# where LoRA is added inside each transformer layer (regular expressions on layer names)
LORA_TARGETS = {
    "qv": r".*encoder.*\.(query|value)$",                                   # query + value
    "attn": r".*encoder.*\.(query|key|value|attention\.output\.dense)$",    # all attention
    "all": r".*encoder.*\.(query|key|value|dense)$",                        # attention + feed-forward
}

run_name = args.method
if args.method == "lora":
    run_name += f"_r{args.rank}_{args.targets}"
run_name += f"_n{args.train_size}_s{args.seed}"
print("run:", run_name, "| lr:", lr, "| epochs:", epochs)

# ---------------- reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU (slow!)")

# ---------------- data
train = pd.read_csv("data/train.csv")
if args.train_size != "full":
    subsets = json.load(open("data/subsets.json"))
    train = train[train["id"].isin(subsets[str(args.seed)][args.train_size])]
val = pd.read_csv("data/val.csv")
evalset = pd.read_csv("data/test.csv") if args.test else val
print(f"train: {len(train)} reviews | {'test' if args.test else 'val'}: {len(evalset)} reviews")

tokenizer = AutoTokenizer.from_pretrained(args.model)


def batches(df, batch_size, shuffle):
    order = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    for i in range(0, len(df), batch_size):
        part = df.iloc[order[i:i + batch_size]]
        enc = tokenizer(part["text"].tolist(), truncation=True, max_length=args.max_len,
                        padding=True, return_tensors="pt")
        yield enc.to(device), torch.tensor(part["label"].values, device=device)


# ---------------- model
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
total_params = sum(p.numel() for p in model.parameters())

if args.method == "head":
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier")
elif args.method == "lora":
    from peft import LoraConfig, TaskType, get_peft_model
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank,
                        lora_dropout=0.1, target_modules=LORA_TARGETS[args.targets])
    model = get_peft_model(model, config)  # the classification head is trained too
elif args.method == "ia3":
    from peft import IA3Config, TaskType, get_peft_model
    config = IA3Config(task_type=TaskType.SEQ_CLS,
                       target_modules=r".*encoder.*\.(key|value|layer\.\d+\.output\.dense)$",
                       feedforward_modules=r".*encoder.*layer\.\d+\.output\.dense$")
    model = get_peft_model(model, config)

model.to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {trainable_params:,} of {total_params:,} ({100 * trainable_params / total_params:.3f}%)")

# ---------------- training
# class weights, because only ~21% of reviews are negative
counts = train["label"].value_counts().sort_index().values
weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float, device=device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
steps = epochs * int(np.ceil(len(train) / args.batch_size))
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
use_fp16 = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)


def predict(df, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for enc, _ in batches(df, batch_size, shuffle=False):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                preds += model(**enc).logits.argmax(-1).tolist()
    return np.array(preds)


if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
start = time.time()
best_f1, best_epoch, best_weights = -1, 0, None

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for enc, labels in batches(train, args.batch_size, shuffle=True):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            loss = loss_fn(model(**enc).logits.float(), labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    # we always choose the best epoch on the VALIDATION set
    val_f1 = f1_score(val["label"], predict(val), average="macro")
    print(f"epoch {epoch}: loss {total_loss:.2f}, val macro-F1 {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1, best_epoch = val_f1, epoch
        best_weights = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

train_time = time.time() - start
peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else 0
model.load_state_dict(best_weights, strict=False)  # go back to the best epoch

# ---------------- evaluation
pred = predict(evalset)
acc = accuracy_score(evalset["label"], pred)
f1 = f1_score(evalset["label"], pred, average="macro")
f1_neg = f1_score(evalset["label"], pred, pos_label=0)
print(f"best epoch {best_epoch}: accuracy {acc:.4f}, macro-F1 {f1:.4f}, F1(negative) {f1_neg:.4f}")
for source in evalset["source"].unique():
    part = (evalset["source"] == source).values
    print(f"  {source}: macro-F1 {f1_score(evalset['label'][part], pred[part], average='macro'):.3f}")

os.makedirs("results", exist_ok=True)
evalset[evalset["label"] != pred].to_csv(f"results/errors_{run_name}.csv", index=False)

# ---------------- inference latency (ms per review)
if args.method == "lora":
    model = model.merge_and_unload()  # LoRA merged into the model = how it is used in practice
# (IA)3 is measured without merging (merging it fails for the XLM-R classification head)
model.eval()
texts = evalset["text"].tolist()[:200]


def latency(batch_size):
    with torch.no_grad():
        for i in range(0, 64, batch_size):  # warm-up
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=args.max_len,
                            padding=True, return_tensors="pt").to(device)
            model(**enc)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=args.max_len,
                            padding=True, return_tensors="pt").to(device)
            model(**enc)
        if device == "cuda":
            torch.cuda.synchronize()
    return 1000 * (time.time() - t) / len(texts)


lat1, lat32 = latency(1), latency(32)
print(f"latency: {lat1:.2f} ms/review (batch 1), {lat32:.2f} ms/review (batch 32)")
print(f"training time: {train_time / 60:.1f} min | peak GPU memory: {peak_mem:.2f} GB")

# ---------------- save one row
row = {
    "run": run_name, "method": args.method,
    "rank": args.rank if args.method == "lora" else "", "targets": args.targets if args.method == "lora" else "",
    "train_size": len(train), "seed": args.seed, "lr": lr, "epochs": epochs, "best_epoch": best_epoch,
    "trainable_params": trainable_params, "total_params": total_params,
    "trainable_pct": round(100 * trainable_params / total_params, 4),
    "peak_mem_gb": round(peak_mem, 2), "train_time_min": round(train_time / 60, 2),
    "latency_ms_bs1": round(lat1, 2), "latency_ms_bs32": round(lat32, 2),
    "split": "test" if args.test else "val", "accuracy": round(acc, 4),
    "macro_f1": round(f1, 4), "f1_negative": round(f1_neg, 4),
    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
}
path = "results/xlmr_runs.csv"
pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
print("saved to", path)
