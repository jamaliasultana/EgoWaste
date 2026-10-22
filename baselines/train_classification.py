#!/usr/bin/env python3
"""
train_classification.py
==========================
Benchmarks ResNet-50, ConvNeXt-Tiny, DenseNet-121, Swin-Tiny and ViT-Base on
ACTIVE (handheld) OBJECT classification in cluttered egocentric waste frames.

WHY THE PAPER'S NUMBERS ARE LOW, AND WHAT IS DONE ABOUT IT
----------------------------------------------------------
EgoWaste is strongly long-tailed, which is why the submitted baselines show
56.1% accuracy against 10.8% macro-F1. Accuracy alone rewards predicting the
head classes. This script therefore:

  * splits by VIDEO, never randomly (frames of one sort are near-duplicates)
  * offers class-balanced sampling and logit-adjusted loss for the tail
  * reports top-1, top-5, macro-F1 AND balanced accuracy together
  * selects the best epoch on validation macro-F1, not accuracy
  * uses a modern recipe: AdamW, cosine schedule with warmup, label smoothing,
    RandAugment, mixed precision, and horizontal-flip test-time augmentation

Everything is per-model configurable, and every run writes a row into one
results table so the five backbones and the input modes are directly
comparable.

USAGE
  # single model, one input mode
  python train_classification.py --data_dir ./active_obj_data \
      --mode crop_ctx --model resnet50 --label_level material

  # full benchmark: 5 models x chosen modes
  python train_classification.py --data_dir ./active_obj_data \
      --mode crop_ctx --all_models --label_level material --epochs 30

REQUIREMENTS
  pip install torch torchvision timm scikit-learn pandas
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image
from sklearn.metrics import f1_score, balanced_accuracy_score, top_k_accuracy_score

MODELS = {
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "densenet121": "densenet121",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "vit_base": "vit_base_patch16_224",
}

# transformers need a gentler learning rate than convnets
DEFAULT_LR = {
    "resnet50": 3e-4, "densenet121": 3e-4,
    "convnext_tiny": 2e-4, "swin_tiny": 1e-4, "vit_base": 5e-5,
}

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================
# DATA
# ============================================================
class ActiveObjectDataset(Dataset):
    def __init__(self, df, data_dir, path_col, label_col, classes, train):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.path_col = path_col
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.label_col = label_col
        self.train = train

        from torchvision import transforms as T
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
                T.RandomHorizontalFlip(),
                T.RandAugment(num_ops=2, magnitude=7),
                T.ColorJitter(0.3, 0.3, 0.3, 0.05),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
                T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(IMG_SIZE * 1.14)),
                T.CenterCrop(IMG_SIZE),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.data_dir, r[self.path_col])).convert("RGB")
        return self.tf(img), self.cls_to_idx[r[self.label_col]]


def video_level_split(df, val_frac=0.15, test_frac=0.15, seed=2026):
    """
    Split by video so no object appears in two splits.

    Videos are processed largest-first and each is placed in the split with the
    most REMAINING CAPACITY in absolute samples. Using absolute rather than
    relative deficit matters: at the start every split is equally empty in
    relative terms, so a relative rule can drop the single largest video into
    the test set and wreck the proportions.
    """
    sizes = df.groupby("video").size().sort_values(ascending=False)
    total = float(sizes.sum())
    quota = {"train": (1 - val_frac - test_frac) * total,
             "val": val_frac * total, "test": test_frac * total}
    got = {k: 0.0 for k in quota}
    assign = {}
    rng = random.Random(seed)

    for video, n in sizes.items():
        room = {k: quota[k] - got[k] for k in quota}
        best = max(room, key=lambda k: (room[k], rng.random()))
        assign[video] = best
        got[best] += float(n)

    # every split needs at least one video; steal the smallest from the largest
    for k in ("val", "test"):
        if got[k] == 0:
            donor = max(got, key=lambda z: got[z])
            cand = [v for v, s in assign.items() if s == donor]
            if len(cand) > 1:
                smallest = min(cand, key=lambda v: sizes[v])
                assign[smallest] = k
                got[k] += float(sizes[smallest])
                got[donor] -= float(sizes[smallest])

    df = df.copy()
    df["split"] = df["video"].map(assign)
    return df


# ============================================================
# LOSS
# ============================================================
class LogitAdjustedLoss(nn.Module):
    """
    Menon et al., 'Long-tail learning via logit adjustment' (ICLR 2021).
    Subtracts tau*log(prior) from the logits, which lifts rare classes without
    resampling. Usually the single most effective change on a long-tailed set.
    """
    def __init__(self, counts, tau=1.0, smoothing=0.1):
        super().__init__()
        p = torch.tensor(counts, dtype=torch.float32)
        p = p / p.sum()
        self.register_buffer("adj", tau * torch.log(p + 1e-12))
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.adj, target,
                               label_smoothing=self.smoothing)


def build_loss(kind, counts, smoothing):
    if kind == "logit_adjust":
        return LogitAdjustedLoss(counts, tau=1.0, smoothing=smoothing)
    if kind == "balanced_ce":
        w = 1.0 / np.power(np.asarray(counts, dtype=np.float64), 0.5)
        w = w / w.mean()
        return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32),
                                   label_smoothing=smoothing)
    return nn.CrossEntropyLoss(label_smoothing=smoothing)


# ============================================================
# TRAIN / EVAL
# ============================================================
def evaluate(model, loader, device, n_classes, tta=True):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                p = F.softmax(model(x).float(), dim=1)
                if tta:
                    p = p + F.softmax(model(torch.flip(x, dims=[3])).float(), dim=1)
                    p = p / 2
            probs.append(p.cpu()); ys.append(y)
    probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()
    pred = probs.argmax(1)
    out = {
        "top1": float((pred == ys).mean()),
        "macro_f1": float(f1_score(ys, pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(ys, pred)),
        "weighted_f1": float(f1_score(ys, pred, average="weighted", zero_division=0)),
    }
    try:
        out["top5"] = float(top_k_accuracy_score(
            ys, probs, k=min(5, n_classes), labels=list(range(n_classes))))
    except Exception:
        out["top5"] = float("nan")
    return out, pred, ys


def run_one(args, df, classes, path_col, label_col, model_key, device):
    import timm

    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]

    ds_tr = ActiveObjectDataset(tr, args.data_dir, path_col, label_col, classes, True)
    ds_va = ActiveObjectDataset(va, args.data_dir, path_col, label_col, classes, False)
    ds_te = ActiveObjectDataset(te, args.data_dir, path_col, label_col, classes, False)

    counts = np.array([max((tr[label_col] == c).sum(), 1) for c in classes],
                      dtype=np.float64)

    if args.sampler == "balanced":
        # sqrt-inverse frequency: full inverse over-corrects and destabilises
        w_cls = 1.0 / np.sqrt(counts)
        w = tr[label_col].map({c: w_cls[i] for i, c in enumerate(classes)}).values
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(tr), replacement=True)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                           num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=True)

    dl_va = DataLoader(ds_va, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    # model = timm.create_model(MODELS[model_key], pretrained=True,
    #                           num_classes=len(classes),
    #                           drop_path_rate=args.drop_path).to(device)

        # not every architecture supports stochastic depth -- DenseNet, for one,
    # has no drop_path implementation, so only pass it when it is accepted
    kw = {}
    if args.drop_path > 0 and model_key not in ("densenet121",):
        kw["drop_path_rate"] = args.drop_path
    try:
        model = timm.create_model(MODELS[model_key], pretrained=True,
                                  num_classes=len(classes), **kw).to(device)
    except TypeError:
        model = timm.create_model(MODELS[model_key], pretrained=True,
                                  num_classes=len(classes)).to(device)
    
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    crit = build_loss(args.loss, counts, args.smoothing).to(device)
    lr = args.lr or DEFAULT_LR[model_key]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    steps = max(len(dl_tr), 1) * args.epochs
    warm = max(len(dl_tr), 1) * args.warmup_epochs

    def lr_at(step):
        if step < warm:
            return step / max(warm, 1)
        p = (step - warm) / max(steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = {"macro_f1": -1}
    best_state, gstep = None, 0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            gstep += 1
            tot += loss.item()

        vm, _, _ = evaluate(model, dl_va, device, len(classes), tta=False)
        print(f"    ep {ep+1:>2}/{args.epochs}  loss {tot/max(len(dl_tr),1):.3f}  "
              f"val top1 {vm['top1']:.3f}  macroF1 {vm['macro_f1']:.3f}")
        # select on macro-F1: accuracy alone would favour head classes
        if vm["macro_f1"] > best["macro_f1"]:
            best = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    tm, pred, ys = evaluate(model, dl_te, device, len(classes), tta=True)

    row = {"model": model_key, "mode": args.mode, "label_level": args.label_level,
           "loss": args.loss, "sampler": args.sampler, "lr": lr,
           "epochs": args.epochs, "n_classes": len(classes),
           "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "minutes": (time.time() - t0) / 60.0}
    row.update({f"val_{k}": v for k, v in best.items()})
    row.update({f"test_{k}": v for k, v in tm.items()})

    if args.save_preds:
        pd.DataFrame({"y": ys, "pred": pred,
                      "y_name": [classes[i] for i in ys],
                      "pred_name": [classes[i] for i in pred]}).to_csv(
            os.path.join(args.out_dir,
                         f"preds_{model_key}_{args.mode}_{args.label_level}.csv"),
            index=False)
    return row


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./active_obj_data")
    ap.add_argument("--mode", default="crop_ctx",
                    choices=["crop", "crop_ctx", "masked", "frame"])
    ap.add_argument("--label_level", default="material",
                    choices=["material", "waste_class"])
    ap.add_argument("--model", default="resnet50", choices=list(MODELS))
    ap.add_argument("--all_models", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--loss", default="logit_adjust",
                    choices=["ce", "balanced_ce", "logit_adjust"])
    ap.add_argument("--sampler", default="none", choices=["none", "balanced"])
    ap.add_argument("--min_class_count", type=int, default=10,
                    help="drop classes with fewer samples than this")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--channels_last", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--save_preds", action="store_true", default=True)
    ap.add_argument("--out_dir", default="./active_obj_results")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_csv(os.path.join(args.data_dir, "index.csv"))
    path_col = f"path_{args.mode}"
    if path_col not in idx.columns:
        raise SystemExit(f"mode '{args.mode}' not in index.csv; rebuild with it")
    idx = idx[idx[path_col].notna()].copy()

    label_col = "material" if args.label_level == "material" else "waste_class"

    vc = idx[label_col].value_counts()
    keep = vc[vc >= args.min_class_count].index
    dropped = int((~idx[label_col].isin(keep)).sum())
    idx = idx[idx[label_col].isin(keep)].copy()
    classes = sorted(keep.tolist())

    idx = video_level_split(idx)

    # a class missing from train cannot be learned; drop it everywhere
    tr_classes = set(idx[idx.split == "train"][label_col])
    missing = [c for c in classes if c not in tr_classes]
    if missing:
        idx = idx[idx[label_col].isin(tr_classes)].copy()
        classes = sorted(tr_classes)

    print("=" * 78)
    print(f"Active-object classification | mode={args.mode} | "
          f"labels={args.label_level}")
    print("=" * 78)
    print(f"  samples {len(idx)}  classes {len(classes)}  "
          f"(dropped {dropped} samples in rare classes"
          f"{', ' + str(len(missing)) + ' classes absent from train' if missing else ''})")
    for s in ("train", "val", "test"):
        sub = idx[idx.split == s]
        print(f"  {s:<5s} {len(sub):>6d} samples  {sub['video'].nunique():>2d} videos")
    print(f"  loss={args.loss}  sampler={args.sampler}  device={device}\n")

    idx.to_csv(os.path.join(args.out_dir, f"split_{args.mode}_{args.label_level}.csv"),
               index=False)

    to_run = list(MODELS) if args.all_models else [args.model]
    rows = []
    for mk in to_run:
        print(f"  === {mk} ===")
        try:
            rows.append(run_one(args, idx, classes, path_col, label_col, mk, device))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [OOM] {mk} skipped -- lower --batch_size")
                torch.cuda.empty_cache()
                continue
            raise

    res = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir,
                           f"results_{args.mode}_{args.label_level}.csv")
    if os.path.isfile(out_csv):
        res = pd.concat([pd.read_csv(out_csv), res], ignore_index=True)
    res.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print(f"RESULTS  mode={args.mode}  labels={args.label_level}")
    print("=" * 78)
    show = ["model", "test_top1", "test_top5", "test_macro_f1",
            "test_balanced_acc", "minutes"]
    print(res[[c for c in show if c in res.columns]].round(3).to_string(index=False))
    print("\n  Report macro-F1 and balanced accuracy alongside top-1: the class")
    print("  distribution is long-tailed, so top-1 alone overstates performance.")
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()
