#!/usr/bin/env python3
"""
finetune_zerowaste.py   --  STAGE 2 of 2
======================================
Fine-tunes on ZeroWaste, starting either from the EgoWaste-pretrained
checkpoint or from ImageNet, and evaluates on the ZeroWaste test split.

THE COMPARISON THIS SUPPORTS
----------------------------
  --init <ego checkpoint>   EgoWaste pretraining -> ZeroWaste fine-tuning
  --init imagenet           ZeroWaste only, the control

Both use the same architecture, schedule, and metric code, and both select the
epoch on the ZeroWaste validation split, so the difference between them is
attributable to the initialisation. Run BOTH; the pretrained number means
nothing without its control.

  ZeroWaste's published DeepLabv3+ baseline is 46.93 mIoU, obtained with a
  ResNet-101 backbone and the first three backbone stages frozen. Those are the
  defaults here so the control is comparable to their number.

SEED VARIANCE
-------------
Repeated runs of the same configuration have differed by several mIoU points in
this setting, which is larger than some of the effects being measured. Use
--seeds 1 2 3 to train each configuration several times; the script reports
mean and standard deviation, and a difference smaller than that spread should
not be claimed.

USAGE
  # control
  python finetune_zerowaste.py --init imagenet \
      --zero_root /path/to/splits_final_deblurred --epochs 20 --seeds 1 2 3

  # EgoWaste-pretrained
  python finetune_zerowaste.py --init ./w6_pretrain/ego_pretrained_resnet101.pt \
      --zero_root /path/to/splits_final_deblurred --epochs 20 --seeds 1 2 3
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from run_cross_dataset_matrix import (CLASSES, SegSet, load_zero, metrics,
                             EVAL_CLASSES_DEFAULT)
from pretrain_shared_materials import build_model


def run_once(args, zero, seed, device):
    import torch
    from torch.utils.data import DataLoader
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = build_model(args.encoder, args.freeze_stages, device)

    init_tag = "imagenet"
    if args.init and args.init.lower() != "imagenet":
        sd = torch.load(args.init, map_location="cpu")
        enc = sd.get("encoder")
        if enc and enc != args.encoder:
            raise SystemExit(f"checkpoint was trained with encoder '{enc}' but "
                             f"--encoder is '{args.encoder}'")
        src = sd.get("model", sd)
        tgt = model.state_dict()

        # The pretraining head may be a different width -- 16-way when
        # pretrained on all 15 EgoWaste materials, versus 5-way for ZeroWaste.
        # Transfer every tensor whose shape matches and leave the rest at their
        # ImageNet/random initialisation, so the encoder and decoder carry over
        # while the classifier is learned fresh.
        # --transfer controls how much of the pretrained network is reused.
        # DeepLabV3+'s decoder learns class-specific spatial priors: one
        # centred hand-held object in EgoWaste versus many scattered items on a
        # conveyor. Those priors can actively hurt, so 'encoder' re-initialises
        # the decoder and head and keeps only the backbone.
        transfer, shape_skip, scope_skip = {}, [], 0
        for k, v in src.items():
            if k not in tgt:
                continue
            if args.transfer == "encoder" and not k.startswith("encoder."):
                scope_skip += 1
                continue
            if tgt[k].shape == v.shape:
                transfer[k] = v
            else:
                shape_skip.append(k)
        missing, unexpected = model.load_state_dict(transfer, strict=False)
        n_src = len(src)
        print(f"    transferred {len(transfer)}/{n_src} tensors from "
              f"{os.path.basename(args.init)}")
        if shape_skip:
            print(f"    re-initialised {len(shape_skip)} tensor(s) of different "
                  f"shape (e.g. {shape_skip[0]})")
        if scope_skip:
            print(f"    --transfer encoder: decoder and head re-initialised "
                  f"({scope_skip} tensors skipped)")
        if sd.get("classes"):
            print(f"    pretrained on {len(sd['classes'])} classes -> "
                  f"fine-tuning on {len(CLASSES)}")
        init_tag = os.path.basename(args.init)
    else:
        print("    initialised from ImageNet (control)")

    # low-shot: a fixed random subset of ZeroWaste, identical across
    # initialisations for a given seed, so the comparison isolates the effect
    # of pretraining rather than which images were drawn
    train_items = zero["train"]
    if args.label_frac < 1.0:
        rng = random.Random(seed)
        k = max(args.batch_size, int(round(len(train_items) * args.label_frac)))
        train_items = rng.sample(train_items, min(k, len(train_items)))
        print(f"    low-shot: {len(train_items)}/{len(zero['train'])} "
              f"training images ({100*args.label_frac:.0f}%)")

    dl_tr = DataLoader(SegSet(train_items, args.size, True),
                       batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(SegSet(zero["val"] or zero["test"], args.size, False),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    dl_te = DataLoader(SegSet(zero["test"], args.size, False),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    def set_encoder_frozen(flag):
        for p in model.encoder.parameters():
            p.requires_grad = not flag

    # With --warmup_frozen N the backbone is held fixed for the first N epochs
    # so the fresh decoder/head can settle before large gradients reach the
    # pretrained features. Without it, an untrained head can wash the
    # pretrained representation out in the first few hundred steps.
    if args.warmup_frozen > 0:
        set_encoder_frozen(True)
        print(f"    encoder frozen for the first {args.warmup_frozen} epoch(s)")

    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    steps = max(len(dl_tr), 1) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    def predict(loader):
        model.eval()
        P, G = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=(device.type == "cuda")):
                    o = model(x)
                    logits = (o["out"] if isinstance(o, dict) else o).float()
                P.extend(list(logits.argmax(1).cpu().numpy()))
                G.extend(list(y.numpy()))
        return P, G

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        if args.warmup_frozen > 0 and ep == args.warmup_frozen:
            set_encoder_frozen(False)
            # rebuild the optimiser so the newly trainable tensors are tracked,
            # at a reduced rate to protect the pretrained features
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr * args.unfreeze_lr_scale, weight_decay=1e-4)
            rem = max((args.epochs - ep) * max(len(dl_tr), 1), 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.lr * args.unfreeze_lr_scale,
                total_steps=rem, pct_start=0.1)
            print(f"    encoder unfrozen at epoch {ep+1}, "
                  f"lr x{args.unfreeze_lr_scale}")
        model.train(); tot = n = 0
        bar = tqdm(dl_tr, desc=f"    ep {ep+1}/{args.epochs}", leave=False,
                   dynamic_ncols=True)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                o = model(x)
                logits = o["out"] if isinstance(o, dict) else o
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); n += 1
            bar.set_postfix(loss=f"{tot/max(n,1):.3f}")
        bar.close()
        vm = metrics(*predict(dl_va), eval_classes=args.eval_classes)
        print(f"    ep {ep+1}/{args.epochs}  loss {tot/max(n,1):.3f}  "
              f"ZeroWaste-val mIoU {100*vm['miou_with_bg_dataset']:.2f}")
        # selection on the ZeroWaste validation split, never on test
        if vm["miou_with_bg_dataset"] > best:
            best = vm["miou_with_bg_dataset"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    tm = metrics(*predict(dl_te), eval_classes=args.eval_classes)
    tm["init"] = init_tag
    tm["seed"] = seed
    tm["val_miou"] = best
    return tm, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero_root", required=True)
    ap.add_argument("--init", default="imagenet",
                    help="path to the EgoWaste checkpoint, or 'imagenet' for "
                         "the control")
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--freeze_stages", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="lower than pretraining: this is a fine-tuning stage")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026])
    ap.add_argument("--label_frac", type=float, default=1.0,
                    help="fraction of the ZeroWaste training set to use. "
                         "Pretraining usually helps most when target labels "
                         "are scarce, so run 0.1 / 0.25 / 0.5 before "
                         "concluding it does not help at all.")
    ap.add_argument("--transfer", choices=["full", "encoder"], default="full",
                    help="'encoder' keeps only the backbone and re-initialises "
                         "the decoder and head")
    ap.add_argument("--warmup_frozen", type=int, default=0,
                    help="hold the encoder frozen for this many epochs so the "
                         "fresh head can settle first")
    ap.add_argument("--unfreeze_lr_scale", type=float, default=0.1,
                    help="learning-rate multiplier applied when the encoder "
                         "unfreezes")
    ap.add_argument("--eval_classes", nargs="+", default=EVAL_CLASSES_DEFAULT,
                    choices=CLASSES)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default="./w6_finetune")
    args = ap.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    base = ("ego_pretrained" if args.init.lower() != "imagenet"
            else "zerowaste_only")
    if args.label_frac < 1.0:
        base += f"_{int(round(100*args.label_frac))}pct"
    if args.transfer != "full":
        base += f"_{args.transfer}"
    if args.warmup_frozen:
        base += f"_warm{args.warmup_frozen}"
    tag = args.tag or base

    import torch
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(f"STAGE 2  ::  fine-tune on ZeroWaste  [{tag}]   {device}")
    print("=" * 78)

    zero = load_zero(args.zero_root)
    print(f"\n  ZeroWaste train {len(zero['train'])}  val {len(zero['val'])}  "
          f"test {len(zero['test'])}")
    print(f"  evaluated classes: {', '.join(args.eval_classes)}")

    rows = []
    for seed in args.seeds:
        print(f"\n  --- seed {seed} ---")
        tm, state = run_once(args, zero, seed, device)
        rows.append({
            "tag": tag, "seed": seed, "init": tm["init"],
            "label_frac": args.label_frac, "transfer": args.transfer,
            "warmup_frozen": args.warmup_frozen,
            "val_miou": 100 * tm["val_miou"],
            "miou": 100 * tm["miou_with_bg_dataset"],
            "miou_no_bg": 100 * tm["miou_no_bg_dataset"],
            "fwiou": 100 * tm["fwiou"],
            "pixel_acc": 100 * tm["pixel_acc"],
            **{f"iou_{c}": (None if v is None else 100 * v)
               for c, v in tm["per_class_iou"].items()},
        })
        print(f"    test mIoU {rows[-1]['miou']:.2f}   "
              f"no-bg {rows[-1]['miou_no_bg']:.2f}   "
              f"pixel acc {rows[-1]['pixel_acc']:.2f}")
        if state and seed == args.seeds[0]:
            torch.save({"model": state, "encoder": args.encoder,
                        "freeze_stages": args.freeze_stages},
                       os.path.join(args.out_dir, f"zerowaste_{tag}.pt"))
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out_dir, f"finetune_{tag}.csv"), index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"RESULTS  [{tag}]   ZeroWaste test")
    print("=" * 78)
    print(df[["seed", "val_miou", "miou", "miou_no_bg", "pixel_acc"]]
          .round(2).to_string(index=False))
    if len(df) > 1:
        print(f"\n  mIoU {df['miou'].mean():.2f} +- {df['miou'].std():.2f} "
              f"over {len(df)} seeds")
        print("  Quote the mean with this spread. A difference between two")
        print("  configurations smaller than the spread is not a result.")
    print(f"\n  ZeroWaste's published DeepLabv3+ baseline: 46.93 mIoU")
    print(f"\nsaved -> {os.path.join(args.out_dir, f'finetune_{tag}.csv')}")


if __name__ == "__main__":
    main()
