#!/usr/bin/env python3
"""
finetune_egowaste.py
==================
The mirror of finetune_zerowaste.py: fine-tunes on EgoWaste, starting either from
a ZeroWaste-pretrained checkpoint or from ImageNet, and evaluates on the
EgoWaste test split.

WHAT THIS IS AND IS NOT
-----------------------
This is cross-domain PRETRAINING, not zero-shot generalisation: the model
trains on EgoWaste labels and is selected on the EgoWaste validation split. The
zero-shot direction is the separate zero->ego cell of the cross matrix.

Run it with --init imagenet as well. The pretrained number is meaningless
without that control, since both must share the schedule, the split and the
metric code for the difference to be attributable to the initialisation.

WHY THE SYMMETRY IS WORTH MEASURING
-----------------------------------
If EgoWaste pretraining improves ZeroWaste but ZeroWaste pretraining does not
improve EgoWaste, the benefit is one-directional, which supports the claim that
EgoWaste supplies appearance variation that conveyor imagery lacks rather than
the two datasets simply being interchangeable extra data.

LABEL SPACE
-----------
EgoWaste is labelled here in ZeroWaste's four categories plus background, so
both directions share one label space and one metric. Metal and soft plastic
are excluded from the reported means by default: EgoWaste has fewer than 60
training instances of each. The same restriction is applied in the other
direction.

USAGE
  # control
  python finetune_egowaste.py --init imagenet \
      --ego_roots "/path/to/Landfill 20250602 Preproc" \
                  "/path/to/MRF 20250604 Preproc" \
      --ego_frames_roots "/path/to/egowaste_frames" \
      --cameras 1 2 3 4 5 --epochs 20 --seeds 1 2 3

  # ZeroWaste-pretrained
  python finetune_egowaste.py --init ./w6_finetune/zerowaste_zerowaste_only.pt \
      --ego_roots ... --epochs 20 --seeds 1 2 3
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from run_cross_dataset_matrix import (CLASSES, SegSet, load_ego, metrics,
                             class_support, EVAL_CLASSES_DEFAULT)
from pretrain_egowaste import build_model as build_model_n


def run_once(args, ego, seed, device):
    import torch
    from torch.utils.data import DataLoader
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = build_model_n(args.encoder, args.freeze_stages, len(CLASSES), device)

    init_tag = "imagenet"
    if args.init and args.init.lower() != "imagenet":
        sd = torch.load(args.init, map_location="cpu")
        enc = sd.get("encoder")
        if enc and enc != args.encoder:
            raise SystemExit(f"checkpoint uses encoder '{enc}' but --encoder "
                             f"is '{args.encoder}'")
        src, tgt = sd.get("model", sd), model.state_dict()

        # --transfer encoder keeps only the backbone. A decoder trained on
        # conveyor scenes encodes a spatial prior of many scattered items,
        # which conflicts with EgoWaste's single centred object.
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
        model.load_state_dict(transfer, strict=False)
        print(f"    transferred {len(transfer)}/{len(src)} tensors from "
              f"{os.path.basename(args.init)}")
        if shape_skip:
            print(f"    re-initialised {len(shape_skip)} tensor(s) of "
                  f"different shape (e.g. {shape_skip[0]})")
        if scope_skip:
            print(f"    --transfer encoder: decoder and head re-initialised "
                  f"({scope_skip} tensors skipped)")
        init_tag = os.path.basename(args.init)
    else:
        print("    initialised from ImageNet (control)")

    train_items = ego["train"]
    if args.label_frac < 1.0:
        rng = random.Random(seed)
        k = max(args.batch_size,
                int(round(len(train_items) * args.label_frac)))
        train_items = rng.sample(train_items, min(k, len(train_items)))
        print(f"    low-shot: {len(train_items)}/{len(ego['train'])} images "
              f"({100*args.label_frac:.0f}%)")

    dl_tr = DataLoader(SegSet(train_items, args.size, True),
                       batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(SegSet(ego["val"] or ego["test"], args.size, False),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    dl_te = DataLoader(SegSet(ego["test"], args.size, False),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    def set_encoder_frozen(flag):
        for p in model.encoder.parameters():
            p.requires_grad = not flag

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
                    logits = model(x).float()
                P.extend(list(logits.argmax(1).cpu().numpy()))
                G.extend(list(y.numpy()))
        return P, G

    best, best_state = -1.0, None
    for ep in range(args.epochs):
        if args.warmup_frozen > 0 and ep == args.warmup_frozen:
            set_encoder_frozen(False)
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
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); n += 1
            bar.set_postfix(loss=f"{tot/max(n,1):.3f}")
        bar.close()

        vm = metrics(*predict(dl_va), eval_classes=args.eval_classes)
        print(f"    ep {ep+1}/{args.epochs}  loss {tot/max(n,1):.3f}  "
              f"EgoWaste-val mIoU {100*vm['miou_with_bg_dataset']:.2f}")
        # selection on the EgoWaste validation split, never on test
        if vm["miou_with_bg_dataset"] > best:
            best = vm["miou_with_bg_dataset"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    tm = metrics(*predict(dl_te), eval_classes=args.eval_classes)
    tm["init"] = init_tag
    tm["val_miou"] = best
    return tm, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego_roots", nargs="+", required=True)
    ap.add_argument("--ego_frames_roots", nargs="*", default=[])
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gt_globs", nargs="+", default=None)
    ap.add_argument("--init", default="imagenet",
                    help="ZeroWaste checkpoint, or 'imagenet' for the control")
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--freeze_stages", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_per_tracklet", type=int, default=8)
    ap.add_argument("--label_frac", type=float, default=1.0)
    ap.add_argument("--transfer", choices=["full", "encoder"], default="full")
    ap.add_argument("--warmup_frozen", type=int, default=0)
    ap.add_argument("--unfreeze_lr_scale", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026])
    ap.add_argument("--eval_classes", nargs="+", default=EVAL_CLASSES_DEFAULT,
                    choices=CLASSES)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default="./w6_finetune_ego")
    args = ap.parse_args()

    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    base = ("zero_pretrained" if args.init.lower() != "imagenet"
            else "egowaste_only")
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
    print(f"fine-tune on EgoWaste  [{tag}]   {device}")
    print("=" * 78)

    ego, skipped = load_ego(args.ego_roots,
                            args.ego_frames_roots + args.ego_roots,
                            args.cameras, args.gt_globs, args.max_per_tracklet,
                            seed=args.seeds[0])
    print(f"\n  EgoWaste train {len(ego['train'])}  val {len(ego['val'])}  "
          f"test {len(ego['test'])}   ({len(skipped)} videos skipped)")
    sup = class_support(ego["train"])
    print("  training objects per class:")
    for c in CLASSES:
        if c != "background":
            print(f"    {c:<16s} {sup.get(c, 0):>6d}")
    print(f"  evaluated classes: {', '.join(args.eval_classes)}")

    rows = []
    for seed in args.seeds:
        print(f"\n  --- seed {seed} ---")
        tm, state = run_once(args, ego, seed, device)
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
              f"no-bg {rows[-1]['miou_no_bg']:.2f}")
        if state and seed == args.seeds[0]:
            torch.save({"model": state, "encoder": args.encoder,
                        "freeze_stages": args.freeze_stages,
                        "classes": CLASSES, "n_classes": len(CLASSES)},
                       os.path.join(args.out_dir, f"egowaste_{tag}.pt"))
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out_dir, f"finetune_ego_{tag}.csv"), index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"RESULTS  [{tag}]   EgoWaste test")
    print("=" * 78)
    print(df[["seed", "val_miou", "miou", "miou_no_bg", "pixel_acc"]]
          .round(2).to_string(index=False))
    if len(df) > 1:
        print(f"\n  mIoU {df['miou'].mean():.2f} +- {df['miou'].std():.2f} "
              f"over {len(df)} seeds")
        print("  A difference smaller than this spread is not a result.")
    print(f"\nsaved -> {os.path.join(args.out_dir, f'finetune_ego_{tag}.csv')}")


if __name__ == "__main__":
    main()
