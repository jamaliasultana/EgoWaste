#!/usr/bin/env python3
"""
pretrain_shared_materials.py   --  STAGE 1 of 2
=====================================
Pretrains DeepLabV3+ on all verified EgoWaste data, in ZeroWaste's label space,
and saves a checkpoint for finetune_zerowaste.py to fine-tune.

MATCHING THE ZEROWASTE BASELINE
-------------------------------
The ZeroWaste paper trains DeepLabv3+ with a ResNet-101 backbone and freezes
the first three backbone stages. Those defaults are mirrored here, because the
comparison against their published 46.93 mIoU is only meaningful if both sides
use the same architecture and the same frozen-stage policy. Override with
--encoder / --freeze_stages if you want the ResNet-50 variant.

DATA
----
Only videos carrying data/integrated_corrections/*.csv are used, so unreviewed
folders are skipped. Masks resolve to the manually corrected copy where one
exists and to the pipeline mask otherwise, exactly as integrated_reviewer.py
does.

--use_all_data folds the EgoWaste validation split into training, which is the
right choice here: this stage produces an initialisation, and model selection
happens later on ZeroWaste. Keep it off if you want a held-out EgoWaste split
for monitoring.

USAGE
  python pretrain_shared_materials.py \
     --ego_roots "/path/to/Brookhaven Landfill 20250602 Preproc" \
                 "/path/to/Brookhaven MRF 20250604 Preproc" \
     --ego_frames_roots "/path/to/egowaste_frames" \
     --cameras 1 2 3 4 5 --epochs 40 --use_all_data \
     --out_dir ./w6_pretrain
"""

import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm

from run_cross_dataset_matrix import (CLASSES, SegSet, load_ego, metrics,
                             class_support, EVAL_CLASSES_DEFAULT)


def build_model(encoder, freeze_stages, device):
    """
    DeepLabV3+ with the requested encoder. freeze_stages mirrors the ZeroWaste
    setup: freezing the early stages keeps generic low-level features fixed and
    is what their baseline does.
    """
    import torch
    import segmentation_models_pytorch as smp
    m = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet",
                          classes=len(CLASSES))
    enc = m.encoder
    frozen = []
    if freeze_stages > 0:
        for name in ["conv1", "bn1"]:
            if hasattr(enc, name):
                for p in getattr(enc, name).parameters():
                    p.requires_grad = False
                frozen.append(name)
        for i in range(1, freeze_stages + 1):
            layer = getattr(enc, f"layer{i}", None)
            if layer is not None:
                for p in layer.parameters():
                    p.requires_grad = False
                frozen.append(f"layer{i}")
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    print(f"  DeepLabV3+ / {encoder}   frozen: {frozen or 'none'}")
    print(f"  trainable {n_tr/1e6:.1f}M of {n_all/1e6:.1f}M "
          f"({100*n_tr/n_all:.0f}%)")
    return m.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego_roots", nargs="+", required=True)
    ap.add_argument("--ego_frames_roots", nargs="*", default=[])
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gt_globs", nargs="+", default=None)
    ap.add_argument("--encoder", default="resnet101",
                    help="ZeroWaste's baseline uses resnet101")
    ap.add_argument("--freeze_stages", type=int, default=3,
                    help="freeze the stem plus this many residual stages, as "
                         "the ZeroWaste baseline does")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_per_tracklet", type=int, default=4)
    ap.add_argument("--use_all_data", action="store_true",
                    help="train on EgoWaste train+val; selection happens later "
                         "on ZeroWaste, so no EgoWaste split is needed here")
    ap.add_argument("--eval_classes", nargs="+", default=EVAL_CLASSES_DEFAULT,
                    choices=CLASSES)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="./w6_pretrain")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(f"STAGE 1  ::  pretrain on EgoWaste     seed={args.seed}  {device}")
    print("=" * 78)

    ego, skipped = load_ego(args.ego_roots,
                            args.ego_frames_roots + args.ego_roots,
                            args.cameras, args.gt_globs, args.max_per_tracklet,
                            seed=args.seed)
    tr = ego["train"] + (ego["val"] if args.use_all_data else [])
    va = ego["test"] if args.use_all_data else ego["val"]

    print(f"\n  train {len(tr)}   monitor {len(va)}   "
          f"({len(skipped)} videos skipped: no integrated_corrections)")
    sup = class_support(tr)
    print("  training objects per class:")
    for c in CLASSES:
        if c != "background":
            print(f"    {c:<16s} {sup.get(c, 0):>6d}")

    model = build_model(args.encoder, args.freeze_stages, device)

    dl_tr = DataLoader(SegSet(tr, args.size, True), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.workers, pin_memory=True,
                       drop_last=True)
    dl_va = DataLoader(SegSet(va, args.size, False), batch_size=args.batch_size,
                       shuffle=False, num_workers=args.workers, pin_memory=True)

    params = [p for p in model.parameters() if p.requires_grad]
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
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

    log, best = [], -1.0
    ck = os.path.join(args.out_dir, f"ego_pretrained_{args.encoder}.pt")
    for ep in range(args.epochs):
        model.train(); tot = n = 0
        bar = tqdm(dl_tr, desc=f"  ep {ep+1}/{args.epochs}", leave=False,
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
        m = metrics(*predict(dl_va), eval_classes=args.eval_classes)
        print(f"  ep {ep+1}/{args.epochs}  loss {tot/max(n,1):.3f}  "
              f"EgoWaste mIoU {100*m['miou_with_bg_dataset']:.2f}")
        log.append({"epoch": ep + 1, "loss": tot / max(n, 1),
                    "ego_miou": m["miou_with_bg_dataset"]})
        # save every epoch: this is an initialisation, and a crash late in a
        # long run should not cost the whole thing
        if m["miou_with_bg_dataset"] > best:
            best = m["miou_with_bg_dataset"]
        torch.save({"model": model.state_dict(), "encoder": args.encoder,
                    "freeze_stages": args.freeze_stages,
                    "classes": CLASSES, "epoch": ep + 1}, ck)

    with open(os.path.join(args.out_dir, "pretrain_log.json"), "w") as f:
        json.dump({"encoder": args.encoder, "freeze_stages": args.freeze_stages,
                   "seed": args.seed, "n_train": len(tr),
                   "skipped_videos": skipped, "epochs": log,
                   "best_ego_miou": best}, f, indent=2)

    print(f"\n  best EgoWaste mIoU during pretraining: {100*best:.2f}")
    print(f"  checkpoint -> {ck}")
    print("\nSTAGE 2:")
    print(f"  python finetune_zerowaste.py --init {ck} \\")
    print(f"      --zero_root /path/to/splits_final_deblurred --epochs 20")


if __name__ == "__main__":
    main()
