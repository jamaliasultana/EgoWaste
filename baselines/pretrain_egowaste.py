#!/usr/bin/env python3
"""
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import annotation_utils as E
from taxonomy import TAXONOMY, build_lookup, resolve

# background plus the 15 material categories, in a fixed order
MATERIALS = list(TAXONOMY.keys())
CLASSES = ["background"] + MATERIALS
IDX = {c: i for i, c in enumerate(CLASSES)}

FRAMES_FOLDER_NAME = "frames_resized_1024"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ============================================================
# DATA
# ============================================================
def waste_class_of(row):
    for col in ("waste_class", "class_name"):
        v = row.get(col)
        if not E.is_missing(v):
            return str(v).strip()
    cat = row.get("category_name", row.get("category_str"))
    if E.is_missing(cat):
        return None
    s = str(cat).strip()
    if s.startswith("["):
        try:
            parts = [str(x).strip() for x in json.loads(s.replace("'", '"'))]
        except Exception:
            parts = [s]
    else:
        parts = [p.strip() for p in s.split("|") if p.strip()]
    return parts[-1] if parts else None


def find_frame_image(roots, cam, gx, frame, csv_paths=None):
    for p in (csv_paths or []):
        if not E.is_missing(p) and os.path.isfile(str(p)):
            return str(p)
    for root in roots:
        c = E.get_camera_dir(root, cam)
        if c is None:
            continue
        d = c / gx / FRAMES_FOLDER_NAME
        for ext in IMAGE_EXTS:
            q = d / f"{frame}{ext}"
            if q.is_file():
                return str(q)
    return None


def load_ego_materials(roots, frames_roots, cameras, gt_globs,
                       max_per_tracklet, val_frac=0.15, test_frac=0.15,
                       seed=2026):
    """
    Every verified object, labelled with its material category.
    Only videos carrying data/integrated_corrections/*.csv are used; masks
    resolve to the manually corrected copy when one exists.
    """
    lut = build_lookup()
    per_frame, skipped, unmapped = {}, [], Counter()

    for root in roots:
        day = Path(root).name
        for cam in cameras:
            cd = E.get_camera_dir(root, cam)
            if cd is None:
                continue
            for gx in sorted(cd.glob("GX*")):
                p, _ = E.find_gt_csv(gx, gt_globs, require_integrated=True)
                if p is None:
                    skipped.append(f"{day}|C{cam}_{gx.name}")
                    continue
                df = E.drop_deleted(pd.read_csv(p, low_memory=False))
                df = df.sort_values("frame_name")
                mask_index, mask_src = E.build_mask_index(gx)
                video = f"{day}|C{cam}_{gx.name}"
                per_track = defaultdict(int)

                for _, r in df.iterrows():
                    f = E.normalize_frame_name(r.get("frame_name"))
                    if f is None:
                        continue
                    oid = E.safe_int(r.get("object_track_id"))
                    if max_per_tracklet and per_track[oid] >= max_per_tracklet:
                        continue
                    mp, _s = E.resolve_mask(r, mask_index, mask_src)
                    if mp is None:
                        continue
                    wc = waste_class_of(r)
                    if wc is None:
                        continue
                    mat, _cls, _how = resolve(wc, lut)
                    if mat is None or mat not in IDX:
                        unmapped[wc] += 1
                        continue

                    key = (video, f)
                    if key not in per_frame:
                        img = find_frame_image(frames_roots, cam, gx.name, f,
                                               [r.get("image_path"),
                                                r.get("source_image_path")])
                        if img is None:
                            continue
                        per_frame[key] = {"kind": "ego", "image": img,
                                          "video": video, "objs": []}
                    per_track[oid] += 1
                    per_frame[key]["objs"].append((mp, IDX[mat]))

    items = list(per_frame.values())
    sizes = defaultdict(int)
    for it in items:
        sizes[it["video"]] += 1
    total = float(sum(sizes.values()))
    quota = {"train": (1 - val_frac - test_frac) * total,
             "val": val_frac * total, "test": test_frac * total}
    got = {k: 0.0 for k in quota}
    assign, rng = {}, random.Random(seed)
    for v, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
        room = {k: quota[k] - got[k] for k in quota}
        best = max(room, key=lambda k: (room[k], rng.random()))
        assign[v] = best
        got[best] += n
    out = {"train": [], "val": [], "test": []}
    for it in items:
        out[assign[it["video"]]].append(it)
    return out, skipped, unmapped


class SegSet:
    def __init__(self, items, size, train):
        import torch
        self.torch = torch
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        torch = self.torch
        r = self.items[i]
        img = cv2.imread(r["image"])
        if img is None:
            img = np.zeros((576, 1024, 3), np.uint8)
        img = img[:, :, ::-1]
        h, w = img.shape[:2]
        m = np.zeros((h, w), np.uint8)
        # paint smaller objects last so they are not erased by larger ones
        for mp, cls in sorted(r["objs"], key=lambda t: -os.path.getsize(t[0])
                              if os.path.isfile(t[0]) else 0):
            mm = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mm is None:
                continue
            if mm.shape[:2] != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            m[mm > 127] = cls

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return (torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(m).astype(np.int64)))


# ============================================================
# METRICS
# ============================================================
def confusion(pred, gt, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(int) + pred[k], minlength=n * n).reshape(n, n)


def miou(preds, gts, n):
    tot = np.zeros((n, n), np.int64)
    for p, g in zip(preds, gts):
        tot += confusion(p.ravel(), g.ravel(), n)
    inter = np.diag(tot).astype(float)
    union = tot.sum(1) + tot.sum(0) - np.diag(tot)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / union
    present = tot.sum(1) > 0          # classes absent from this split
    return {
        "miou_all": float(np.nanmean(iou)),
        "miou_present": float(np.nanmean(iou[present])),
        "n_present": int(present.sum()),
        "pixel_acc": float(inter.sum() / max(tot.sum(), 1)),
        "per_class_iou": {c: (None if np.isnan(v) else float(v))
                          for c, v in zip(CLASSES, iou)},
    }


def build_model(encoder, freeze_stages, n_classes, device):
    import segmentation_models_pytorch as smp
    m = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet",
                          classes=n_classes)
    enc, frozen = m.encoder, []
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
    print(f"  DeepLabV3+ / {encoder}   {n_classes}-way head   "
          f"frozen: {frozen or 'none'}")
    print(f"  trainable {n_tr/1e6:.1f}M of {n_all/1e6:.1f}M")
    return m.to(device)


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego_roots", nargs="+", required=True)
    ap.add_argument("--ego_frames_roots", nargs="*", default=[])
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gt_globs", nargs="+", default=None)
    ap.add_argument("--encoder", default="resnet101")
    ap.add_argument("--freeze_stages", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_per_tracklet", type=int, default=4)
    ap.add_argument("--use_all_data", action="store_true",
                    help="train on train+val; selection happens later on "
                         "ZeroWaste, so no EgoWaste split is needed here")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="./w6_pretrain_all")
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
    print(f"STAGE 1 :: pretrain on ALL {len(MATERIALS)} EgoWaste materials "
          f"  seed={args.seed}  {device}")
    print("=" * 78)

    ego, skipped, unmapped = load_ego_materials(
        args.ego_roots, args.ego_frames_roots + args.ego_roots, args.cameras,
        args.gt_globs, args.max_per_tracklet, seed=args.seed)

    tr = ego["train"] + (ego["val"] if args.use_all_data else [])
    va = ego["test"] if args.use_all_data else ego["val"]

    cnt = Counter()
    for it in tr:
        for _mp, cls in it["objs"]:
            cnt[CLASSES[cls]] += 1
    print(f"\n  train {len(tr)} frames   monitor {len(va)} frames")
    print(f"  {len(skipped)} videos skipped (no integrated_corrections)")
    print("\n  training objects per material:")
    for c in MATERIALS:
        print(f"    {c:<20s} {cnt.get(c, 0):>6d}")
    print(f"    {'TOTAL':<20s} {sum(cnt.values()):>6d}")
    if unmapped:
        print(f"\n  [warn] {sum(unmapped.values())} objects in "
              f"{len(unmapped)} unmapped classes were skipped:")
        for k, v in unmapped.most_common(10):
            print(f"    {k[:44]:<46s} {v:>6d}")

    model = build_model(args.encoder, args.freeze_stages, len(CLASSES), device)

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

    ck = os.path.join(args.out_dir, f"ego_all_materials_{args.encoder}.pt")
    log, best = [], -1.0
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

        m = miou(*predict(dl_va), n=len(CLASSES))
        print(f"  ep {ep+1}/{args.epochs}  loss {tot/max(n,1):.3f}  "
              f"mIoU {100*m['miou_present']:.2f} "
              f"({m['n_present']}/{len(CLASSES)} classes present)")
        log.append({"epoch": ep + 1, "loss": tot / max(n, 1),
                    "miou_present": m["miou_present"]})
        best = max(best, m["miou_present"])
        torch.save({"model": model.state_dict(), "encoder": args.encoder,
                    "freeze_stages": args.freeze_stages, "classes": CLASSES,
                    "n_classes": len(CLASSES), "epoch": ep + 1}, ck)

    m = miou(*predict(dl_va), n=len(CLASSES))
    print("\n  per-class IoU on the monitoring split:")
    for c, v in m["per_class_iou"].items():
        print(f"    {c:<20s} {'--' if v is None else f'{100*v:6.2f}'}")

    with open(os.path.join(args.out_dir, "pretrain_all_log.json"), "w") as f:
        json.dump({"encoder": args.encoder, "classes": CLASSES,
                   "seed": args.seed, "n_train_frames": len(tr),
                   "objects_per_material": dict(cnt),
                   "skipped_videos": skipped,
                   "unmapped": dict(unmapped), "epochs": log,
                   "best_miou_present": best,
                   "final_per_class_iou": m["per_class_iou"]}, f, indent=2)

    print(f"\n  checkpoint -> {ck}")
    print("\nSTAGE 2 (the head is re-initialised for ZeroWaste's 5 classes):")
    print(f"  python finetune_zerowaste.py --init {ck} \\")
    print(f"      --zero_root /path/to/splits_final_deblurred \\")
    print(f"      --epochs 20 --seeds 1 2 3")


if __name__ == "__main__":
    main()
