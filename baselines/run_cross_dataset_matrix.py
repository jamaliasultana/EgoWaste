#!/usr/bin/env python3
"""
run_cross_dataset_matrix.py
==================
Runs the full train x test matrix behind the cross-dataset claim, so every cell
comes from one implementation, one architecture, one input size and one metric.

           test:  ZeroWaste        EgoWaste
  train
  EgoWaste       transfer          in-domain
  ZeroWaste      in-domain         transfer
  Ego+Zero       joint             joint

Reported per cell: mIoU over classes (the standard segmentation metric) and
pixel accuracy. They are reported side by side because the paper previously
quoted the second while calling it the first, and on this data they differ by
roughly 40 points: background occupies ~90% of ZeroWaste pixels, so pixel
accuracy is dominated by it, whereas mIoU weights every waste class equally.

PROTOCOL RULES ENFORCED HERE
----------------------------
  * model selection always uses the TRAINING domain's validation split. Using
    the target domain's validation split would be adaptation and would void a
    zero-shot claim.
  * identical DeepLabV3+ ResNet-50, identical 512px input, identical class
    indices, identical metric code in every cell.

A LABEL-CONVENTION ASYMMETRY WORTH KNOWING BEFORE READING THE MATRIX
--------------------------------------------------------------------
EgoWaste annotates only the object being held; all other waste in the frame is
background. ZeroWaste annotates every item on the conveyor. A ZeroWaste-trained
model therefore segments waste that EgoWaste labels as background, which
depresses the Zero->Ego cell for reasons of annotation scope rather than visual
transfer. The script prints the foreground pixel share of each test set so the
size of that asymmetry is visible.

USAGE
  # one cell
  python run_cross_dataset_matrix.py --train ego --test zero \
      --ego_roots "/path/to/Brookhaven Landfill 20250602 Preproc" \
                  "/path/to/Brookhaven MRF 20250604 Preproc" \
      --ego_frames_roots "/path/to/egowaste_frames" \
      --zero_root "/path/to/datasets/zerowaste-f-final/splits_final_deblurred" \
      --epochs 40 --out_dir ./w6_matrix

  # all six cells (three trainings, each evaluated on both test sets)
  python run_cross_dataset_matrix.py --all ...same paths...
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import annotation_utils as E
from taxonomy import build_lookup, resolve

CLASSES = ["background", "rigid_plastic", "cardboard", "metal", "soft_plastic"]
IDX = {c: i for i, c in enumerate(CLASSES)}

# Classes evaluated by default. EgoWaste contains fewer than 60 training
# instances of metal and soft plastic, and its held-out videos contain 14 and 1
# test instances, so their IoU is not estimable. Restricting to the three
# well-supported categories keeps the mean meaningful -- and the SAME
# restriction is applied to ZeroWaste, so neither dataset is scored on a
# different class set from the other.
EVAL_CLASSES_DEFAULT = ["background", "rigid_plastic", "cardboard"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".PNG")
FRAMES_FOLDER_NAME = "frames_resized_1024"
# priority matches integrated_reviewer.py / export_clean_gx.py
MASK_COLS = E.MASK_COL_PRIORITY

# EgoWaste fine-grained class -> ZeroWaste category; anything absent is
# background, which is how ZeroWaste treats glass, food, textiles and the
# non-corrugated paper grades
ZW_MAP = {
    "containers 1 clamshells": "rigid_plastic", "containers 2 colored": "rigid_plastic",
    "containers 2 natural": "rigid_plastic", "containers 3 7 not 5": "rigid_plastic",
    "containers 5 pp": "rigid_plastic", "deposit 1 bottles": "rigid_plastic",
    "other 1 bottles": "rigid_plastic", "non deposit": "rigid_plastic",
    "black plastic containers": "rigid_plastic", "rigid plastic": "rigid_plastic",
    "other rigid bulky plastics": "rigid_plastic",
    "other unnumbered containers and fragments": "rigid_plastic",
    "plastic": "rigid_plastic", "other plastics": "rigid_plastic",
    "other plastic film": "soft_plastic", "garbage bags": "soft_plastic",
    "retail plastic bags": "soft_plastic",
    "old corrugated cardboard": "cardboard", "boxboard": "cardboard",
    "pizza boxes": "cardboard",
    "aluminum foil": "metal", "deposit aluminum containers": "metal",
    "other aluminum": "metal", "other aluminum containers": "metal",
    "ferrous containers ferris container": "metal", "other ferrous": "metal",
    "other non ferrous": "metal",
}


def norm(s):
    s = str(s).lower().strip().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


ZW_MAP_N = {norm(k): v for k, v in ZW_MAP.items()}


# ============================================================
# DATASETS
# ============================================================
class SegSet:
    """Uniform interface: returns (image tensor, class-index mask tensor)."""

    def __init__(self, items, size, train):
        import torch
        self.torch = torch
        self.items, self.size, self.train = items, size, train

    def __len__(self):
        return len(self.items)

    def _norm(self, img, m):
        torch = self.torch
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return (torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(m).astype(np.int64)))

    def __getitem__(self, i):
        r = self.items[i]
        img = cv2.imread(r["image"])
        if img is None:
            img = np.zeros((576, 1024, 3), np.uint8)
        img = img[:, :, ::-1]
        h, w = img.shape[:2]

        if r["kind"] == "zero":
            m = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
            m = np.zeros((h, w), np.uint8) if m is None else m
            m[m > len(CLASSES) - 1] = 0
        else:
            m = np.zeros((h, w), np.uint8)
            for mp, cls in r["objs"]:
                if cls == 0:
                    continue
                mm = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if mm is None:
                    continue
                if mm.shape[:2] != (h, w):
                    mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                m[mm > 127] = cls
        return self._norm(img, m)


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


def waste_class_of(row):
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


def load_ego(roots, frames_roots, cameras, gt_globs, max_per_tracklet,
             val_frac=0.15, test_frac=0.15, seed=2026):
    """Split EgoWaste by video into train/val/test, in ZeroWaste's label space."""
    per_frame, skipped = {}, []
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
                    mp, _s = E.resolve_mask(r, mask_index, mask_src, cols=MASK_COLS)
                    if mp is None:
                        continue
                    wc = waste_class_of(r)
                    if wc is None:
                        continue
                    cls = IDX.get(ZW_MAP_N.get(norm(wc), ""), 0)
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
                    per_frame[key]["objs"].append((mp, cls))

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
    return out, skipped


def load_zero(root):
    root = Path(root)
    out = {}
    for split in ("train", "val", "test"):
        base = root / split
        if not base.is_dir():
            out[split] = []
            continue
        img_dir = next((base / c for c in ("data", "images", "image")
                        if (base / c).is_dir()), None)
        msk_dir = next((base / c for c in ("sem_seg", "semantic", "masks", "labels")
                        if (base / c).is_dir()), None)
        if img_dir is None or msk_dir is None:
            raise SystemExit(f"cannot find image/mask folders under {base}; "
                             f"contains {[p.name for p in base.iterdir() if p.is_dir()]}")
        masks = {p.stem: p for p in msk_dir.iterdir() if p.suffix in IMAGE_EXTS}
        items = []
        for p in sorted(img_dir.iterdir()):
            if p.suffix not in IMAGE_EXTS:
                continue
            m = masks.get(p.stem)
            if m is not None:
                items.append({"kind": "zero", "image": str(p), "mask": str(m)})
        out[split] = items
    return out


# ============================================================
# METRICS
# ============================================================
def confusion(pred, gt, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(int) + pred[k], minlength=n * n).reshape(n, n)


def iou_from_conf(cm):
    inter = np.diag(cm).astype(float)
    union = cm.sum(1) + cm.sum(0) - np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        return inter / union


def metrics(preds, gts, n=len(CLASSES), eval_classes=None):
    """
    Every mIoU convention in common use, reported side by side, because on a
    background-dominated set they disagree by tens of points and the paper
    previously quoted one while naming another.

      miou_with_bg_dataset    confusion matrix accumulated over the whole test
                              set, averaged over background + 4 classes. This is
                              the standard semantic-segmentation convention and
                              the one ZeroWaste's published baseline uses.
      miou_no_bg_dataset      same, background excluded
      miou_with_bg_per_image  mIoU computed per image over the classes present
                              in that image, then averaged over images
      miou_no_bg_per_image    same, background excluded
      fwiou                   frequency-weighted IoU: each class IoU weighted by
                              its pixel share. This is the legitimate
                              pixel-weighted IoU.
      pixel_acc               correct pixels / all pixels. NOT an IoU: there is
                              no union term, so it must never be reported under
                              an mIoU heading.
      mean_pixel_acc          per-class recall averaged over classes

    eval_classes restricts every mean to a subset of the label space. Pixels of
    excluded classes are NOT deleted -- they remain in the confusion matrix as
    predictions and as ground truth, so a model that confuses an excluded class
    for an evaluated one is still penalised. Only the averaging is restricted.
    """
    keep = ([IDX[c] for c in eval_classes] if eval_classes
            else list(range(n)))
    keep_fg = [i for i in keep if i != 0]
    tot = np.zeros((n, n), np.int64)
    per_img_with, per_img_no = [], []

    for p, g in zip(preds, gts):
        cm = confusion(p.ravel(), g.ravel(), n)
        tot += cm
        iou_i = iou_from_conf(cm)
        present = [c for c in np.unique(g) if c in keep]
        if not present:
            continue
        v = iou_i[present]
        v = v[~np.isnan(v)]
        if len(v):
            per_img_with.append(float(v.mean()))
        fg = [c for c in present if c != 0]
        if fg:
            vf = iou_i[fg]
            vf = vf[~np.isnan(vf)]
            if len(vf):
                per_img_no.append(float(vf.mean()))

    iou = iou_from_conf(tot)
    inter = np.diag(tot).astype(float)
    gt_share = tot.sum(1) / max(tot.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cls_recall = inter / tot.sum(1)

    w = gt_share[keep]
    w = w / w.sum() if w.sum() > 0 else w

    return {
        "miou_with_bg_dataset": float(np.nanmean(iou[keep])),
        "miou_no_bg_dataset": (float(np.nanmean(iou[keep_fg]))
                               if keep_fg else float("nan")),
        "miou_with_bg_per_image": (float(np.mean(per_img_with))
                                   if per_img_with else float("nan")),
        "miou_no_bg_per_image": (float(np.mean(per_img_no))
                                 if per_img_no else float("nan")),
        "fwiou": float(np.nansum(w * iou[keep])),
        "pixel_acc": float(inter.sum() / max(tot.sum(), 1)),
        "mean_pixel_acc": float(np.nanmean(cls_recall[keep])),
        "miou_all_classes": float(np.nanmean(iou)),   # for reference
        "eval_classes": [CLASSES[i] for i in keep],
        "per_class_iou": {c: (None if np.isnan(v) else float(v))
                          for c, v in zip(CLASSES, iou)},
        "per_class_pixel_share": {c: float(v) for c, v in zip(CLASSES, gt_share)},
        "fg_pixel_share": float(1.0 - gt_share[0]),
    }


# ============================================================
# TRAIN / EVAL
# ============================================================
def build_model(device):
    import torch
    try:
        import segmentation_models_pytorch as smp
        m = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet",
                              classes=len(CLASSES))
        tag = "DeepLabV3+ (smp)"
    except ImportError:
        import torchvision
        m = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT",
                                                               aux_loss=True)
        m.classifier[4] = torch.nn.Conv2d(256, len(CLASSES), 1)
        m.aux_classifier[4] = torch.nn.Conv2d(256, len(CLASSES), 1)
        tag = "DeepLabV3 (torchvision fallback)"
    return m.to(device), tag


def predict(model, loader, device):
    import torch
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


def train_model(tr_items, va_items, args, device):
    import torch
    from torch.utils.data import DataLoader
    model, tag = build_model(device)
    print(f"    model: {tag}   train {len(tr_items)}  val {len(va_items)}")

    dl_tr = DataLoader(SegSet(tr_items, args.size, True), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.workers, pin_memory=True,
                       drop_last=True)
    dl_va = DataLoader(SegSet(va_items, args.size, False),
                       batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(len(dl_tr), 1) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best, best_state = -1.0, None
    for ep in range(args.epochs):
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
                if isinstance(o, dict) and "aux" in o:
                    loss = loss + 0.4 * crit(o["aux"], y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); n += 1
            bar.set_postfix(loss=f"{tot/max(n,1):.3f}")
        bar.close()
        vm = metrics(*predict(model, dl_va, device),
                     eval_classes=getattr(args, "eval_classes", None))
        print(f"    ep {ep+1}/{args.epochs}  loss {tot/max(n,1):.3f}  "
              f"source-val mIoU(with bg) {100*vm['miou_with_bg_dataset']:.2f}")
        # selection on the TRAINING domain's val split only, using the standard
        # dataset-level mIoU
        if vm["miou_with_bg_dataset"] > best:
            best = vm["miou_with_bg_dataset"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_on(model, items, args, device, eval_classes=None):
    from torch.utils.data import DataLoader
    dl = DataLoader(SegSet(items, args.size, False), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    return metrics(*predict(model, dl, device), eval_classes=eval_classes)


def class_support(items):
    """Objects per class, so the report can state what each split contains."""
    from collections import Counter
    c = Counter()
    for it in items:
        if it["kind"] == "ego":
            for _mp, cls in it["objs"]:
                c[CLASSES[cls]] += 1
    return c


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego_roots", nargs="+", required=True)
    ap.add_argument("--ego_frames_roots", nargs="*", default=[])
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gt_globs", nargs="+", default=None)
    ap.add_argument("--zero_root", required=True)
    ap.add_argument("--train", choices=["ego", "zero", "both"], default="ego")
    ap.add_argument("--test", choices=["ego", "zero"], default="zero")
    ap.add_argument("--all", action="store_true",
                    help="run all three trainings, each evaluated on both test sets")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_per_tracklet", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--eval_classes", nargs="+", default=EVAL_CLASSES_DEFAULT,
                    choices=CLASSES,
                    help="classes averaged into the reported means. Applied "
                         "identically to BOTH datasets so neither is scored on "
                         "a different label set. Default excludes metal and "
                         "soft plastic, which EgoWaste cannot support.")
    ap.add_argument("--all_classes", dest="eval_classes",
                    action="store_const", const=CLASSES,
                    help="average over all five classes instead")
    ap.add_argument("--ego_oversample", type=float, default=1.0,
                    help="scale EgoWaste's contribution to the joint training "
                         "set; 1.0 equalises frame counts, higher values "
                         "compensate for its sparser foreground")
    ap.add_argument("--out_dir", default="./w6_matrix")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(f"W6 cross-dataset matrix   device={device}   outputs={args.out_dir}")
    print("=" * 78)

    ego, skipped = load_ego(args.ego_roots, args.ego_frames_roots + args.ego_roots,
                            args.cameras, args.gt_globs, args.max_per_tracklet,
                            seed=args.seed)
    zero = load_zero(args.zero_root)
    print(f"\n  EgoWaste  train {len(ego['train'])}  val {len(ego['val'])}  "
          f"test {len(ego['test'])}   ({len(skipped)} videos skipped)")
    print(f"  ZeroWaste train {len(zero['train'])}  val {len(zero['val'])}  "
          f"test {len(zero['test'])}")

    print(f"\n  EVALUATED CLASSES: {', '.join(args.eval_classes)}")
    excluded = [c for c in CLASSES if c not in args.eval_classes]
    if excluded:
        print(f"  excluded from the means: {', '.join(excluded)}")
        print("  Excluded classes remain in the confusion matrix, so confusing")
        print("  one of them for an evaluated class is still penalised; only the")
        print("  averaging is restricted. The same restriction is applied to")
        print("  ZeroWaste, so both datasets are scored on one label set.")
    print("\n  EgoWaste objects per class:")
    for split in ("train", "val", "test"):
        sup = class_support(ego[split])
        row = "  ".join(f"{c}={sup.get(c,0)}" for c in CLASSES if c != "background")
        print(f"    {split:<6s} {row}")

    combos = ([("ego", "zero"), ("zero", "zero"), ("both", "zero"),
               ("zero", "ego"), ("ego", "ego"), ("both", "ego")]
              if args.all else [(args.train, args.test)])

    rows, cache = [], {}
    for tr_name, te_name in combos:
        print(f"\n{'='*78}\n  train={tr_name}  ->  test={te_name}\n{'='*78}")
        if tr_name not in cache:
            if tr_name == "ego":
                tr, va = ego["train"], ego["val"]
            elif tr_name == "zero":
                tr, va = zero["train"], zero["val"] or zero["test"]
            else:
                # balance the two sources: ZeroWaste has ~3.5x denser
                # foreground, so an unweighted concatenation lets it dominate
                k = max(1, round(len(zero["train"]) / max(len(ego["train"]), 1)
                                 * args.ego_oversample))
                tr = ego["train"] * k + zero["train"]
                print(f"    joint set: EgoWaste x{k} ({len(ego['train'])*k} frames) "
                      f"+ ZeroWaste ({len(zero['train'])} frames)")
                va = ego["val"] + (zero["val"] or [])
            cache[tr_name] = train_model(tr, va, args, device)
            torch.save({"model": cache[tr_name].state_dict()},
                       os.path.join(args.out_dir, f"deeplabv3plus_{tr_name}.pt"))
        te = ego["test"] if te_name == "ego" else zero["test"]
        m = evaluate_on(cache[tr_name], te, args, device,
                        eval_classes=args.eval_classes)
        row = {"train": tr_name, "test": te_name, "n_test": len(te)}
        for k in ("miou_with_bg_dataset", "miou_no_bg_dataset",
                  "miou_with_bg_per_image", "miou_no_bg_per_image",
                  "fwiou", "pixel_acc", "mean_pixel_acc", "fg_pixel_share"):
            row[k] = 100 * m[k]
        row["miou_all_classes"] = 100 * m["miou_all_classes"]
        row["eval_classes"] = "|".join(m["eval_classes"])
        for c, v in m["per_class_iou"].items():
            row[f"iou_{c}"] = None if v is None else 100 * v
        rows.append(row)
        print(f"    mIoU(with bg) {row['miou_with_bg_dataset']:6.2f}   "
              f"mIoU(no bg) {row['miou_no_bg_dataset']:6.2f}   "
              f"FWIoU {row['fwiou']:6.2f}   pixel acc {row['pixel_acc']:6.2f}")
        print(f"    per-image: with bg {row['miou_with_bg_per_image']:6.2f}   "
              f"no bg {row['miou_no_bg_per_image']:6.2f}   "
              f"(foreground = {row['fg_pixel_share']:.1f}% of test pixels)")

        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(args.out_dir, "w6_matrix.csv"), index=False)

    res = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("CROSS-DATASET MATRIX")
    print("=" * 78)
    cols = ["train", "test", "miou_with_bg_dataset", "miou_no_bg_dataset",
            "miou_with_bg_per_image", "miou_no_bg_per_image", "fwiou",
            "pixel_acc", "miou_all_classes", "fg_pixel_share", "n_test"]
    print(res[cols].round(2).to_string(index=False))
    print("\n  WHICH NUMBER TO QUOTE")
    print("    miou_with_bg_dataset is the standard convention and the one the")
    print("    ZeroWaste baseline uses -- report this as mIoU.")
    print("    fwiou is the legitimate pixel-weighted IoU if a pixel-weighted")
    print("    figure is wanted.")
    print("    pixel_acc is NOT an IoU: it has no union term and must never be")
    print("    reported under an mIoU heading. On a set that is ~90% background")
    print("    it stays high however poorly the waste classes are segmented.")
    print(f"\n  All means above cover: {', '.join(args.eval_classes)}.")
    print("    miou_all_classes is the five-class figure, shown only for")
    print("    reference; it is dominated by metal and soft plastic, which")
    print("    EgoWaste has too few instances to learn or to measure.")
    print("\n  EgoWaste labels only the manipulated object, ZeroWaste labels every")
    print("  item on the belt, so a ZeroWaste-trained model marks waste that")
    print("  EgoWaste treats as background; read the zero->ego cell with that in")
    print("  mind.")
    print(f"\nsaved -> {os.path.join(args.out_dir, 'w6_matrix.csv')}")


if __name__ == "__main__":
    main()
