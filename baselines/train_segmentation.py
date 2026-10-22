#!/usr/bin/env python3
"""
train_segmentation.py
============================
Benchmarks segmentation models on the ACTIVE (handheld) OBJECT: recover the
exact shape, size and boundary of the item the expert is manipulating, inside a
frame containing dozens of similar items.

MODELS
  mask_rcnn        torchvision maskrcnn_resnet50_fpn_v2       (instance)
  deeplabv3plus    segmentation_models_pytorch, ResNet-50     (semantic)
                   falls back to torchvision DeepLabV3 if smp is absent
  yolo9_seg        ultralytics YOLOv9 segmentation            (instance)
  cascade_mask_rcnn  detectron2; trained separately, imported via --import_preds

METRICS
  mask AP50/AP75/AP   standard instance-segmentation quality
  top1_mask_iou       IoU of the HIGHEST-SCORING predicted mask with the true
                      object. The annotation pipeline consumes one mask per
                      frame, so this is the number that matters, and it is not
                      what AP measures.
  top1_hit@0.5/0.75   that mask exceeds an IoU threshold
  boundary_f1         contour agreement at a 3 px tolerance -- rewards correct
                      boundaries rather than merely overlapping blobs
  mIoU                foreground IoU, the natural metric for the semantic model

DeepLabV3+ is semantic, so it has no instances or scores. Its single foreground
prediction is treated as the one prediction, which is exactly how it would be
used in the pipeline.

USAGE
  python train_segmentation.py --data_dir ./active_seg_data \
      --label_mode agnostic --model mask_rcnn --epochs 20

  python train_segmentation.py --data_dir ./active_seg_data \
      --label_mode agnostic --all_models

REQUIREMENTS
  pip install torch torchvision ultralytics segmentation-models-pytorch pycocotools
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

INSTANCE_MODELS = ("mask_rcnn", "yolo9_seg")
SEMANTIC_MODELS = ("deeplabv3plus",)


# ============================================================
# MASK METRICS
# ============================================================
def mask_iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def boundary_f1(pred, gt, tol=3):
    """
    F1 between mask contours within `tol` pixels. Two masks can overlap well
    yet have very different boundaries; this separates the two cases.
    """
    p, g = (pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    k = np.ones((3, 3), np.uint8)
    pe = p - cv2.erode(p, k)
    ge = g - cv2.erode(g, k)
    if pe.sum() == 0 or ge.sum() == 0:
        return 0.0
    d = 2 * tol + 1
    pe_d = cv2.dilate(pe, np.ones((d, d), np.uint8))
    ge_d = cv2.dilate(ge, np.ones((d, d), np.uint8))
    prec = float(np.logical_and(pe, ge_d).sum()) / max(pe.sum(), 1)
    rec = float(np.logical_and(ge, pe_d).sum()) / max(ge.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def voc_ap(scores, matched, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return float("nan")
    o = np.argsort(-np.asarray(scores))
    m = np.asarray(matched, float)[o]
    tp, fp = np.cumsum(m), np.cumsum(1 - m)
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    rec = np.concatenate([[0], rec]); prec = np.concatenate([[prec[0]], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def gt_masks_for(coco, img_id, ann_by_img, hw):
    h, w = hw
    out = []
    for a in ann_by_img.get(img_id, []):
        m = np.zeros((h, w), np.uint8)
        for poly in a["segmentation"]:
            pts = np.asarray(poly, np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
        out.append({"mask": m, "category_id": a["category_id"]})
    return out


def evaluate(preds_by_img, gt_json, use_class=True):
    """preds_by_img: image_id -> [{mask HxW uint8, score, category_id}] sorted or not"""
    with open(gt_json) as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco["images"]}
    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    thrs = np.arange(0.5, 1.0, 0.05)
    scores, matched = [], {t: [] for t in thrs}
    n_gt = sum(len(v) for v in ann_by_img.values())

    top_ious, top_bf1 = [], []
    hit50 = hit75 = n = 0
    inter_sum = union_sum = 0

    for iid, im in imgs.items():
        hw = (im["height"], im["width"])
        gts = gt_masks_for(coco, iid, ann_by_img, hw)
        if not gts:
            continue
        n += 1
        preds = sorted(preds_by_img.get(iid, []), key=lambda z: -z["score"])

        # ---- top-1, what the pipeline would consume ----
        if preds:
            best = max(mask_iou(preds[0]["mask"], g["mask"]) for g in gts)
            gbest = max(gts, key=lambda g: mask_iou(preds[0]["mask"], g["mask"]))
            top_ious.append(best)
            top_bf1.append(boundary_f1(preds[0]["mask"], gbest["mask"]))
            hit50 += int(best >= 0.5)
            hit75 += int(best >= 0.75)
        else:
            top_ious.append(0.0); top_bf1.append(0.0)

        # ---- foreground mIoU, natural for the semantic model ----
        gfg = np.zeros(hw, np.uint8)
        for g in gts:
            gfg |= (g["mask"] > 0)
        pfg = np.zeros(hw, np.uint8)
        for p in preds:
            pfg |= (p["mask"] > 0)
        inter_sum += int(np.logical_and(pfg, gfg).sum())
        union_sum += int(np.logical_or(pfg, gfg).sum())

        # ---- AP ----
        used = {t: set() for t in thrs}
        for p in preds:
            scores.append(p["score"])
            for t in thrs:
                bi, bk = 0.0, -1
                for k, g in enumerate(gts):
                    if k in used[t]:
                        continue
                    if use_class and g["category_id"] != p["category_id"]:
                        continue
                    v = mask_iou(p["mask"], g["mask"])
                    if v > bi:
                        bi, bk = v, k
                hit = bi >= t and bk >= 0
                if hit:
                    used[t].add(bk)
                matched[t].append(1.0 if hit else 0.0)

    aps = {t: voc_ap(scores, matched[t], n_gt) for t in thrs}
    return {
        "AP50": aps[thrs[0]],
        "AP75": aps[[t for t in thrs if abs(t - 0.75) < 1e-6][0]],
        "AP": float(np.nanmean(list(aps.values()))),
        "top1_mask_iou": float(np.mean(top_ious)) if top_ious else float("nan"),
        "top1_hit@0.5": hit50 / max(n, 1),
        "top1_hit@0.75": hit75 / max(n, 1),
        "boundary_f1": float(np.mean(top_bf1)) if top_bf1 else float("nan"),
        "mIoU_fg": inter_sum / max(union_sum, 1),
        "n_eval_images": n,
    }


# ============================================================
# DATA
# ============================================================
class CocoSeg:
    def __init__(self, img_dir, ann_json, train):
        import torch
        from torchvision import transforms as T
        self.torch = torch
        with open(ann_json) as f:
            c = json.load(f)
        self.imgs = {im["id"]: im for im in c["images"]}
        self.ids = sorted(self.imgs)
        self.anns = defaultdict(list)
        for a in c["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.img_dir, self.train = img_dir, train
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image
        torch = self.torch
        iid = self.ids[i]
        info = self.imgs[iid]
        im = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        h, w = info["height"], info["width"]

        boxes, labels, masks = [], [], []
        for a in self.anns[iid]:
            m = np.zeros((h, w), np.uint8)
            for poly in a["segmentation"]:
                pts = np.asarray(poly, np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
            if m.sum() < 20:
                continue
            ys, xs = np.nonzero(m)
            boxes.append([float(xs.min()), float(ys.min()),
                          float(xs.max() + 1), float(ys.max() + 1)])
            labels.append(a["category_id"])
            masks.append(m)

        img = self.to_tensor(im)
        if self.train and masks and random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            masks = [np.ascontiguousarray(m[:, ::-1]) for m in masks]
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]

        t = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": (torch.as_tensor(np.stack(masks), dtype=torch.uint8)
                      if masks else torch.zeros((0, h, w), dtype=torch.uint8)),
            "image_id": torch.tensor([iid]),
        }
        return img, t


def collate(b):
    return tuple(zip(*b))


class SemSeg:
    """Frames + class-index PNG masks, for the semantic model."""

    def __init__(self, img_dir, mask_dir, size, train):
        import torch
        self.torch = torch
        self.files = sorted(Path(img_dir).glob("*.jpg"))
        self.mask_dir = Path(mask_dir)
        self.size, self.train = size, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        torch = self.torch
        p = self.files[i]
        img = cv2.imread(str(p))[:, :, ::-1]
        m = cv2.imread(str(self.mask_dir / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if m is None:
            m = np.zeros(img.shape[:2], np.uint8)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return (torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32),
                torch.as_tensor(np.ascontiguousarray(m), dtype=torch.long),
                p.stem)


# ============================================================
# MODELS
# ============================================================
def run_mask_rcnn(args, n_classes, paths, device):
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    tr = CocoSeg(paths["img_train"], paths["ann_train"], True)
    if args.limit_train:
        tr.ids = tr.ids[:args.limit_train]
    te = CocoSeg(paths["img_test"], paths["ann_test"], False)
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, collate_fn=collate,
                       pin_memory=True, drop_last=True,
                       persistent_workers=args.workers > 0,
                       prefetch_factor=(4 if args.workers > 0 else None))
    dl_te = DataLoader(te, batch_size=1, shuffle=False,
                       num_workers=args.workers, collate_fn=collate,
                       persistent_workers=args.workers > 0)
    print(f"    train {len(tr)} images, {len(dl_tr)} steps/epoch  |  "
          f"test {len(te)} images")

    # min_size=800 (the default) upscales a 1024x576 frame and roughly triples
    # the cost; keeping the native short side is much faster at similar quality
    m = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT", min_size=args.min_size, max_size=args.max_size,
        trainable_backbone_layers=args.trainable_layers)
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, n_classes + 1)
    inm = m.roi_heads.mask_predictor.conv5_mask.in_channels
    m.roi_heads.mask_predictor = MaskRCNNPredictor(inm, 256, n_classes + 1)
    m = m.to(device)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr or 1e-4, weight_decay=1e-4)
    steps = max(len(dl_tr), 1) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=(args.lr or 1e-4),
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    from tqdm import tqdm
    t0 = time.time()
    for ep in range(args.epochs):
        m.train(); tot, seen = 0.0, 0
        bar = tqdm(dl_tr, desc=f"    ep {ep+1}/{args.epochs}", leave=False,
                   dynamic_ncols=True)
        for imgs, tgts in bar:
            if all(len(t["boxes"]) == 0 for t in tgts):
                continue
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = sum(m(imgs, tgts).values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); seen += 1
            bar.set_postfix(loss=f"{tot/max(seen,1):.3f}")
        bar.close()
        el = (time.time() - t0) / 60.0
        eta = el / (ep + 1) * (args.epochs - ep - 1)
        print(f"    ep {ep+1}/{args.epochs}  loss {tot/max(seen,1):.3f}  "
              f"elapsed {el:.1f} min  eta {eta:.1f} min")

    m.eval()
    out = defaultdict(list)
    with torch.no_grad():
        for imgs, tgts in tqdm(dl_te, desc="    inference", leave=False,
                               dynamic_ncols=True):
            iid = int(tgts[0]["image_id"].item())
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                o = m([imgs[0].to(device)])[0]
            for msk, s, l in zip(o["masks"].cpu().numpy(),
                                 o["scores"].cpu().numpy(),
                                 o["labels"].cpu().numpy()):
                if s < args.score_thresh:
                    continue
                out[iid].append({"mask": (msk[0] > 0.5).astype(np.uint8),
                                 "score": float(s), "category_id": int(l)})
    return out, (time.time() - t0) / 60.0


def run_deeplab(args, n_classes, paths, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    try:
        import segmentation_models_pytorch as smp
        model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet",
                                  classes=n_classes + 1)
        tag = "DeepLabV3+ (smp)"
    except ImportError:
        import torchvision
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights="DEFAULT", aux_loss=True)
        model.classifier[4] = torch.nn.Conv2d(256, n_classes + 1, 1)
        model.aux_classifier[4] = torch.nn.Conv2d(256, n_classes + 1, 1)
        tag = "DeepLabV3 (torchvision fallback)"
    print(f"    using {tag}")
    model = model.to(device)

    tr = SemSeg(paths["img_train"], paths["mask_train"], args.seg_size, True)
    te = SemSeg(paths["img_test"], paths["mask_test"], args.seg_size, False)
    dl_tr = DataLoader(tr, batch_size=args.batch_size * 2, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_te = DataLoader(te, batch_size=1, shuffle=False, num_workers=args.workers)

    # foreground is a small fraction of the frame, so weight it up
    w = torch.ones(n_classes + 1, device=device)
    w[0] = args.bg_weight
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr or 3e-4, weight_decay=1e-4)
    steps = max(len(dl_tr), 1) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=(args.lr or 3e-4),
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    from tqdm import tqdm
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot, seen = 0.0, 0
        bar = tqdm(dl_tr, desc=f"    ep {ep+1}/{args.epochs}", leave=False,
                   dynamic_ncols=True)
        for x, y, _ in bar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                o = model(x)
                logits = o["out"] if isinstance(o, dict) else o
                loss = crit(logits, y)
                if isinstance(o, dict) and "aux" in o:
                    loss = loss + 0.4 * crit(o["aux"], y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss); seen += 1
            bar.set_postfix(loss=f"{tot/max(seen,1):.3f}")
        bar.close()
        el = (time.time() - t0) / 60.0
        print(f"    ep {ep+1}/{args.epochs}  loss {tot/max(seen,1):.3f}  "
              f"elapsed {el:.1f} min  "
              f"eta {el/(ep+1)*(args.epochs-ep-1):.1f} min")

    # map predictions back to full resolution and to COCO image ids
    with open(paths["ann_test"]) as f:
        coco = json.load(f)
    stem_to_id = {os.path.splitext(im["file_name"])[0]: (im["id"], im["height"], im["width"])
                  for im in coco["images"]}

    model.eval()
    out = defaultdict(list)
    with torch.no_grad():
        for x, _y, stems in dl_te:
            stem = stems[0]
            if stem not in stem_to_id:
                continue
            iid, H, W = stem_to_id[stem]
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                o = model(x.to(device))
                logits = o["out"] if isinstance(o, dict) else o
            prob = F.softmax(logits.float(), 1)
            prob = F.interpolate(prob, size=(H, W), mode="bilinear",
                                 align_corners=False)[0].cpu().numpy()
            lab = prob.argmax(0)
            for c in range(1, n_classes + 1):
                m = (lab == c).astype(np.uint8)
                if m.sum() < 30:
                    continue
                out[iid].append({"mask": m, "category_id": c,
                                 "score": float(prob[c][m > 0].mean())})
    return out, (time.time() - t0) / 60.0


def run_yolo_seg(args, paths, label_mode, out_dir):
    from ultralytics import YOLO
    t0 = time.time()
    model = YOLO(args.yolo_weights)
    model.train(data=paths["yolo_yaml"], epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch_size, project=str(out_dir),
                name=f"yolo9seg_{label_mode}", exist_ok=True, verbose=False,
                seed=args.seed, patience=args.epochs)

    with open(paths["ann_test"]) as f:
        coco = json.load(f)
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    file_to = {im["file_name"]: (im["id"], im["height"], im["width"])
               for im in coco["images"]}

    out = defaultdict(list)
    for r in model.predict(source=paths["img_test"], imgsz=args.imgsz,
                           conf=args.score_thresh, verbose=False, stream=True,
                           retina_masks=True):
        key = os.path.basename(r.path)
        if key not in file_to or r.masks is None:
            continue
        iid, H, W = file_to[key]
        names = r.names
        for m, b in zip(r.masks.data.cpu().numpy(), r.boxes):
            mm = (m > 0.5).astype(np.uint8)
            if mm.shape != (H, W):
                mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
            out[iid].append({"mask": mm, "score": float(b.conf.item()),
                             "category_id": name_to_id.get(names[int(b.cls.item())], 1)})
    return out, (time.time() - t0) / 60.0


# ============================================================
# MAIN
# ============================================================
def _save_rows(rows, out_dir, label_mode):
    """Persist results immediately, so a later error cannot lose finished runs."""
    if not rows:
        return None
    out_csv = os.path.join(out_dir, f"results_{label_mode}.csv")
    df = pd.DataFrame(rows)
    if os.path.isfile(out_csv):
        try:
            prev = pd.read_csv(out_csv)
            keys = set(zip(df.get("model", []), df.get("label_mode", [])))
            prev = prev[~prev.apply(
                lambda r: (r.get("model"), r.get("label_mode")) in keys, axis=1)]
            df = pd.concat([prev, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./active_seg_data")
    ap.add_argument("--label_mode", default="agnostic",
                    choices=["agnostic", "material"])
    ap.add_argument("--model", default="mask_rcnn",
                    choices=list(INSTANCE_MODELS) + list(SEMANTIC_MODELS))
    ap.add_argument("--all_models", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO training size; 640 is much faster than 1024")
    ap.add_argument("--min_size", type=int, default=576,
                    help="Mask R-CNN shorter side; the default 800 upscales "
                         "these frames and wastes compute")
    ap.add_argument("--max_size", type=int, default=1024)
    ap.add_argument("--trainable_layers", type=int, default=3,
                    choices=[0,1,2,3,4,5])
    ap.add_argument("--limit_train", type=int, default=0,
                    help="use only N training images, for a quick smoke test")
    ap.add_argument("--seg_size", type=int, default=512,
                    help="training resolution for the semantic model")
    ap.add_argument("--bg_weight", type=float, default=0.2,
                    help="background class weight for DeepLabV3+; the active "
                         "object covers a small fraction of the frame")
    ap.add_argument("--score_thresh", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--yolo_weights", default="yolov9c-seg.pt")
    ap.add_argument("--import_preds", default=None,
                    help="COCO-format results JSON with RLE/polygon masks from a "
                         "model trained elsewhere, e.g. Cascade Mask R-CNN")
    ap.add_argument("--import_name", default="cascade_mask_rcnn")
    ap.add_argument("--out_dir", default="./active_seg_results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    import torch as _t
    _t.backends.cudnn.benchmark = True
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True
    d = Path(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    lm = args.label_mode
    paths = {
        "img_train": str(d / "images" / "train"),
        "img_test": str(d / "images" / "test"),
        "mask_train": str(d / f"masks_{lm}" / "train"),
        "mask_test": str(d / f"masks_{lm}" / "test"),
        "ann_train": str(d / "coco" / f"train_{lm}.json"),
        "ann_test": str(d / "coco" / f"test_{lm}.json"),
        "yolo_yaml": str(d / f"yolo_seg_{lm}" / "data.yaml"),
    }
    if not os.path.isfile(paths["ann_train"]):
        raise SystemExit(f"missing {paths['ann_train']} -- run the build script")

    with open(paths["ann_train"]) as f:
        n_classes = len(json.load(f)["categories"])

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print(f"Active-object SEGMENTATION | labels={lm} | {n_classes} class(es) | {device}")
    print("=" * 78)

    to_run = (list(INSTANCE_MODELS) + list(SEMANTIC_MODELS)) if args.all_models \
        else [args.model]
    rows = []
    for name in to_run:
        print(f"\n  === {name} ===")
        try:
            if name == "mask_rcnn":
                preds, mins = run_mask_rcnn(args, n_classes, paths, device)
            elif name == "deeplabv3plus":
                preds, mins = run_deeplab(args, n_classes, paths, device)
            elif name == "yolo9_seg":
                preds, mins = run_yolo_seg(args, paths, lm, Path(args.out_dir))
            else:
                continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("    [OOM] skipped; lower --batch_size / --imgsz / --seg_size")
                torch.cuda.empty_cache()
                continue
            raise
        m = evaluate(preds, paths["ann_test"])
        m.update({"model": name, "label_mode": lm, "epochs": args.epochs,
                  "minutes": mins})
        rows.append(m)
        # write after every model: a later failure must not discard hours of
        # finished training
        _save_rows(rows, args.out_dir, lm)
        print(f"    AP50 {m['AP50']:.3f}  top1_mask_IoU {m['top1_mask_iou']:.3f}  "
              f"boundaryF1 {m['boundary_f1']:.3f}  ({mins:.1f} min)")

    if args.import_preds and not os.path.isfile(args.import_preds):
        print(f"\n  [skip] --import_preds not found: {args.import_preds}")
        print("         Train that model first, or point this at a saved")
        print("         predictions file to re-score it without retraining.")
        args.import_preds = None

    if args.import_preds:
        print(f"\n  === {args.import_name} (imported) ===")
        with open(args.import_preds) as f:
            raw = json.load(f)
        with open(paths["ann_test"]) as f:
            coco = json.load(f)
        hw = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}
        preds = defaultdict(list)
        try:
            from pycocotools import mask as maskutil
        except ImportError:
            maskutil = None
        for r in raw:
            iid = r["image_id"]
            if iid not in hw:
                continue
            H, W = hw[iid]
            seg = r.get("segmentation")
            if isinstance(seg, dict) and maskutil is not None:
                m = maskutil.decode(seg).astype(np.uint8)
            elif isinstance(seg, list):
                m = np.zeros((H, W), np.uint8)
                for poly in seg:
                    pts = np.asarray(poly, np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(m, [pts], 1)
            else:
                continue
            preds[iid].append({"mask": m, "score": float(r.get("score", 1.0)),
                               "category_id": int(r.get("category_id", 1))})
        m = evaluate(preds, paths["ann_test"])
        m.update({"model": args.import_name, "label_mode": lm,
                  "epochs": np.nan, "minutes": np.nan})
        rows.append(m)
        print(f"    AP50 {m['AP50']:.3f}  top1_mask_IoU {m['top1_mask_iou']:.3f}")

    if not rows:
        raise SystemExit("nothing ran")

    out_csv = _save_rows(rows, args.out_dir, lm)
    res = pd.read_csv(out_csv)

    print("\n" + "=" * 78)
    print(f"RESULTS  labels={lm}")
    print("=" * 78)
    cols = ["model", "AP50", "AP75", "AP", "top1_mask_iou", "top1_hit@0.5",
            "boundary_f1", "mIoU_fg", "minutes"]
    print(res[[c for c in cols if c in res.columns]].round(3).to_string(index=False))
    print("\n  top1_mask_iou is what the annotation pipeline consumes: the mask of")
    print("  the highest-scoring prediction. boundary_f1 checks the outline rather")
    print("  than mere overlap. DeepLabV3+ is semantic, so it has no instances and")
    print("  its AP is not directly comparable -- read its mIoU_fg and top1 columns.")
    print(f"\nsaved -> {out_csv}")


CASCADE_NOTE = """
Cascade Mask R-CNN is not in torchvision. Train it in detectron2 and import:

  register_coco_instances("ego_train", {},
      "active_seg_data/coco/train_agnostic.json", "active_seg_data/images/train")
  register_coco_instances("ego_test", {},
      "active_seg_data/coco/test_agnostic.json", "active_seg_data/images/test")

  python tools/train_net.py --config-file \\
      configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml \\
      DATASETS.TRAIN '("ego_train",)' DATASETS.TEST '("ego_test",)' \\
      MODEL.ROI_HEADS.NUM_CLASSES <n> OUTPUT_DIR ./cascade_out

  python train_segmentation.py --data_dir ./active_seg_data \\
      --label_mode agnostic --import_preds cascade_out/coco_instances_results.json
"""

if __name__ == "__main__":
    main()
