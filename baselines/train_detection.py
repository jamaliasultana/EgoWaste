#!/usr/bin/env python3
"""
train_detection.py
=========================
Benchmarks detectors on ACTIVE (handheld) OBJECT DETECTION: find the one item
the expert is manipulating, inside a frame holding dozens of similar items.

MODELS
  faster_rcnn   torchvision fasterrcnn_resnet50_fpn_v2
  retinanet     torchvision retinanet_resnet50_fpn_v2
  yolo11s       ultralytics YOLO11-S
  tridentnet    detectron2; trained separately, then imported with
                --import_preds (see the note at the bottom)

METRICS
  AP50, AP75, AP    standard COCO-style detection quality
  top1_loc          the highest-scoring box matches a ground-truth active
                    object at IoU>=0.5. This is the metric that matters for
                    the annotation pipeline, which consumes ONE box per frame,
                    and it is not the same as AP.
  top1_loc_cls      as above, and the predicted class is also correct
  hit_rate@k        a correct box appears among the top k

Everything trains and is scored on the identical video-level split written by
prepare_detection_data.py, so the four detectors are comparable.

USAGE
  # class-agnostic first: can the model find the manipulated object at all?
  python train_detection.py --data_dir ./active_det_data \
      --label_mode agnostic --model yolo11s --epochs 50

  # then with the 15 material classes
  python train_detection.py --data_dir ./active_det_data \
      --label_mode material --model faster_rcnn --epochs 20

  # everything torchvision + yolo in one go
  python train_detection.py --data_dir ./active_det_data \
      --label_mode agnostic --all_models

REQUIREMENTS
  pip install torch torchvision pycocotools ultralytics
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TORCH_MODELS = ("faster_rcnn", "retinanet")


# ============================================================
# METRICS
# ============================================================
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


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


def ap_at(preds_by_img, gts_by_img, thr, use_class=True):
    scores, matched, n_gt = [], [], 0
    for img, gts in gts_by_img.items():
        n_gt += len(gts)
    for img, preds in preds_by_img.items():
        gts = gts_by_img.get(img, [])
        used = set()
        for p in sorted(preds, key=lambda z: -z["score"]):
            best, bk = 0.0, -1
            for k, g in enumerate(gts):
                if k in used:
                    continue
                if use_class and g["category_id"] != p["category_id"]:
                    continue
                v = iou_xywh(p["bbox"], g["bbox"])
                if v > best:
                    best, bk = v, k
            hit = best >= thr and bk >= 0
            if hit:
                used.add(bk)
            scores.append(p["score"]); matched.append(1.0 if hit else 0.0)
    return voc_ap(scores, matched, n_gt)


def evaluate(preds, gt_json, ks=(1, 3, 5)):
    """preds: list of {image_id, bbox xywh, score, category_id}"""
    with open(gt_json) as f:
        coco = json.load(f)
    gts_by_img = defaultdict(list)
    for a in coco["annotations"]:
        gts_by_img[a["image_id"]].append(a)
    preds_by_img = defaultdict(list)
    for p in preds:
        preds_by_img[p["image_id"]].append(p)

    out = {
        "AP50": ap_at(preds_by_img, gts_by_img, 0.50),
        "AP75": ap_at(preds_by_img, gts_by_img, 0.75),
        "AP": float(np.nanmean([ap_at(preds_by_img, gts_by_img, t)
                                for t in np.arange(0.5, 1.0, 0.05)])),
        "AP50_agnostic": ap_at(preds_by_img, gts_by_img, 0.50, use_class=False),
    }

    # the pipeline consumes ONE box per frame, so score that directly
    n = hit_loc = hit_cls = 0
    hits_k = {k: 0 for k in ks}
    for img, gts in gts_by_img.items():
        if not gts:
            continue
        n += 1
        ps = sorted(preds_by_img.get(img, []), key=lambda z: -z["score"])
        if not ps:
            continue
        if any(iou_xywh(ps[0]["bbox"], g["bbox"]) >= 0.5 for g in gts):
            hit_loc += 1
            if any(iou_xywh(ps[0]["bbox"], g["bbox"]) >= 0.5
                   and ps[0]["category_id"] == g["category_id"] for g in gts):
                hit_cls += 1
        for k in ks:
            if any(iou_xywh(p["bbox"], g["bbox"]) >= 0.5
                   for p in ps[:k] for g in gts):
                hits_k[k] += 1
    out["top1_loc"] = hit_loc / max(n, 1)
    out["top1_loc_cls"] = hit_cls / max(n, 1)
    for k in ks:
        out[f"hit_rate@{k}"] = hits_k[k] / max(n, 1)
    out["n_eval_images"] = n
    return out


# ============================================================
# TORCHVISION
# ============================================================
class CocoDet:
    """Minimal COCO detection dataset; avoids a pycocotools dependency."""

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
        self.cats = sorted(x["id"] for x in c["categories"])
        self.img_dir = img_dir
        self.train = train
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        from PIL import Image
        torch = self.torch
        iid = self.ids[i]
        im = Image.open(os.path.join(self.img_dir,
                                     self.imgs[iid]["file_name"])).convert("RGB")
        boxes, labels = [], []
        for a in self.anns[iid]:
            x, y, w, h = a["bbox"]
            if w < 2 or h < 2:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        img = self.to_tensor(im)
        if self.train and random.random() < 0.5:      # horizontal flip
            img = torch.flip(img, dims=[2])
            W = img.shape[2]
            boxes = [[W - b[2], b[1], W - b[0], b[3]] for b in boxes]
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([iid]),
        }
        return img, target


def collate(b):
    return tuple(zip(*b))


def build_torch_model(name, n_classes, args):
    """
    min_size/max_size matter a lot for speed. torchvision defaults to
    min_size=800, which UPSCALES a 1024x576 frame to about 800x1422 and roughly
    triples the cost. Setting min_size=576/max_size=1024 keeps the native
    resolution. trainable_backbone_layers<5 also cuts the backward pass.
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    kw = dict(min_size=args.min_size, max_size=args.max_size,
              trainable_backbone_layers=args.trainable_layers)
    if name == "faster_rcnn":
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT", **kw)
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(inf, n_classes)
        return m
    if name == "retinanet":
        import torch
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        m = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights="DEFAULT", **kw)
        na = m.head.classification_head.num_anchors
        inf = m.backbone.out_channels
        m.head.classification_head = RetinaNetClassificationHead(
            in_channels=inf, num_anchors=na, num_classes=n_classes,
            norm_layer=lambda c: torch.nn.GroupNorm(32, c))
        return m
    raise ValueError(name)


def run_torch(args, name, n_classes, paths, device):
    import torch
    from torch.utils.data import DataLoader

    tr = CocoDet(paths["img_train"], paths["ann_train"], True)
    if args.limit_train:
        tr.ids = tr.ids[:args.limit_train]
    te = CocoDet(paths["img_test"], paths["ann_test"], False)
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

    model = build_torch_model(name, n_classes + 1, args).to(device)  # +1 background
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr or 1e-4, weight_decay=1e-4)
    steps = max(len(dl_tr), 1) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=(args.lr or 1e-4), total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    from tqdm import tqdm
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, seen = 0.0, 0
        bar = tqdm(dl_tr, desc=f"    ep {ep+1}/{args.epochs}", leave=False,
                   dynamic_ncols=True)
        for imgs, tgts in bar:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            if all(len(t["boxes"]) == 0 for t in tgts):
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                losses = model(imgs, tgts)
                loss = sum(losses.values())
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

    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, tgts in tqdm(dl_te, desc="    inference", leave=False,
                               dynamic_ncols=True):
            img = imgs[0].to(device)
            iid = int(tgts[0]["image_id"].item())
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                o = model([img])[0]
            for b, s, l in zip(o["boxes"].cpu().numpy(),
                               o["scores"].cpu().numpy(),
                               o["labels"].cpu().numpy()):
                if s < args.score_thresh:
                    continue
                x1, y1, x2, y2 = b
                preds.append({"image_id": iid, "score": float(s),
                              "category_id": int(l),
                              "bbox": [float(x1), float(y1),
                                       float(x2 - x1), float(y2 - y1)]})
    return preds, (time.time() - t0) / 60.0


# ============================================================
# YOLO
# ============================================================
def run_yolo(args, paths, label_mode, out_dir):
    from ultralytics import YOLO
    t0 = time.time()
    model = YOLO(args.yolo_weights)
    model.train(data=paths["yolo_yaml"], epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch_size,
                project=str(out_dir), name=f"yolo_{label_mode}",
                exist_ok=True, verbose=False, seed=args.seed,
                patience=args.epochs)

    with open(paths["ann_test"]) as f:
        coco = json.load(f)
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    file_to_img = {im["file_name"]: im["id"] for im in coco["images"]}

    preds = []
    res = model.predict(source=paths["img_test"], imgsz=args.imgsz,
                        conf=args.score_thresh, verbose=False, stream=True)
    for r in res:
        iid = file_to_img.get(os.path.basename(r.path))
        if iid is None:
            continue
        names = r.names
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cname = names[int(b.cls.item())]
            preds.append({"image_id": iid, "score": float(b.conf.item()),
                          "category_id": name_to_id.get(cname, 1),
                          "bbox": [x1, y1, x2 - x1, y2 - y1]})
    return preds, (time.time() - t0) / 60.0


# ============================================================
# MAIN
# ============================================================
def _save_rows(rows, args, name_fmt):
    """Persist results immediately, so a later error cannot lose finished runs."""
    if not rows:
        return None
    import pandas as _pd
    out_csv = os.path.join(args.out_dir, name_fmt.format(mode=args.label_mode))
    df = _pd.DataFrame(rows)
    if os.path.isfile(out_csv):
        try:
            prev = _pd.read_csv(out_csv)
            keys = set(zip(df.get("model", []), df.get("label_mode", [])))
            prev = prev[~prev.apply(
                lambda r: (r.get("model"), r.get("label_mode")) in keys, axis=1)]
            df = _pd.concat([prev, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./active_det_data")
    ap.add_argument("--label_mode", default="agnostic",
                    choices=["agnostic", "material"])
    ap.add_argument("--model", default="yolo11s",
                    choices=list(TORCH_MODELS) + ["yolo11s"])
    ap.add_argument("--all_models", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO training size; 640 is much faster than 1024 and "
                         "usually within a point of it")
    ap.add_argument("--min_size", type=int, default=576,
                    help="torchvision detector shorter side; the default 800 "
                         "upscales these frames and wastes compute")
    ap.add_argument("--max_size", type=int, default=1024)
    ap.add_argument("--trainable_layers", type=int, default=3, choices=[0,1,2,3,4,5],
                    help="trainable backbone stages; 3 trains ~half the trunk")
    ap.add_argument("--limit_train", type=int, default=0,
                    help="use only N training images, for a quick smoke test")
    ap.add_argument("--score_thresh", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--yolo_weights", default="yolo11s.pt")
    ap.add_argument("--import_preds", default=None,
                    help="COCO-format predictions JSON from a detector trained "
                         "elsewhere (e.g. TridentNet in detectron2); scored with "
                         "the same metrics")
    ap.add_argument("--import_name", default="tridentnet")
    ap.add_argument("--out_dir", default="./active_det_results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    import torch as _t
    _t.backends.cudnn.benchmark = True          # fixed-ish input sizes
    _t.backends.cuda.matmul.allow_tf32 = True
    _t.backends.cudnn.allow_tf32 = True
    d = Path(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = {
        "img_train": str(d / "images" / "train"),
        "img_val": str(d / "images" / "val"),
        "img_test": str(d / "images" / "test"),
        "ann_train": str(d / "coco" / f"train_{args.label_mode}.json"),
        "ann_val": str(d / "coco" / f"val_{args.label_mode}.json"),
        "ann_test": str(d / "coco" / f"test_{args.label_mode}.json"),
        "yolo_yaml": str(d / f"yolo_{args.label_mode}" / "data.yaml"),
    }
    for k in ("ann_train", "ann_test"):
        if not os.path.isfile(paths[k]):
            raise SystemExit(f"missing {paths[k]} -- run the build script first")

    with open(paths["ann_train"]) as f:
        n_classes = len(json.load(f)["categories"])

    print("=" * 78)
    print(f"Active-object DETECTION | labels={args.label_mode} | "
          f"{n_classes} class(es)")
    print("=" * 78)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device {device}\n")

    res_name_fmt = "results_{mode}.csv"
    to_run = (list(TORCH_MODELS) + ["yolo11s"]) if args.all_models else [args.model]
    rows = []

    for name in to_run:
        print(f"  === {name} ===")
        try:
            if name == "yolo11s":
                preds, mins = run_yolo(args, paths, args.label_mode,
                                       Path(args.out_dir))
            else:
                preds, mins = run_torch(args, name, n_classes, paths, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    [OOM] skipped; lower --batch_size or --imgsz")
                torch.cuda.empty_cache()
                continue
            raise
        m = evaluate(preds, paths["ann_test"])
        m.update({"model": name, "label_mode": args.label_mode,
                  "epochs": args.epochs, "minutes": mins})
        rows.append(m)
        with open(os.path.join(args.out_dir,
                               f"preds_{name}_{args.label_mode}.json"), "w") as f:
            json.dump(preds, f)
        # write after every model: a later failure must not discard hours of
        # finished training
        _save_rows(rows, args, res_name_fmt)
        print(f"    AP50 {m['AP50']:.3f}  top1_loc {m['top1_loc']:.3f}  "
              f"({mins:.1f} min)")

    if args.import_preds and not os.path.isfile(args.import_preds):
        print(f"\n  [skip] --import_preds not found: {args.import_preds}")
        print("         Train that model first, or point this at any")
        print("         preds_*.json already written here to re-score it")
        print("         without retraining.")
        args.import_preds = None

    if args.import_preds:
        with open(args.import_preds) as f:
            preds = json.load(f)
        m = evaluate(preds, paths["ann_test"])
        m.update({"model": args.import_name, "label_mode": args.label_mode,
                  "epochs": np.nan, "minutes": np.nan})
        rows.append(m)
        print(f"  imported {args.import_name}: AP50 {m['AP50']:.3f}  "
              f"top1_loc {m['top1_loc']:.3f}")

    if not rows:
        raise SystemExit("nothing ran")

    out_csv = _save_rows(rows, args, res_name_fmt)
    res = pd.read_csv(out_csv)

    print("\n" + "=" * 78)
    print(f"RESULTS  labels={args.label_mode}")
    print("=" * 78)
    cols = ["model", "AP50", "AP75", "AP", "top1_loc", "top1_loc_cls",
            "hit_rate@3", "minutes"]
    print(res[[c for c in cols if c in res.columns]].round(3).to_string(index=False))
    print("\n  top1_loc is the number the annotation pipeline cares about: the")
    print("  highest-scoring box is the one it would consume. AP rewards a good")
    print("  ranking over many boxes, which is a different question.")
    print(f"\nsaved -> {out_csv}")


# ----------------------------------------------------------------
# TridentNet, via detectron2 (trained separately, then imported)
# ----------------------------------------------------------------
TRIDENT_NOTE = """
TridentNet lives in detectron2's projects/ and is not pip-installable, so train
it separately and import its predictions:

  1. register the dataset in a detectron2 script
       from detectron2.data.datasets import register_coco_instances
       register_coco_instances("ego_train", {},
           "active_det_data/coco/train_agnostic.json",
           "active_det_data/images/train")
       register_coco_instances("ego_test", {},
           "active_det_data/coco/test_agnostic.json",
           "active_det_data/images/test")

  2. train with the TridentNet config
       cd detectron2/projects/TridentNet
       python train_net.py --config-file configs/tridentnet_fast_R_50_C4_1x.yaml \\
           DATASETS.TRAIN '("ego_train",)' DATASETS.TEST '("ego_test",)' \\
           MODEL.ROI_HEADS.NUM_CLASSES <n> OUTPUT_DIR ./trident_out

  3. dump predictions as COCO json, then
       python train_detection.py --data_dir ./active_det_data \\
           --label_mode agnostic --import_preds trident_out/coco_instances_results.json
"""

if __name__ == "__main__":
    main()
