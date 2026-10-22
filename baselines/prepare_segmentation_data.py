#!/usr/bin/env python3
"""
prepare_segmentation_data.py
====================================
Builds an INSTANCE SEGMENTATION dataset for the active (handheld) object.

The task: in a cluttered egocentric frame, produce the pixel mask of the object
the expert is manipulating -- its exact shape, size and boundary -- not just a
box, and not every waste item in view.

Three formats are written from ONE video-level split, so every model trains and
is scored on identical data:

  coco/{split}_{mode}.json        polygons, for Mask R-CNN / Cascade Mask R-CNN
  masks/{split}/*.png            single-channel class-index masks, for DeepLabV3+
  yolo_seg_{mode}/               normalised polygons + data.yaml, for YOLO-seg

Label modes:
  agnostic   one class, "active_object" -- isolates shape/localisation
  material   the 15 material categories -- adds recognition

LEAKAGE CONTROL
  * split by VIDEO, never random
  * at most --max_per_tracklet frames per object
  * videos without a meta CSV are skipped, not zero-filled

USAGE
  python prepare_segmentation_data.py \
     --roots "/path/to/egowaste_annotations/Brookhaven Landfill 20250602 Preproc" \
             "/path/to/egowaste_annotations/Brookhaven MRF 20250604 Preproc" \
     --frames_roots "/path/to/egowaste_frames/Brookhaven Landfill 20250602 Preproc" \
                    "/path/to/egowaste_frames/Brookhaven MRF 20250604 Preproc" \
     --cameras 1 2 3 4 5 --out_dir ./active_seg_data
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import annotation_utils as E
from prepare_classification_data import parse_labels, load_mask, find_frame_image
from prepare_detection_data import video_level_split

SPLITS = ("train", "val", "test")
MIN_POLY_AREA = 60          # ignore specks
MIN_POLY_POINTS = 3


def mask_to_polygons(mask, epsilon_frac=0.004):
    """
    Contours -> flat [x1,y1,x2,y2,...] polygons, lightly simplified.
    Small holes are dropped; a multi-part object keeps all its parts.
    """
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_POLY_AREA:
            continue
        eps = epsilon_frac * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < MIN_POLY_POINTS:
            continue
        polys.append(c.astype(np.float32).flatten().tolist())
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--frames_roots", nargs="*", default=[])
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gt_globs", nargs="+", default=None)
    ap.add_argument("--max_per_tracklet", type=int, default=4)
    ap.add_argument("--min_class_count", type=int, default=20)
    ap.add_argument("--copy_images", action="store_true", default=False)
    ap.add_argument("--out_dir", default="./active_seg_data")
    args = ap.parse_args()

    frames_roots = args.frames_roots + args.roots
    out = Path(args.out_dir)
    E.ensure_dir(out)

    print("=" * 78)
    print("Building the active-object SEGMENTATION dataset")
    print("=" * 78)

    per_frame = defaultdict(list)
    frame_src, skipped = {}, []

    for root in args.roots:
        day = Path(root).name
        for cam in args.cameras:
            cam_dir = E.get_camera_dir(root, cam)
            if cam_dir is None:
                continue
            for gx_dir in sorted(cam_dir.glob("GX*")):
                gt_csv, _ = E.find_gt_csv(gx_dir, args.gt_globs)
                if gt_csv is None:
                    skipped.append(f"{day}|C{cam}_{gx_dir.name}")
                    continue
                df = pd.read_csv(gt_csv, low_memory=False)
                if "integrated_review_status" in df.columns:
                    df = df[df["integrated_review_status"] != "deleted_false_object"]
                df = df.sort_values("frame_name")

                video = f"{day}|C{cam}_{gx_dir.name}"
                per_track = defaultdict(int)

                for _, r in tqdm(df.iterrows(), total=len(df),
                                 desc=video[-24:], leave=False):
                    fname = E.normalize_frame_name(r.get("frame_name"))
                    if fname is None:
                        continue
                    mat, _ = parse_labels(r)
                    if mat is None:
                        continue
                    oid = E.safe_int(r.get("object_track_id"))
                    if args.max_per_tracklet and per_track[oid] >= args.max_per_tracklet:
                        continue

                    mask = load_mask(r)          # the verified object mask
                    if mask is None or mask.sum() < MIN_POLY_AREA:
                        continue
                    polys = mask_to_polygons(mask)
                    if not polys:
                        continue
                    ys, xs = np.nonzero(mask)
                    bbox = [float(xs.min()), float(ys.min()),
                            float(xs.max() - xs.min()), float(ys.max() - ys.min())]

                    key = (video, fname)
                    if key not in frame_src:
                        p = find_frame_image(frames_roots, cam, gx_dir.name, fname,
                                             [r.get("image_path"),
                                              r.get("source_image_path")])
                        if p is None:
                            continue
                        frame_src[key] = p

                    per_track[oid] += 1
                    per_frame[key].append({
                        "polys": polys, "bbox": bbox, "material": mat,
                        "area": float((mask > 0).sum()),
                        "mask_path": (str(r.get("corrected_mask_path"))
                                      if not E.is_missing(r.get("corrected_mask_path"))
                                      else str(r.get("mask_path"))),
                        "object_track_id": oid})

    if not per_frame:
        raise RuntimeError("no annotated masks found -- check --gt_globs")

    from prepare_classification_data import report_unmapped
    report_unmapped()

    # ---------- classes ----------
    counts = defaultdict(int)
    for anns in per_frame.values():
        for a in anns:
            counts[a["material"]] += 1
    keep = {c for c, n in counts.items() if n >= args.min_class_count}
    for anns in per_frame.values():
        for a in anns:
            a["material"] = a["material"] if a["material"] in keep else "other"
    materials = sorted({a["material"] for anns in per_frame.values() for a in anns})
    if len(materials) > 20:
        print(f"\n[!] {len(materials)} 'material' classes resolved -- the taxonomy "
              f"has 15. The labels are probably still fine-grained waste classes; "
              f"check taxonomy.py is importable from this folder.")

    # ---------- split ----------
    vsize = defaultdict(int)
    for (video, _f) in per_frame:
        vsize[video] += 1
    assign = video_level_split(vsize)

    for s in SPLITS:
        E.ensure_dir(out / "images" / s)
        E.ensure_dir(out / "masks_agnostic" / s)
        E.ensure_dir(out / "masks_material" / s)
    E.ensure_dir(out / "coco")

    mat_idx = {c: i + 1 for i, c in enumerate(materials)}   # 0 = background
    records = []

    for (video, fname), anns in tqdm(sorted(per_frame.items()), desc="writing"):
        split = assign[video]
        stem = f"{video.replace('|','_').replace(' ','')}__{fname}"
        dst = out / "images" / split / f"{stem}.jpg"
        src = frame_src[(video, fname)]
        if not dst.exists():
            if args.copy_images:
                shutil.copyfile(src, dst)
            else:
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copyfile(src, dst)
        im = cv2.imread(str(dst))
        if im is None:
            continue
        h, w = im.shape[:2]

        # semantic masks for DeepLabV3+
        sem_a = np.zeros((h, w), np.uint8)
        sem_m = np.zeros((h, w), np.uint8)
        for a in anns:
            m = cv2.imread(a["mask_path"], cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            sel = m > 127
            sem_a[sel] = 1
            sem_m[sel] = mat_idx[a["material"]]
        cv2.imwrite(str(out / "masks_agnostic" / split / f"{stem}.png"), sem_a)
        cv2.imwrite(str(out / "masks_material" / split / f"{stem}.png"), sem_m)

        records.append({"video": video, "frame_name": fname, "split": split,
                        "stem": stem, "file": str(dst), "width": w, "height": h,
                        "anns": anns})

    pd.DataFrame([{k: v for k, v in r.items() if k != "anns"}
                  for r in records]).to_csv(out / "split.csv", index=False)

    # ---------- COCO with polygons ----------
    def write_coco(split, mode):
        cats = ["active_object"] if mode == "agnostic" else materials
        cid = {c: i + 1 for i, c in enumerate(cats)}
        images, annos, aid = [], [], 1
        for i, r in enumerate([x for x in records if x["split"] == split], 1):
            images.append({"id": i, "file_name": os.path.basename(r["file"]),
                           "width": r["width"], "height": r["height"]})
            for a in r["anns"]:
                c = "active_object" if mode == "agnostic" else a["material"]
                annos.append({"id": aid, "image_id": i, "category_id": cid[c],
                              "bbox": a["bbox"], "area": a["area"],
                              "segmentation": a["polys"], "iscrowd": 0})
                aid += 1
        with open(out / "coco" / f"{split}_{mode}.json", "w") as f:
            json.dump({"images": images, "annotations": annos,
                       "categories": [{"id": cid[c], "name": c} for c in cats]}, f)
        return len(images), len(annos), cats

    print("\nCOCO (polygons)")
    for mode in ("agnostic", "material"):
        for s in SPLITS:
            ni, na, cats = write_coco(s, mode)
            print(f"  {mode:<9s} {s:<5s} images {ni:>5d}  masks {na:>5d}  "
                  f"classes {len(cats)}")

    # ---------- YOLO-seg ----------
    for mode in ("agnostic", "material"):
        cats = ["active_object"] if mode == "agnostic" else materials
        idx = {c: i for i, c in enumerate(cats)}
        root = out / f"yolo_seg_{mode}"
        for s in SPLITS:
            E.ensure_dir(root / "images" / s)
            E.ensure_dir(root / "labels" / s)
        for r in records:
            s = r["split"]
            link = root / "images" / s / f"{r['stem']}.jpg"
            if not link.exists():
                try:
                    os.symlink(os.path.abspath(r["file"]), link)
                except OSError:
                    shutil.copyfile(r["file"], link)
            lines = []
            for a in r["anns"]:
                c = "active_object" if mode == "agnostic" else a["material"]
                for poly in a["polys"]:
                    pts = np.asarray(poly, np.float32).reshape(-1, 2)
                    pts[:, 0] /= r["width"]; pts[:, 1] /= r["height"]
                    pts = np.clip(pts, 0.0, 1.0)
                    if len(pts) < MIN_POLY_POINTS:
                        continue
                    lines.append(f"{idx[c]} " +
                                 " ".join(f"{v:.6f}" for v in pts.flatten()))
            (root / "labels" / s / f"{r['stem']}.txt").write_text("\n".join(lines))
        (root / "data.yaml").write_text(
            f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
            f"test: images/test\nnames:\n" +
            "".join(f"  {i}: {c}\n" for i, c in enumerate(cats)))
        print(f"  yolo_seg_{mode}/data.yaml written ({len(cats)} classes)")

    n_mask = sum(len(r["anns"]) for r in records)
    print(f"\nframes {len(records)}   masks {n_mask}   videos {len(vsize)}")
    for s in SPLITS:
        sub = [r for r in records if r["split"] == s]
        print(f"  {s:<5s} {len(sub):>5d} frames  "
              f"{len({r['video'] for r in sub})} videos")
    print(f"material classes ({len(materials)}): {materials}")
    if skipped:
        print(f"skipped videos without a meta CSV: {len(skipped)}")

    E.save_json({"materials": materials, "material_index": mat_idx,
                 "n_frames": len(records), "n_masks": n_mask,
                 "skipped_videos": skipped, "split_assignment": assign},
                str(out / "build_summary.json"))
    print(f"\ndataset -> {out}")


if __name__ == "__main__":
    main()
