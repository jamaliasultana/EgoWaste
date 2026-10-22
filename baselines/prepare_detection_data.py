#!/usr/bin/env python3
"""
prepare_detection_data.py
=================================
Builds a DETECTION dataset for the active (handheld) object.

The task: given a full cluttered egocentric frame containing dozens of waste
items, output the box of the object the expert is actually manipulating. The
detector must not simply find waste -- it must find the ONE item in the hand.

Two label settings are written, and both matter:
  agnostic   a single class, "active_object". Isolates localisation: can the
             model find the manipulated item at all?
  material   the 15 material categories. Adds recognition on top.

Outputs (same split, two formats, so every detector sees identical data):
  <out>/images/{train,val,test}/*.jpg          full frames, symlinked or copied
  <out>/coco/{train,val,test}_{agnostic,material}.json
  <out>/yolo_{agnostic,material}/...           labels + data.yaml
  <out>/split.csv                              frame -> split, for auditing

LEAKAGE CONTROL
  * split is by VIDEO, never random
  * at most --max_per_tracklet frames per object, so one long sort cannot
    dominate
  * videos without a meta CSV are skipped, not zero-filled

USAGE
  python prepare_detection_data.py \
     --roots "/path/to/egowaste_annotations/Brookhaven Landfill 20250602 Preproc" \
             "/path/to/egowaste_annotations/Brookhaven MRF 20250604 Preproc" \
     --frames_roots "/path/to/egowaste_frames/Brookhaven Landfill 20250602 Preproc" \
                    "/path/to/egowaste_frames/Brookhaven MRF 20250604 Preproc" \
     --cameras 1 2 3 4 5 --out_dir ./active_det_data
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import annotation_utils as E
from prepare_classification_data import parse_labels, load_mask, get_bbox, find_frame_image

SPLITS = ("train", "val", "test")


def video_level_split(videos_sizes, val_frac=0.15, test_frac=0.15, seed=2026):
    """Largest-first, into the split with most remaining absolute capacity."""
    sizes = dict(sorted(videos_sizes.items(), key=lambda kv: -kv[1]))
    total = float(sum(sizes.values()))
    quota = {"train": (1 - val_frac - test_frac) * total,
             "val": val_frac * total, "test": test_frac * total}
    got = {k: 0.0 for k in quota}
    assign, rng = {}, random.Random(seed)
    for v, n in sizes.items():
        room = {k: quota[k] - got[k] for k in quota}
        best = max(room, key=lambda k: (room[k], rng.random()))
        assign[v] = best
        got[best] += float(n)
    for k in ("val", "test"):
        if got[k] == 0:
            donor = max(got, key=lambda z: got[z])
            cand = [v for v, s in assign.items() if s == donor]
            if len(cand) > 1:
                sm = min(cand, key=lambda v: sizes[v])
                assign[sm] = k
                got[k] += sizes[sm]; got[donor] -= sizes[sm]
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--frames_roots", nargs="*", default=[])
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gt_globs", nargs="+", default=None)
    ap.add_argument("--max_per_tracklet", type=int, default=4)
    ap.add_argument("--min_class_count", type=int, default=20,
                    help="material classes rarer than this are merged into 'other'")
    ap.add_argument("--copy_images", action="store_true", default=False,
                    help="copy frames instead of symlinking (needed on some FS)")
    ap.add_argument("--out_dir", default="./active_det_data")
    args = ap.parse_args()

    frames_roots = args.frames_roots + args.roots
    out = Path(args.out_dir)
    E.ensure_dir(out)

    print("=" * 78)
    print("Building the active-object DETECTION dataset")
    print("=" * 78)

    # ---------- collect annotations ----------
    per_frame = defaultdict(list)     # (video, frame) -> [ {bbox, material} ]
    frame_src = {}
    skipped = []

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

                    mask = load_mask(r)
                    bbox = get_bbox(r, mask)
                    if bbox is None or bbox[2] < 4 or bbox[3] < 4:
                        continue

                    key = (video, fname)
                    if key not in frame_src:
                        p = find_frame_image(frames_roots, cam, gx_dir.name, fname,
                                             [r.get("image_path"),
                                              r.get("source_image_path")])
                        if p is None:
                            continue
                        frame_src[key] = p

                    per_track[oid] += 1
                    per_frame[key].append({"bbox": [float(v) for v in bbox],
                                           "material": mat,
                                           "object_track_id": oid})

    if not per_frame:
        raise RuntimeError("no annotated frames found -- check --gt_globs")

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

    # ---------- split by video ----------
    vsize = defaultdict(int)
    for (video, _f) in per_frame:
        vsize[video] += 1
    assign = video_level_split(vsize)

    # ---------- write images + records ----------
    for s in SPLITS:
        E.ensure_dir(out / "images" / s)
    E.ensure_dir(out / "coco")

    records = []
    for (video, fname), anns in tqdm(sorted(per_frame.items()), desc="images"):
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
        records.append({"video": video, "frame_name": fname, "split": split,
                        "stem": stem, "file": str(dst), "width": w, "height": h,
                        "anns": anns})

    pd.DataFrame([{k: v for k, v in r.items() if k != "anns"}
                  for r in records]).to_csv(out / "split.csv", index=False)

    # ---------- COCO ----------
    def write_coco(split, label_mode):
        cats = (["active_object"] if label_mode == "agnostic" else materials)
        cat_id = {c: i + 1 for i, c in enumerate(cats)}
        images, annos, aid = [], [], 1
        for i, r in enumerate([x for x in records if x["split"] == split], start=1):
            images.append({"id": i, "file_name": os.path.basename(r["file"]),
                           "width": r["width"], "height": r["height"]})
            for a in r["anns"]:
                x, y, bw, bh = a["bbox"]
                c = "active_object" if label_mode == "agnostic" else a["material"]
                annos.append({"id": aid, "image_id": i, "category_id": cat_id[c],
                              "bbox": [x, y, bw, bh], "area": bw * bh,
                              "iscrowd": 0})
                aid += 1
        js = {"images": images, "annotations": annos,
              "categories": [{"id": cat_id[c], "name": c} for c in cats]}
        p = out / "coco" / f"{split}_{label_mode}.json"
        with open(p, "w") as f:
            json.dump(js, f)
        return len(images), len(annos), cats

    print("\nCOCO")
    for lm in ("agnostic", "material"):
        for s in SPLITS:
            ni, na, cats = write_coco(s, lm)
            print(f"  {lm:<9s} {s:<5s} images {ni:>5d}  boxes {na:>5d}  "
                  f"classes {len(cats)}")

    # ---------- YOLO ----------
    for lm in ("agnostic", "material"):
        cats = (["active_object"] if lm == "agnostic" else materials)
        idx = {c: i for i, c in enumerate(cats)}
        root = out / f"yolo_{lm}"
        for s in SPLITS:
            E.ensure_dir(root / "images" / s)
            E.ensure_dir(root / "labels" / s)
        for r in records:
            s = r["split"]
            img_link = root / "images" / s / f"{r['stem']}.jpg"
            if not img_link.exists():
                try:
                    os.symlink(os.path.abspath(r["file"]), img_link)
                except OSError:
                    shutil.copyfile(r["file"], img_link)
            lines = []
            for a in r["anns"]:
                x, y, bw, bh = a["bbox"]
                cx, cy = (x + bw / 2) / r["width"], (y + bh / 2) / r["height"]
                nw, nh = bw / r["width"], bh / r["height"]
                c = "active_object" if lm == "agnostic" else a["material"]
                lines.append(f"{idx[c]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (root / "labels" / s / f"{r['stem']}.txt").write_text("\n".join(lines))
        yaml = (f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnames:\n" +
                "".join(f"  {i}: {c}\n" for i, c in enumerate(cats)))
        (root / "data.yaml").write_text(yaml)
        print(f"  yolo_{lm}/data.yaml written ({len(cats)} classes)")

    n_box = sum(len(r["anns"]) for r in records)
    print(f"\nframes {len(records)}   boxes {n_box}   videos {len(vsize)}")
    for s in SPLITS:
        sub = [r for r in records if r["split"] == s]
        print(f"  {s:<5s} {len(sub):>5d} frames  "
              f"{len({r['video'] for r in sub})} videos")
    print(f"material classes ({len(materials)}): {materials}")
    if skipped:
        print(f"skipped videos without a meta CSV: {len(skipped)}")

    E.save_json({"materials": materials, "n_frames": len(records),
                 "n_boxes": n_box, "skipped_videos": skipped,
                 "split_assignment": assign},
                str(out / "build_summary.json"))
    print(f"\ndataset -> {out}")


if __name__ == "__main__":
    main()
