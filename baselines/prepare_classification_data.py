#!/usr/bin/env python3
"""
prepare_classification_data.py
==============================
Builds a classification dataset for ACTIVE (handheld) OBJECT RECOGNITION from
the verified annotations.

The task: in a heavily cluttered egocentric frame containing dozens of waste
items, predict the material category / waste class of the object the expert is
actually holding. Unlike ordinary object-centric classification, the model must
key on the manipulated object, not the most salient one.

INPUT MODES (extract several, compare them later)
------------------------------------------------
  crop          tight box around the active object                  (easiest)
  crop_ctx      box expanded by --ctx, so surrounding clutter is visible
  masked        crop_ctx with the background darkened using the mask
  frame         the whole frame, no localisation given              (hardest)

'frame' is the interesting egocentric setting: the model has to find the
manipulated object itself. 'crop' is the conventional setting and gives the
upper bound.

LEAKAGE CONTROL
---------------
  * split is by VIDEO, never random -- frames of one sort are near-duplicates
  * at most --max_per_tracklet frames are kept per object, so a long
    manipulation cannot dominate the split
  * videos with no meta CSV are skipped, not silently zero-filled

USAGE
  python prepare_classification_data.py \
     --roots "/path/to/egowaste_annotations/Brookhaven Landfill 20250602 Preproc" \
             "/path/to/egowaste_annotations/Brookhaven MRF 20250604 Preproc" \
     --frames_roots "/path/to/egowaste_frames/Brookhaven Landfill 20250602 Preproc" \
                    "/path/to/egowaste_frames/Brookhaven MRF 20250604 Preproc" \
     --cameras 1 2 3 4 5 \
     --modes crop_ctx masked frame \
     --out_dir ./active_obj_data
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import annotation_utils as E

FRAMES_FOLDER_NAME = "frames_resized_1024"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
OUT_SIZE = 256          # stored a little larger than 224 so training can crop


# ============================================================
# LABELS
# ============================================================
# The meta CSVs usually record only the fine-grained waste class, with no
# material prefix. Splitting on "|" therefore yields a single part, and
# treating that part as the material silently produces ~50 "material" classes
# instead of the 15 in the taxonomy. Every downstream benchmark then trains on
# the wrong task. We resolve the material through the published taxonomy
# instead of trusting the string layout.
try:
    from taxonomy import build_lookup as _build_lookup, resolve as _resolve
    _LUT = _build_lookup()
except Exception:                       # taxonomy helper unavailable
    _LUT = None
    _resolve = None

_UNMAPPED = {}


def parse_labels(row):
    """
    Returns (material, waste_class), with material always one of the 15
    taxonomy categories.
    """
    mat = row.get("material_category", row.get("material"))
    cls = row.get("waste_class", row.get("class_name"))
    if not E.is_missing(mat) and not E.is_missing(cls):
        return str(mat).strip(), str(cls).strip()

    cat = row.get("category_name", row.get("category_str"))
    if E.is_missing(cat):
        return None, None
    s = str(cat).strip()

    if s.startswith("["):
        try:
            parts = [str(x).strip() for x in json.loads(s.replace("'", '"'))]
        except Exception:
            parts = [s]
    else:
        parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return None, None

    if len(parts) >= 2:
        return parts[0], parts[-1]

    # single token: it is the waste class, so look the material up
    wc = parts[0]
    if _LUT is not None:
        m, c, _how = _resolve(wc, _LUT)
        if m is not None:
            return m, (c or wc)
    _UNMAPPED[wc] = _UNMAPPED.get(wc, 0) + 1
    return None, None


def report_unmapped():
    """Print any waste classes the taxonomy could not resolve."""
    if not _UNMAPPED:
        return
    print(f"\n[warn] {len(_UNMAPPED)} waste class(es) could not be mapped to a "
          f"material and were skipped:")
    for k, v in sorted(_UNMAPPED.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {k[:48]:<50s} {v:>6d}")
    print("    Add them to ALIASES in taxonomy.py.")


# ============================================================
# CROPPING
# ============================================================
def square_box(x, y, w, h, ctx, W, H):
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * ctx)
    side = max(side, 16.0)
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


def make_view(img, mask, bbox, mode, ctx, bg_alpha):
    H, W = img.shape[:2]

    if mode == "frame":
        return cv2.resize(img, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

    if bbox is None:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w < 4 or h < 4:
        return None

    use_ctx = 0.0 if mode == "crop" else ctx
    x1, y1, x2, y2 = square_box(x, y, w, h, use_ctx, W, H)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    patch = img[y1:y2, x1:x2].copy()

    if mode == "masked":
        if mask is None:
            return None
        m = mask[y1:y2, x1:x2]
        if m.shape[:2] != patch.shape[:2] or m.sum() == 0:
            return None
        out = patch.astype(np.float32)
        bg = out * bg_alpha
        sel = (m > 0)[..., None]
        patch = np.where(sel, out, bg).astype(np.uint8)

    return cv2.resize(patch, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)


# ============================================================
# SOURCES
# ============================================================
def find_frame_image(frames_roots, cam_id, gx_id, frame_name, csv_paths=None):
    for p in (csv_paths or []):
        if not E.is_missing(p) and os.path.isfile(str(p)):
            return str(p)
    for root in frames_roots:
        cam = E.get_camera_dir(root, cam_id)
        if cam is None:
            continue
        d = cam / gx_id / FRAMES_FOLDER_NAME
        for ext in IMAGE_EXTS:
            q = d / f"{frame_name}{ext}"
            if q.is_file():
                return str(q)
    return None


def load_mask(row):
    for col in ("corrected_mask_path", "mask_path"):
        v = row.get(col)
        if not E.is_missing(v) and os.path.isfile(str(v)):
            m = cv2.imread(str(v), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return (m > 127).astype(np.uint8)
    return None


def get_bbox(row, mask):
    for col in ("corrected_bbox_xywh", "bbox_xywh", "object_bbox"):
        b = E.parse_bbox(row.get(col))
        if b is not None and b[2] > 0 and b[3] > 0:
            return b
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return [float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min())]
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--frames_roots", nargs="*", default=[])
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gt_globs", nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=["crop_ctx", "masked", "frame"],
                    choices=["crop", "crop_ctx", "masked", "frame"])
    ap.add_argument("--ctx", type=float, default=0.4,
                    help="context padding as a fraction of the object's larger side")
    ap.add_argument("--bg_alpha", type=float, default=0.35,
                    help="how much background survives in 'masked' mode")
    ap.add_argument("--max_per_tracklet", type=int, default=4,
                    help="cap frames kept per object, so one long sort cannot "
                         "dominate; set 0 for no cap")
    ap.add_argument("--out_dir", default="./active_obj_data")
    args = ap.parse_args()

    frames_roots = args.frames_roots + args.roots
    E.ensure_dir(args.out_dir)
    for m in args.modes:
        E.ensure_dir(os.path.join(args.out_dir, m))

    print("=" * 78)
    print("Building the active-object classification dataset")
    print("=" * 78)

    records, skipped_videos = [], []
    for root in args.roots:
        day = Path(root).name
        for cam in args.cameras:
            cam_dir = E.get_camera_dir(root, cam)
            if cam_dir is None:
                continue
            for gx_dir in sorted(cam_dir.glob("GX*")):
                gt_csv, _ = E.find_gt_csv(gx_dir, args.gt_globs)
                if gt_csv is None:
                    skipped_videos.append(f"{day}|C{cam}_{gx_dir.name}")
                    continue

                df = pd.read_csv(gt_csv, low_memory=False)
                if "integrated_review_status" in df.columns:
                    df = df[df["integrated_review_status"] != "deleted_false_object"]

                video = f"{day}|C{cam}_{gx_dir.name}"
                per_track = {}
                cache_img, cache_path = None, None

                for _, r in tqdm(df.iterrows(), total=len(df),
                                 desc=video[-24:], leave=False):
                    fname = E.normalize_frame_name(r.get("frame_name"))
                    if fname is None:
                        continue
                    mat, cls = parse_labels(r)
                    if mat is None or cls is None:
                        continue

                    oid = E.safe_int(r.get("object_track_id"))
                    key = (oid, cls)
                    if args.max_per_tracklet:
                        if per_track.get(key, 0) >= args.max_per_tracklet:
                            continue

                    img_path = find_frame_image(
                        frames_roots, cam, gx_dir.name, fname,
                        [r.get("image_path"), r.get("source_image_path")])
                    if img_path is None:
                        continue
                    if img_path != cache_path:
                        cache_img = cv2.imread(img_path)
                        cache_path = img_path
                    if cache_img is None:
                        continue

                    mask = load_mask(r)
                    bbox = get_bbox(r, mask)

                    stem = f"{video.replace('|','_').replace(' ','')}__{fname}__{oid}"
                    wrote = {}
                    for mode in args.modes:
                        view = make_view(cache_img, mask, bbox, mode,
                                         args.ctx, args.bg_alpha)
                        if view is None:
                            continue
                        out_p = os.path.join(args.out_dir, mode, stem + ".jpg")
                        cv2.imwrite(out_p, view, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        wrote[mode] = os.path.relpath(out_p, args.out_dir)

                    if not wrote:
                        continue
                    per_track[key] = per_track.get(key, 0) + 1
                    rec = {"video": video, "day": day, "camera_id": cam,
                           "gx_id": gx_dir.name, "frame_name": fname,
                           "object_track_id": oid, "material": mat,
                           "waste_class": cls, "stem": stem}
                    rec.update({f"path_{k}": v for k, v in wrote.items()})
                    records.append(rec)

    if not records:
        raise RuntimeError("no samples built -- check --gt_globs and --frames_roots")

    idx = pd.DataFrame(records)
    idx.to_csv(os.path.join(args.out_dir, "index.csv"), index=False)

    print(f"\nsamples          : {len(idx)}")
    print(f"videos           : {idx['video'].nunique()}")
    print(f"objects          : {idx['object_track_id'].nunique()}")
    print(f"material classes : {idx['material'].nunique()}")
    print(f"waste classes    : {idx['waste_class'].nunique()}")
    if skipped_videos:
        print(f"skipped (no meta CSV): {len(skipped_videos)}")

    print("\ntop material categories:")
    print(idx["material"].value_counts().head(10).to_string())
    tail = idx["waste_class"].value_counts()
    print(f"\nwaste-class counts: max {tail.max()}, median {int(tail.median())}, "
          f"min {tail.min()}; {int((tail < 10).sum())} classes have <10 samples")
    print("  -> long-tailed, so report macro-F1 and balanced accuracy, not just top-1")

    E.save_json({"n_samples": int(len(idx)), "modes": args.modes,
                 "ctx": args.ctx, "bg_alpha": args.bg_alpha,
                 "max_per_tracklet": args.max_per_tracklet,
                 "videos": sorted(idx["video"].unique().tolist()),
                 "skipped_videos": skipped_videos},
                os.path.join(args.out_dir, "build_summary.json"))
    print(f"\nindex -> {os.path.join(args.out_dir, 'index.csv')}")


if __name__ == "__main__":
    main()
