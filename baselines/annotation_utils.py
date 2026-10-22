#!/usr/bin/env python3
"""
annotation_utils.py
================
Shared utilities for W2: evaluating the hand-object edge classifier.

THE TASK
--------
For each (frame, hand) the interaction graph proposes several candidate
objects. Exactly one of them is the object actually being held. The question
is: does the method pick it?

GROUND TRUTH
------------
Rows in <GX>/data/integrated_corrections/*.csv that
  * came from the edge classifier  (source_type is NaN, i.e. NOT
    'recovered_from_meta_anchor' -- those are temporal extensions of an object
    the classifier had already found in a neighbouring frame, so they were
    never a decision the classifier made), and
  * were not marked 'deleted_false_object'.

CANDIDATES
----------
<GX>/handheld_objects_iter_0_glove_filtered/handheld_candidate_edges.csv,
the full candidate set the selector chose from.

LEAKAGE
-------
The MLP re-ranker was trained on these same approved positives, so evaluating
it on all videos at once is testing on training data. The rule score and every
geometric heuristic are parameter-free and leakage-free; the MLP must be run
leave-one-video-out.
"""

import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
DATA_FOLDER = "data"
INTEGRATED_FOLDER = "integrated_corrections"
ITER0_FOLDER = "handheld_objects_iter_0_glove_filtered"
CANDIDATE_EDGES_CSV = "handheld_candidate_edges.csv"
CANDIDATE_EDGES_JSON = "handheld_candidate_edges.json"

EDGE_FEATURE_NAMES = [
    "rule_edge_score", "contact_score", "boundary_touch_score",
    "motion_match_score", "object_glove_prob",
    "bbox_area_norm", "bbox_w_norm", "bbox_h_norm",
]

IMG_W, IMG_H = 1024.0, 576.0
MIN_FINAL_SCORE = 0.25          # matches the deployed selector


# ============================================================
# SMALL HELPERS
# ============================================================
def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))

    def safe(o):
        if isinstance(o, dict):
            return {str(k): safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [safe(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            return None if np.isnan(v) else v
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, float) and np.isnan(o):
            return None
        return o

    with open(path, "w") as f:
        json.dump(safe(obj), f, indent=2)


def get_camera_dir(root, cam_id):
    for name in (f"Camera {cam_id}", f"Camera{cam_id}"):
        p = Path(root) / name
        if p.is_dir():
            return p
    return None


def normalize_frame_name(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    s = Path(s).stem
    for suf in ("_overlay", "_compare", "_hand_only"):
        s = s.replace(suf, "")
    return f"{int(s):06d}" if s.isdigit() else s


def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "null"):
        return True
    return False


def safe_float(v, default=0.0) -> float:
    if is_missing(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def safe_int(v) -> Optional[int]:
    if is_missing(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_bbox(v) -> Optional[List[float]]:
    if is_missing(v):
        return None
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v[:4]] if len(v) >= 4 else None
    if isinstance(v, str):
        s = v.strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                p = parser(s)
                if isinstance(p, (list, tuple)) and len(p) >= 4:
                    return [float(x) for x in p[:4]]
            except Exception:
                continue
    return None


def bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return float(inter / max(aw * ah + bw * bh - inter, 1e-6))


def edge_feature_vector(row) -> np.ndarray:
    bbox = parse_bbox(row.get("object_bbox", row.get("bbox_xywh")))
    if bbox is None:
        area = wn = hn = 0.0
    else:
        _, _, w, h = bbox
        area, wn, hn = (w * h) / (IMG_W * IMG_H), w / IMG_W, h / IMG_H
    return np.array([
        safe_float(row.get("rule_edge_score", row.get("edge_score", 0.0))),
        safe_float(row.get("contact_score")),
        safe_float(row.get("boundary_touch_score")),
        safe_float(row.get("motion_match_score")),
        safe_float(row.get("object_glove_prob")),
        area, wn, hn,
    ], dtype=np.float32)


# ============================================================
# JOB DISCOVERY
# ============================================================
DEFAULT_GT_PATTERNS = [
    "data/integrated_corrections/*.csv",
    "data/meta2*.csv",
    "data/*integrated*correct*.csv",
    "data/meta*.csv",
]

# a file is only useful for correction statistics if it records what changed
CORRECTION_STATUS_COLUMNS = ["integrated_review_status", "review_status"]


def csv_has_correction_column(path) -> bool:
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    return any(c in head.columns for c in CORRECTION_STATUS_COLUMNS)


def find_gt_csv(gx_dir: Path, patterns=None, require_corrections=False,
                require_integrated=False):
    """
    Locate the annotation CSV for one video.

    Patterns are tried in order; within a pattern the most recently modified
    file wins. Files that record a review status are preferred, since only
    those support correction statistics.

    require_integrated=True restricts the search to
    data/integrated_corrections/, so videos that have not been through the
    integrated correction pass are skipped rather than falling back to an
    earlier, unverified meta CSV.

    Returns (path, has_correction_column) or (None, False).
    """
    if require_integrated:
        patterns = [f"{DATA_FOLDER}/{INTEGRATED_FOLDER}/*.csv"]
    patterns = patterns or DEFAULT_GT_PATTERNS
    fallback = None
    for pat in patterns:
        hits = sorted(gx_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        for h in hits:
            if csv_has_correction_column(h):
                return str(h), True
            if fallback is None:
                fallback = str(h)
    if require_corrections:
        return None, False
    return fallback, False


# ------------------------------------------------------------
# MASK RESOLUTION
# ------------------------------------------------------------
# Masks are looked up by FILENAME in a priority list of directories, so the
# verified copy inside integrated_corrections wins even when the CSV still
# records an older path.
# The reviewer tool writes corrected masks to
#   <GX>/data/integrated_corrections/masks/<gx_id>/<frame:06d>_objG<oid:05d>.png
# so the index must recurse; a flat listing finds only the gx_id subfolders.
DEFAULT_MASK_DIRS = [
    f"{DATA_FOLDER}/{INTEGRATED_FOLDER}/masks",
    f"{DATA_FOLDER}/{INTEGRATED_FOLDER}/*mask*",
    f"{DATA_FOLDER}/*backup*/masks",
    f"{DATA_FOLDER}/*backup*",
    f"{DATA_FOLDER}/masks",
]

# Priority used by integrated_reviewer.py and export_clean_gx.py: a manually
# redrawn mask wins, then the pipeline mask, then the raw source mask.
MASK_COL_PRIORITY = ("corrected_mask_path", "mask_path", "source_mask_path",
                     "final_mask_path")

# a row is excluded if ANY review column marks it deleted or false
DELETED_STATUS_COLUMNS = ("integrated_review_status", "manual_item_review_status",
                          "event_category_review_status", "category_review_status",
                          "review_status")
DELETED_TOKENS = ("deleted", "false_object", "rejected")


def row_is_deleted(row) -> bool:
    for c in DELETED_STATUS_COLUMNS:
        v = row.get(c)
        if is_missing(v):
            continue
        s = str(v).lower()
        if any(t in s for t in DELETED_TOKENS):
            return True
    return False


def drop_deleted(df):
    """Remove rows any reviewer marked deleted, across every status column."""
    keep = ~df.apply(row_is_deleted, axis=1)
    return df.loc[keep].copy()


def build_mask_index(gx_dir: Path, patterns=None):
    """
    filename -> (path, which_directory), searched RECURSIVELY. Earlier patterns
    win, so a manually corrected mask takes precedence over a backup copy.
    """
    patterns = patterns or DEFAULT_MASK_DIRS
    index, sources = {}, {}
    for pat in patterns:
        for d in sorted(gx_dir.glob(pat)):
            if not d.is_dir():
                continue
            rel = str(d.relative_to(gx_dir))
            for f in d.rglob("*"):
                if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                if f.name not in index:            # first pattern wins
                    index[f.name] = str(f)
                    sources[f.name] = rel
    return index, sources


def expected_mask_name(row):
    """The reviewer tool's naming scheme: <frame:06d>_objG<track:05d>.png"""
    f = safe_int(row.get("frame_name"))
    o = safe_int(row.get("object_track_id"))
    if f is None or o is None:
        return None
    return f"{f:06d}_objG{o:05d}.png"


def resolve_mask(row, mask_index, mask_sources, cols=MASK_COL_PRIORITY):
    """
    Resolution order:
      1. the reviewer tool's canonical filename, if present in the index
      2. the basename of each CSV path, in priority order, looked up in the index
      3. the literal CSV path, if it still exists on disk
    Looking up by name first matters because the CSV records absolute paths
    that go stale when the dataset is moved between drives.
    Returns (path, source_label) or (None, None).
    """
    canon = expected_mask_name(row)
    if canon and canon in mask_index:
        return mask_index[canon], mask_sources[canon]

    for c in cols:
        v = row.get(c)
        if is_missing(v):
            continue
        n = os.path.basename(str(v))
        if n in mask_index:
            return mask_index[n], mask_sources[n]

    for c in cols:
        v = row.get(c)
        if not is_missing(v) and os.path.isfile(str(v)):
            return str(v), f"csv:{c}"
    return None, None


def find_integrated_csv(gx_dir: Path):
    """Backwards-compatible wrapper used by the W2 scripts."""
    path, _ = find_gt_csv(gx_dir)
    return path


def find_jobs(roots: List[str], camera_ids: List[int]) -> List[Dict]:
    jobs = []
    for root in roots:
        day = Path(root).name
        for cam_id in camera_ids:
            cam_dir = get_camera_dir(root, cam_id)
            if cam_dir is None:
                continue
            for gx_dir in sorted(cam_dir.glob("GX*")):
                gt_csv = find_integrated_csv(gx_dir)
                if gt_csv is None:
                    continue
                edges_csv = gx_dir / ITER0_FOLDER / CANDIDATE_EDGES_CSV
                edges_json = gx_dir / ITER0_FOLDER / CANDIDATE_EDGES_JSON
                if not edges_csv.is_file() and not edges_json.is_file():
                    print(f"[warn] no candidate edges for {cam_dir.name}/{gx_dir.name} "
                          f"-- re-run inference with --save_all_edges")
                    continue
                jobs.append({
                    "day": day,
                    "camera_id": cam_id,
                    "gx_id": gx_dir.name,
                    "group": f"{day}|C{cam_id}_{gx_dir.name}",
                    "gt_csv": gt_csv,
                    "edges_csv": str(edges_csv) if edges_csv.is_file() else None,
                    "edges_json": str(edges_json) if edges_json.is_file() else None,
                })
    return jobs


# ============================================================
# GROUND TRUTH
# ============================================================
def load_ground_truth(job: Dict) -> Dict:
    """
    Returns
      gt      : (frame, hand_side) -> approved object_track_id
      stats   : counts, including the documented wrong-object deletions
    """
    df = pd.read_csv(job["gt_csv"], low_memory=False)

    n_total = len(df)
    status_col = ("integrated_review_status" if "integrated_review_status" in df.columns
                  else None)
    n_deleted = int((df[status_col] == "deleted_false_object").sum()) if status_col else 0

    keep = df
    if status_col:
        keep = keep[keep[status_col] != "deleted_false_object"]

    # exclude temporal extensions -- never an edge-classifier decision
    if "source_type" in keep.columns:
        n_recovered = int((keep["source_type"] == "recovered_from_meta_anchor").sum())
        keep = keep[keep["source_type"].isna()]
    else:
        n_recovered = 0

    gt = {}
    dup = 0
    for _, r in keep.iterrows():
        f = normalize_frame_name(r.get("frame_name"))
        side = r.get("hand_side")
        oid = safe_int(r.get("object_track_id"))
        if f is None or oid is None or is_missing(side):
            continue
        k = (f, str(side))
        if k in gt and gt[k] != oid:
            dup += 1
        gt[k] = oid

    stats = {
        "group": job["group"],
        "gt_rows_total": n_total,
        "gt_deleted_false_object": n_deleted,
        "gt_recovered_excluded": n_recovered,
        "gt_usable": len(gt),
        "gt_conflicting_keys": dup,
    }
    return gt, stats


# ============================================================
# CANDIDATES
# ============================================================
def load_candidates(job: Dict) -> Dict:
    """(frame, hand_side) -> list of candidate edge dicts."""
    rows = []
    if job.get("edges_csv"):
        try:
            df = pd.read_csv(job["edges_csv"], low_memory=False)
            rows = df.to_dict("records")
        except Exception:
            rows = []
    if not rows and job.get("edges_json"):
        try:
            with open(job["edges_json"]) as f:
                loaded = json.load(f)
            rows = loaded if isinstance(loaded, list) else []
        except Exception:
            rows = []

    out = {}
    for r in rows:
        f = normalize_frame_name(r.get("frame_name"))
        side = r.get("hand_side")
        oid = safe_int(r.get("object_track_id"))
        if f is None or oid is None or is_missing(side):
            continue
        r = dict(r)
        r["frame_name"] = f
        r["object_track_id"] = oid
        r["_group"] = job["group"]
        out.setdefault((f, str(side)), []).append(r)
    return out


def build_examples(job: Dict):
    """
    One example per (frame, hand) that has both ground truth and candidates.
    """
    gt, stats = load_ground_truth(job)
    cand = load_candidates(job)

    examples, no_cand, not_in_cand = [], 0, 0
    for k, true_oid in gt.items():
        cs = cand.get(k)
        if not cs:
            no_cand += 1
            continue
        oids = [c["object_track_id"] for c in cs]
        in_set = true_oid in oids
        if not in_set:
            not_in_cand += 1
        examples.append({
            "group": job["group"], "day": job["day"],
            "camera_id": job["camera_id"], "gx_id": job["gx_id"],
            "frame_name": k[0], "hand_side": k[1],
            "true_object_id": true_oid,
            "candidates": cs,
            "n_candidates": len(cs),
            "true_in_candidates": bool(in_set),
            "segment_id": cs[0].get("segment_id"),
        })

    stats.update({
        "examples": len(examples),
        "gt_without_candidates": no_cand,
        "true_object_missing_from_candidates": not_in_cand,
        "candidate_set_ceiling": (
            float(np.mean([e["true_in_candidates"] for e in examples]))
            if examples else float("nan")),
        "mean_candidates_per_example": (
            float(np.mean([e["n_candidates"] for e in examples]))
            if examples else float("nan")),
    })
    return examples, stats


# ============================================================
# SELECTORS
# ============================================================
def _pick(example, score_fn, min_score=None, apply_glove_filter=True):
    best, best_s = None, -np.inf
    for c in example["candidates"]:
        if apply_glove_filter and bool(c.get("is_glove_like_object", False)):
            continue
        s = score_fn(c)
        if s > best_s:
            best, best_s = c, s
    if best is None:
        return None, float("nan")
    if min_score is not None and best_s < min_score:
        return None, float(best_s)
    return best["object_track_id"], float(best_s)


def selector_rule(example):
    """Iteration 0: the deployed rule score, with its 0.25 abstain threshold."""
    return _pick(example, lambda c: safe_float(c.get("rule_edge_score")),
                 min_score=MIN_FINAL_SCORE)


def selector_contact(example):
    return _pick(example, lambda c: safe_float(c.get("contact_score")))


def selector_boundary(example):
    return _pick(example, lambda c: safe_float(c.get("boundary_touch_score")))


def selector_motion(example):
    return _pick(example, lambda c: safe_float(c.get("motion_match_score")))


def selector_largest(example):
    def area(c):
        b = parse_bbox(c.get("object_bbox", c.get("bbox_xywh")))
        return (b[2] * b[3]) if b else 0.0
    return _pick(example, area)


def selector_nearest_centroid(example):
    """Closest object to the centre of the expanded hand box."""
    def negdist(c):
        hb = parse_bbox(c.get("expanded_hand_bbox"))
        ob = parse_bbox(c.get("object_bbox", c.get("bbox_xywh")))
        if hb is None or ob is None:
            return -1e9
        hx, hy = hb[0] + hb[2] / 2, hb[1] + hb[3] / 2
        ox, oy = ob[0] + ob[2] / 2, ob[1] + ob[3] / 2
        return -float(np.hypot(hx - ox, hy - oy))
    return _pick(example, negdist)


def selector_mlp(model):
    def fn(example):
        def score(c):
            x = edge_feature_vector(c).reshape(1, -1)
            return float(model.predict_proba(x)[0, 1])
        return _pick(example, score, min_score=MIN_FINAL_SCORE)
    return fn


SELECTORS = {
    "max contact": selector_contact,
    "max boundary touch": selector_boundary,
    "max motion match": selector_motion,
    "largest object": selector_largest,
    "nearest to hand": selector_nearest_centroid,
    "rule score (iter 0)": selector_rule,
}


# ============================================================
# SCORING
# ============================================================
def per_video_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Mean +- std of the three outcome rates across held-out videos.

    The pooled rate weights each video by its number of decisions; the
    per-video mean weights each video equally and, unlike the pooled figure,
    exposes the spread. A leave-one-video-out table should report the latter.
    """
    if len(df) == 0 or "group" not in df.columns:
        return {}
    per = df.groupby("group").apply(
        lambda s: pd.Series({
            "acc": s["correct"].mean(),
            "wrong": ((~s["correct"]) & (~s["abstained"])).mean(),
            "nopred": s["abstained"].mean(),
        }),
        include_groups=False,
    )
    return {
        "accuracy_mean": float(per["acc"].mean()),
        "accuracy_std": float(per["acc"].std()),
        "wrong_object_mean": float(per["wrong"].mean()),
        "wrong_object_std": float(per["wrong"].std()),
        "no_prediction_mean": float(per["nopred"].mean()),
        "no_prediction_std": float(per["nopred"].std()),
        "n_videos": int(len(per)),
    }


def evaluate_selector(examples, selector, name):
    n = len(examples)
    correct = abstain = wrong = 0
    rows = []
    for e in examples:
        pred, score = selector(e)
        if pred is None:
            abstain += 1
            ok = False
        else:
            ok = (pred == e["true_object_id"])
            correct += int(ok)
            wrong += int(not ok)
        rows.append({
            "group": e["group"], "day": e["day"], "camera_id": e["camera_id"],
            "frame_name": e["frame_name"], "hand_side": e["hand_side"],
            "segment_id": e["segment_id"],
            "true_object_id": e["true_object_id"], "pred_object_id": pred,
            "score": score, "correct": bool(ok), "abstained": pred is None,
            "n_candidates": e["n_candidates"],
            "true_in_candidates": e["true_in_candidates"],
            "method": name,
        })
    df = pd.DataFrame(rows)

    # interval-level: does the method get the object right at least once
    # in each spoken interval? That is what the pipeline propagates.
    iv = float("nan")
    if len(df) and df["segment_id"].notna().any():
        sub = df[df["segment_id"].notna()]
        g = sub.groupby(["group", "segment_id"])["correct"].max()
        iv = float(g.mean())

    decided = correct + wrong
    out = {
        "method": name,
        "accuracy": correct / max(n, 1),
        "accuracy_when_decided": correct / max(decided, 1),
        "abstain_rate": abstain / max(n, 1),
        "wrong_object_rate": wrong / max(n, 1),
        "interval_accuracy": iv,
        "n_examples": n,
    }
    out.update(per_video_stats(df))
    return out, df
