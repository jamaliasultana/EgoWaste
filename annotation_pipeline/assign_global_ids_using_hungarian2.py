import os
import json
import math
import gc
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ============================================================
# ROOT PATHS
# ============================================================

PREPROC_ROOT = "/media/jisha/D/trash_dataset/Brookhaven MRF 20250604 Preproc"
# [NEW] where SAM masks, DINO features, and Hungarian outputs are stored
OUTPUT_PREPROC_ROOT = "/media/jisha/E/trash_dataset/Brookhaven MRF 20250604 Preproc"
TRANSCRIPT_ROOT = "/media/jisha/D/trash_dataset/output/Brookhaven MRF 20250604"

CAMERA_IDS = [1, 2, 3, 4, 5]

FRAMES_FOLDER_NAME = "frames_resized_1024"
FEATURE_FOLDER_NAME = "dino_bbox_crop_feature_fg_ctx_cache"
META_FOLDER_NAME = "proposal_meta"

# [MODIFIED] new output folder name
OUTPUT_FOLDER_NAME = "offline_hungarian_tracker_segment_adaptive"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# SETTINGS
# ============================================================

MAX_FRAMES = None

FRAME_FPS = 8.0
FRAME_NUMBER_START = 1
TRANSCRIPT_TIME_PADDING_SEC = 0.5

USE_INTRA_FRAME_DUP_SUPPRESSION = True
INTRA_BBOX_IOU_THRESH = 0.80
INTRA_CONCAT_SIM_THRESH = 0.97

# [NEW] containment-based suppression
USE_CONTAINMENT_SUPPRESSION = False #True
CONTAINMENT_THRESH = 0.90
SMALL_TO_LARGE_AREA_RATIO = 0.65

# [MODIFIED] these are now fallback values only
DEFAULT_SHORT_TERM_MEMORY = 16
DEFAULT_LONG_TERM_MEMORY = 48
DEFAULT_INACTIVE_PATIENCE = 6

MIN_APPEARANCE_SIM = 0.38
MIN_FG_SIM = 0.32
MIN_TEMPORAL_SCORE = 0.02

# [MODIFIED] active and inactive thresholds separated
MIN_TOTAL_ACTIVE = 0.48
MIN_TOTAL_INACTIVE = 0.58
MIN_FG_INACTIVE = 0.42

W_APPEARANCE = 0.38
W_TEMPORAL = 0.25
W_MOTION = 0.22
W_PERSISTENCE = 0.05
W_SAM = 0.10

W_FG = 0.65
W_CONCAT = 0.35

SAVE_VISUALIZATIONS = True
MAX_VIS_FRAMES = 10
MASK_ALPHA = 0.45
DRAW_TEXT = True

# ============================================================
# [NEW] WITHIN-SEGMENT STITCHING SETTINGS
# ============================================================

ENABLE_WITHIN_SEGMENT_STITCHING = True

STITCH_MAX_GAP_RATIO = 0.5
STITCH_MAX_GAP_MIN = 1
STITCH_MAX_GAP_MAX = 6

MIN_STITCH_FG_SIM = 0.42
MIN_STITCH_CONCAT_SIM = 0.45
MIN_STITCH_TOTAL_SCORE = 0.55

W_STITCH_FG = 0.40
W_STITCH_CONCAT = 0.25
W_STITCH_CENTER = 0.15
W_STITCH_AREA = 0.10
W_STITCH_GAP = 0.10

# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def category_to_string(category_name: Any) -> str:
    if isinstance(category_name, list):
        return " | ".join(str(x) for x in category_name)
    return str(category_name)


def safe_name(text: Any, max_len: int = 80) -> str:
    s = str(text).strip()
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    s = "".join(c for c in s if c.isalnum() or c in "_-")
    return s[:max_len] if s else "unknown"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def bbox_iou(box_a, box_b) -> float:
    if box_a is None or box_b is None:
        return 0.0

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1e-6))


def bbox_area(box) -> float:
    if box is None:
        return 0.0
    return float(max(box[2], 0) * max(box[3], 0))


def normalize_center_distance(c1, c2, image_shape, max_frac=0.25) -> float:
    h, w = image_shape[:2]
    diag = math.sqrt(h * h + w * w)
    d = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
    frac = d / max(diag, 1e-6)
    return float(max(1.0 - min(frac / max_frac, 1.0), 0.0))


def area_ratio_score(box_a, box_b, tol=4.0) -> float:
    area_a = max(bbox_area(box_a), 1.0)
    area_b = max(bbox_area(box_b), 1.0)
    r = max(area_a, area_b) / max(min(area_a, area_b), 1.0)

    if r >= tol:
        return 0.0

    return float(max(1.0 - ((r - 1.0) / max(tol - 1.0, 1e-6)), 0.0))


def safe_mean_feature(feats: List[np.ndarray]) -> Optional[np.ndarray]:
    if len(feats) == 0:
        return None
    arr = np.stack(feats, axis=0)
    return l2_normalize(arr.mean(axis=0))


def load_mask_from_item(item: Dict) -> np.ndarray:
    mask_archive_path = item["mask_archive_path"]
    mask_index = int(item["mask_index"])

    data = np.load(mask_archive_path)
    masks = data["masks"]

    return (masks[mask_index] > 0).astype(np.uint8)

def load_feature_from_item(item: Dict, feature_type: str) -> np.ndarray:
    archive_path = item["feature_archive_path"]
    feature_index = int(item["feature_index"])

    if feature_type == "fg":
        key = item.get("fg_feature_key", "fg_features")
    elif feature_type == "concat":
        key = item.get("concat_feature_key", "concat_features")
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    data = np.load(archive_path)
    return data[key][feature_index]


def get_color(idx: int):
    rng = np.random.default_rng(seed=idx + 2026)
    c = rng.integers(40, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def overlay_mask(image, mask, color, alpha):
    out = image.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    pix = out[mask > 0].astype(np.float32)
    out[mask > 0] = ((1 - alpha) * pix + alpha * color_arr).astype(np.uint8)
    return out


def draw_text(img, text, xy, color=(255, 255, 255), scale=0.45):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ============================================================
# [NEW] ADAPTIVE TEMPORAL PARAMETERS
# ============================================================

def get_adaptive_temporal_params(num_frames: int) -> Dict[str, int]:
    n = max(int(num_frames), 1)

    return {
        "short_term_memory": max(3, min(n, 16)),
        "long_term_memory": max(5, min(3 * n, 48)),
        "inactive_patience": max(1, min(n // 3, 6)),
    }


# ============================================================
# FIND JOBS
# ============================================================

def get_camera_dir(root: str, cam_id: int) -> Optional[Path]:
    candidates = [
        Path(root) / f"Camera{cam_id}",
        Path(root) / f"Camera {cam_id}",
    ]

    for c in candidates:
        if c.is_dir():
            return c

    return None


def find_jobs() -> List[Dict]:
    jobs = []

    for cam_id in CAMERA_IDS:
        preproc_cam_dir = get_camera_dir(PREPROC_ROOT, cam_id)
        transcript_cam_dir = get_camera_dir(TRANSCRIPT_ROOT, cam_id)

        if preproc_cam_dir is None:
            print(f"Missing preproc camera folder for Camera{cam_id}")
            continue

        if transcript_cam_dir is None:
            print(f"Missing transcript camera folder for Camera{cam_id}")
            continue

        for gx_dir in sorted(preproc_cam_dir.glob("GX*")):
            gx_id = gx_dir.name

            frames_dir = gx_dir / FRAMES_FOLDER_NAME
            transcript_json = transcript_cam_dir / gx_id / "full_transcription.json"
            selected_summary_json = frames_dir / "selected_frame_summary.json"
            # [CHANGED] frames stay on D drive, features/output stay on E drive
            output_cam_dir = get_camera_dir(OUTPUT_PREPROC_ROOT, cam_id)
            if output_cam_dir is None:
                output_cam_dir = Path(OUTPUT_PREPROC_ROOT) / f"Camera {cam_id}"

            output_gx_dir = output_cam_dir / gx_id

            feature_root = output_gx_dir / FEATURE_FOLDER_NAME
            meta_dir = feature_root / META_FOLDER_NAME

            # [CHANGED] save Hungarian tracker output on E drive
            output_root = output_gx_dir / OUTPUT_FOLDER_NAME

            if not frames_dir.is_dir():
                continue

            if not meta_dir.is_dir():
                print(f"Skipping {gx_id}: missing proposal_meta")
                continue

            if not transcript_json.is_file():
                print(f"Skipping {gx_id}: missing transcript JSON")
                continue

            jobs.append({
                "camera_id": cam_id,
                "gx_id": gx_id,
                "frames_dir": str(frames_dir),
                "feature_root": str(feature_root),
                "meta_dir": str(meta_dir),
                "transcript_json": str(transcript_json),
                "selected_summary_json": str(selected_summary_json) if selected_summary_json.is_file() else None,
                "output_root": str(output_root),
                "segment_json_path": str(output_root / "segment_tracks.json"),
                "tracklet_json_path": str(output_root / "tracklets_before_stitching.json"),
                "track_json_path": str(output_root / "global_tracks.json"),
                "segment_info_json_path": str(output_root / "segment_info.json"),
                "overlay_vis_dir": str(output_root / "overlays"),
                "compare_vis_dir": str(output_root / "compares"),
            })

    return jobs


# ============================================================
# FRAME INDEX
# ============================================================

def build_frame_index(frames_dir: str, meta_dir: str) -> List[Dict]:
    frame_paths = sorted([
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if MAX_FRAMES is not None:
        frame_paths = frame_paths[:MAX_FRAMES]

    out = []

    for frame_idx, fp in enumerate(frame_paths):
        frame_name = fp.stem
        meta_path = os.path.join(meta_dir, f"{frame_name}.json")

        if os.path.isfile(meta_path):
            out.append({
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "frame_num": int(frame_name),
                "image_path": str(fp),
                "meta_path": meta_path,
            })

    return out


def frame_lookup(frame_index: List[Dict]) -> Dict[str, Dict]:
    return {fd["frame_name"]: fd for fd in frame_index}


# ============================================================
# [NEW] SEGMENT LOADING
# ============================================================

def has_valid_category(category_name) -> bool:
    if category_name is None:
        return False
    if isinstance(category_name, list):
        return len(category_name) > 0
    if isinstance(category_name, str):
        return category_name.strip() != ""
    return False


def extract_transcript_items(data) -> List[Dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "time_stamped_text" in data:
            return data["time_stamped_text"]
        if "segments" in data:
            return data["segments"]
        if "words" in data:
            return data["words"]
    raise ValueError("Transcript JSON must contain list, time_stamped_text, segments, or words.")


def get_begin_end_from_item(item: Dict) -> Tuple[float, float]:
    begin_keys = ["begin_time", "start_time", "start", "begin"]
    end_keys = ["end_time", "stop_time", "end", "stop"]

    begin = None
    end = None

    for k in begin_keys:
        if k in item:
            begin = float(item[k])
            break

    for k in end_keys:
        if k in item:
            end = float(item[k])
            break

    if begin is None or end is None:
        raise ValueError(f"Missing begin/end time in transcript item: {item}")

    return begin, end


# [NEW] preferred: use selected_frame_summary.json because it directly lists saved frames
def load_segments_from_selected_summary(summary_json_path: str, frame_index: List[Dict]) -> List[Dict]:
    frame_map = frame_lookup(frame_index)
    summary = load_json(summary_json_path)

    segments = []

    for seg_id, row in enumerate(summary):
        saved_frames = row.get("saved_frames", [])
        if len(saved_frames) == 0:
            continue

        category_name = row.get("category_name", None)
        if not has_valid_category(category_name):
            continue

        selected_fds = []
        for fname in saved_frames:
            stem = Path(fname).stem
            if stem in frame_map:
                selected_fds.append(frame_map[stem])

        selected_fds = sorted(selected_fds, key=lambda x: x["frame_num"])

        if len(selected_fds) == 0:
            continue

        segments.append({
            "segment_id": seg_id,
            "source": "selected_frame_summary",
            "category_name": category_name,
            "category_str": category_to_string(category_name),
            "phrase": row.get("phrase", ""),
            "begin_time": row.get("begin_time", None),
            "end_time": row.get("end_time", None),
            "num_saved_frames": row.get("num_saved_frames", len(selected_fds)),
            "saved_frames": saved_frames,
            "frame_index": selected_fds,
        })

    return segments


# [NEW] fallback: build segments from transcript time ranges
def load_segments_from_transcript(transcript_json: str, frame_index: List[Dict]) -> List[Dict]:
    data = load_json(transcript_json)
    main_speaker_id = data.get("main_speaker_id", None)
    items = extract_transcript_items(data)

    segments = []

    for seg_id, item in enumerate(items):
        speaker_id = item.get("speaker_id", None)
        category_name = item.get("category_name", None)

        if main_speaker_id is not None and speaker_id != main_speaker_id:
            continue

        if not has_valid_category(category_name):
            continue

        begin, end = get_begin_end_from_item(item)
        begin_pad = max(0.0, begin - TRANSCRIPT_TIME_PADDING_SEC)
        end_pad = end + TRANSCRIPT_TIME_PADDING_SEC

        start_frame = int(math.floor(begin_pad * FRAME_FPS)) + FRAME_NUMBER_START
        end_frame = int(math.ceil(end_pad * FRAME_FPS)) + FRAME_NUMBER_START

        selected_fds = [
            fd for fd in frame_index
            if start_frame <= fd["frame_num"] <= end_frame
        ]

        if len(selected_fds) == 0:
            continue

        phrase = item.get("phrase") or item.get("text") or item.get("word") or ""

        segments.append({
            "segment_id": seg_id,
            "source": "transcript_time_range",
            "category_name": category_name,
            "category_str": category_to_string(category_name),
            "phrase": phrase,
            "speaker_id": speaker_id,
            "main_speaker_id": main_speaker_id,
            "begin_time": begin,
            "end_time": end,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_saved_frames": len(selected_fds),
            "saved_frames": [fd["frame_name"] + ".jpg" for fd in selected_fds],
            "frame_index": selected_fds,
        })

    return segments


# ============================================================
# LOAD OBJECTS
# ============================================================

def load_frame_objects(fd: Dict) -> Tuple[Tuple[int, int, int], List[Dict]]:
    image = cv2.imread(fd["image_path"])
    if image is None:
        raise RuntimeError(f"Failed to read image: {fd['image_path']}")

    image_shape = image.shape

    with open(fd["meta_path"], "r") as f:
        meta = json.load(f)

    if len(meta) == 0:
        return image_shape, []

    # [CHANGED] group by feature archive so each .npz is opened once
    feature_cache = {}

    objects = []

    for item in meta:
        archive_path = item["feature_archive_path"]

        if archive_path not in feature_cache:
            feature_cache[archive_path] = np.load(archive_path)

        feature_data = feature_cache[archive_path]
        feature_index = int(item["feature_index"])

        fg_key = item.get("fg_feature_key", "fg_features")
        concat_key = item.get("concat_feature_key", "concat_features")

        fg_feat = feature_data[fg_key][feature_index]
        concat_feat = feature_data[concat_key][feature_index]

        objects.append({
            "frame_idx": fd["frame_idx"],
            "frame_name": item["frame_name"],
            "local_idx": int(item["local_idx"]),
            "feature_index": feature_index,

            "mask_index": int(item["mask_index"]),
            "mask_archive_path": item["mask_archive_path"],
            "mask_archive_key": item.get("mask_archive_key", "masks"),

            "feature_archive_path": archive_path,
            "fg_feature_key": fg_key,
            "ctx_feature_key": item.get("ctx_feature_key", "ctx_features"),
            "concat_feature_key": concat_key,

            "bbox": tuple(item["bbox"]) if item["bbox"] is not None else None,
            "expanded_bbox_xyxy": tuple(item["expanded_bbox_xyxy"]) if item.get("expanded_bbox_xyxy", None) is not None else None,
            "centroid": tuple(item["centroid"]),
            "predicted_iou": float(item.get("predicted_iou", 0.0)),
            "stability_score": float(item.get("stability_score", 0.0)),

            "fg_feature": fg_feat,
            "concat_feature": concat_feat,
        })

    # [CHANGED] keep only top proposals to speed Hungarian
    MAX_OBJECTS_PER_FRAME = 15 #80
    objects = sorted(
        objects,
        key=lambda x: 0.5 * x["predicted_iou"] + 0.5 * x["stability_score"],
        reverse=True,
    )[:MAX_OBJECTS_PER_FRAME]

    return image_shape, objects


# ============================================================
# DUPLICATE SUPPRESSION
# ============================================================

def proposal_quality(obj: Dict) -> float:
    return 0.5 * obj["predicted_iou"] + 0.5 * obj["stability_score"]


def should_suppress_duplicate_by_feature(a: Dict, b: Dict) -> bool:
    biou = bbox_iou(a["bbox"], b["bbox"])
    csim = cosine_similarity(a["concat_feature"], b["concat_feature"])
    return biou >= INTRA_BBOX_IOU_THRESH and csim >= INTRA_CONCAT_SIM_THRESH


# [NEW] suppress small mask mostly inside larger mask
def containment_ratio(mask_small: np.ndarray, mask_large: np.ndarray) -> float:
    small_area = float(mask_small.sum())
    if small_area <= 0:
        return 0.0
    inter = float(np.logical_and(mask_small > 0, mask_large > 0).sum())
    return inter / small_area


def should_suppress_by_containment(obj_small: Dict, obj_large: Dict) -> bool:
    area_small = bbox_area(obj_small["bbox"])
    area_large = bbox_area(obj_large["bbox"])

    if area_small <= 0 or area_large <= 0:
        return False

    if area_small > area_large * SMALL_TO_LARGE_AREA_RATIO:
        return False

    try:
        mask_small = load_mask_from_item(obj_small)
        mask_large = load_mask_from_item(obj_large)
    except Exception:
        return False

    contain = containment_ratio(mask_small, mask_large)
    return contain >= CONTAINMENT_THRESH


def suppress_intra_frame_duplicates(objects: List[Dict]) -> List[Dict]:
    if not USE_INTRA_FRAME_DUP_SUPPRESSION or len(objects) <= 1:
        return objects

    objs = sorted(objects, key=proposal_quality, reverse=True)
    kept = []

    for obj in objs:
        suppress = False

        for prev in kept:
            if should_suppress_duplicate_by_feature(obj, prev):
                suppress = True
                break

            # [NEW] small contained mask suppression
            if USE_CONTAINMENT_SUPPRESSION:
                area_obj = bbox_area(obj["bbox"])
                area_prev = bbox_area(prev["bbox"])

                if area_obj < area_prev and should_suppress_by_containment(obj, prev):
                    suppress = True
                    break

        if not suppress:
            kept.append(obj)

    return kept


# ============================================================
# TRACK STRUCTURES
# ============================================================

def init_track(track_id: int, obj: Dict, frame_idx: int) -> Dict:
    return {
        "track_id": track_id,
        "status": "active",
        "created_frame_idx": frame_idx,
        "last_seen_frame_idx": frame_idx,
        "missed_frames": 0,
        "items": [{
            "frame_idx": frame_idx,
            "frame_name": obj["frame_name"],

            "mask_index": obj["mask_index"],
            "mask_archive_path": obj["mask_archive_path"],
            "mask_archive_key": obj["mask_archive_key"],

            "feature_index": obj["feature_index"],
            "feature_archive_path": obj["feature_archive_path"],
            "fg_feature_key": obj["fg_feature_key"],
            "ctx_feature_key": obj["ctx_feature_key"],
            "concat_feature_key": obj["concat_feature_key"],

            "bbox": obj["bbox"],
            "centroid": obj["centroid"],
            "predicted_iou": obj["predicted_iou"],
            "stability_score": obj["stability_score"],
        }],
        "recent_fg_features": [obj["fg_feature"]],
        "recent_concat_features": [obj["concat_feature"]],
        "long_fg_features": [obj["fg_feature"]],
        "long_concat_features": [obj["concat_feature"]],
        "prototype_fg": obj["fg_feature"].copy(),
        "prototype_concat": obj["concat_feature"].copy(),
        "recent_centroids": [obj["centroid"]],
        "recent_bboxes": [obj["bbox"]],
        "match_metrics": [],
    }


# [MODIFIED] pass adaptive memory params instead of global memory values
def update_track(track: Dict, obj: Dict, frame_idx: int, metric: Dict, temporal_params: Dict) -> None:
    short_memory = temporal_params["short_term_memory"]
    long_memory = temporal_params["long_term_memory"]

    track["status"] = "active"
    track["last_seen_frame_idx"] = frame_idx
    track["missed_frames"] = 0

    track["items"].append({
        "frame_idx": frame_idx,
        "frame_name": obj["frame_name"],

        "mask_index": obj["mask_index"],
        "mask_archive_path": obj["mask_archive_path"],
        "mask_archive_key": obj["mask_archive_key"],

        "feature_index": obj["feature_index"],
        "feature_archive_path": obj["feature_archive_path"],
        "fg_feature_key": obj["fg_feature_key"],
        "ctx_feature_key": obj["ctx_feature_key"],
        "concat_feature_key": obj["concat_feature_key"],

        "bbox": obj["bbox"],
        "centroid": obj["centroid"],
        "predicted_iou": obj["predicted_iou"],
        "stability_score": obj["stability_score"],
    })

    track["recent_fg_features"].append(obj["fg_feature"])
    track["recent_fg_features"] = track["recent_fg_features"][-short_memory:]

    track["recent_concat_features"].append(obj["concat_feature"])
    track["recent_concat_features"] = track["recent_concat_features"][-short_memory:]

    track["long_fg_features"].append(obj["fg_feature"])
    track["long_fg_features"] = track["long_fg_features"][-long_memory:]

    track["long_concat_features"].append(obj["concat_feature"])
    track["long_concat_features"] = track["long_concat_features"][-long_memory:]

    proto_fg = safe_mean_feature(track["recent_fg_features"])
    proto_concat = safe_mean_feature(track["recent_concat_features"])

    if proto_fg is not None:
        track["prototype_fg"] = proto_fg

    if proto_concat is not None:
        track["prototype_concat"] = proto_concat

    track["recent_centroids"].append(obj["centroid"])
    track["recent_centroids"] = track["recent_centroids"][-short_memory:]

    track["recent_bboxes"].append(obj["bbox"])
    track["recent_bboxes"] = track["recent_bboxes"][-short_memory:]

    track["match_metrics"].append(metric)


def mark_track_missed(track: Dict) -> None:
    track["missed_frames"] += 1
    track["status"] = "inactive"


# [MODIFIED] adaptive inactive patience
def prune_dead_tracks(tracks: Dict[int, Dict], finished_tracks: List[Dict], temporal_params: Dict) -> None:
    inactive_patience = temporal_params["inactive_patience"]

    for tid in list(tracks.keys()):
        if tracks[tid]["missed_frames"] > inactive_patience:
            finished_tracks.append(tracks[tid])
            del tracks[tid]


# ============================================================
# SCORING
# ============================================================

def compute_motion_consistency(track: Dict, obj: Dict) -> float:
    pts = track["recent_centroids"]

    if len(pts) < 2:
        return 0.5

    last_pt = pts[-1]
    prev_pt = pts[-2]

    track_dx = last_pt[0] - prev_pt[0]
    track_dy = last_pt[1] - prev_pt[1]

    obj_dx = obj["centroid"][0] - last_pt[0]
    obj_dy = obj["centroid"][1] - last_pt[1]

    vel_dist = math.sqrt((track_dx - obj_dx) ** 2 + (track_dy - obj_dy) ** 2)
    return max(0.0, 1.0 - min(vel_dist / 80.0, 1.0))


def compute_temporal_consistency(track: Dict, obj: Dict, image_shape) -> float:
    last_item = track["items"][-1]

    center_score = normalize_center_distance(
        last_item["centroid"],
        obj["centroid"],
        image_shape,
        max_frac=0.25,
    )

    bbox_score = bbox_iou(last_item["bbox"], obj["bbox"])
    area_score = area_ratio_score(last_item["bbox"], obj["bbox"], tol=4.0)

    return 0.5 * center_score + 0.3 * bbox_score + 0.2 * area_score


def compute_persistence_bonus(track: Dict) -> float:
    return min(len(track["items"]) / 5.0, 1.0)


def compute_appearance_similarity(track: Dict, obj: Dict) -> Tuple[float, float, float]:
    fg_current = obj["fg_feature"]
    concat_current = obj["concat_feature"]

    fg_sim_proto = cosine_similarity(track["prototype_fg"], fg_current)
    fg_sim_recent = max(cosine_similarity(f, fg_current) for f in track["recent_fg_features"])
    fg_sim = max(fg_sim_proto, fg_sim_recent)

    concat_sim_proto = cosine_similarity(track["prototype_concat"], concat_current)
    concat_sim_recent = max(cosine_similarity(f, concat_current) for f in track["recent_concat_features"])
    concat_sim = max(concat_sim_proto, concat_sim_recent)

    appearance = W_FG * fg_sim + W_CONCAT * concat_sim

    return float(appearance), float(fg_sim), float(concat_sim)


def score_track_to_object(track: Dict, obj: Dict, image_shape) -> Dict:
    appearance, fg_sim, concat_sim = compute_appearance_similarity(track, obj)
    temporal = compute_temporal_consistency(track, obj, image_shape)
    motion = compute_motion_consistency(track, obj)
    persistence = compute_persistence_bonus(track)
    sam_quality = 0.5 * obj["predicted_iou"] + 0.5 * obj["stability_score"]

    total = (
        W_APPEARANCE * appearance +
        W_TEMPORAL * temporal +
        W_MOTION * motion +
        W_PERSISTENCE * persistence +
        W_SAM * sam_quality
    )

    return {
        "track_id": track["track_id"],
        "track_status": track["status"],
        "appearance": float(appearance),
        "fg_sim": float(fg_sim),
        "concat_sim": float(concat_sim),
        "temporal": float(temporal),
        "motion": float(motion),
        "persistence": float(persistence),
        "sam_quality": float(sam_quality),
        "total": float(total),
    }


# [MODIFIED] stricter inactive matching to reduce force matching
def associate_objects_to_tracks_hungarian(
    tracks: Dict[int, Dict],
    curr_objects: List[Dict],
    image_shape,
) -> Tuple[List[Dict], set]:

    live_tracks = [tr for tr in tracks.values()]

    if len(live_tracks) == 0 or len(curr_objects) == 0:
        return [], set()

    score_cache = {}
    cost_matrix = np.full((len(live_tracks), len(curr_objects)), 1e6, dtype=np.float32)

    for i, tr in enumerate(live_tracks):
        for j, obj in enumerate(curr_objects):
            s = score_track_to_object(tr, obj, image_shape)
            score_cache[(i, j)] = s

            if s["appearance"] < MIN_APPEARANCE_SIM:
                continue

            if s["fg_sim"] < MIN_FG_SIM:
                continue

            if s["temporal"] < MIN_TEMPORAL_SCORE:
                continue

            if tr["status"] == "inactive":
                if s["fg_sim"] < MIN_FG_INACTIVE:
                    continue
                if s["total"] < MIN_TOTAL_INACTIVE:
                    continue
            else:
                if s["total"] < MIN_TOTAL_ACTIVE:
                    continue

            cost_matrix[i, j] = 1.0 - s["total"]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    used_objects = set()

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= 1e5:
            continue

        s = score_cache[(i, j)]

        matches.append({
            "track_id": live_tracks[i]["track_id"],
            "obj_idx": j,
            **s,
        })

        used_objects.add(j)

    return matches, used_objects


# ============================================================
# SAVE + VISUALIZE
# ============================================================

def summarize_track(track: Dict, segment: Dict) -> Dict:
    items = track["items"]
    start_item = items[0]
    end_item = items[-1]

    return {
        "segment_id": segment["segment_id"],
        "segment_source": segment["source"],
        "category_name": segment["category_name"],
        "category_str": segment["category_str"],
        "phrase": segment.get("phrase", ""),
        "begin_time": segment.get("begin_time", None),
        "end_time": segment.get("end_time", None),
        "segment_num_frames": len(segment["frame_index"]),
        "track_id": track["track_id"],
        "status": track["status"],
        "items": items,
        "start_frame_idx": start_item["frame_idx"],
        "end_frame_idx": end_item["frame_idx"],
        "start_frame_name": start_item["frame_name"],
        "end_frame_name": end_item["frame_name"],
        "num_items": len(items),
        "match_metrics": track["match_metrics"],
    }

# ============================================================
# [NEW] WITHIN-SEGMENT STITCHING
# ============================================================

def load_track_item_feature(item: Dict, feature_type: str) -> np.ndarray:
    return load_feature_from_item(item, feature_type)


def mean_track_feature(tracklet: Dict, feature_type: str) -> Optional[np.ndarray]:
    feats = []

    for item in tracklet["items"]:
        try:
            feat = load_track_item_feature(item, feature_type)
            feats.append(feat)
        except Exception:
            continue

    return safe_mean_feature(feats)


def get_tracklet_end_velocity(tracklet: Dict) -> Tuple[float, float]:
    items = tracklet["items"]

    if len(items) < 2:
        return 0.0, 0.0

    c1 = items[-2]["centroid"]
    c2 = items[-1]["centroid"]

    return float(c2[0] - c1[0]), float(c2[1] - c1[1])


def predict_next_center(tracklet: Dict, gap: int) -> Tuple[float, float]:
    end_c = tracklet["items"][-1]["centroid"]
    vx, vy = get_tracklet_end_velocity(tracklet)

    return float(end_c[0] + vx * gap), float(end_c[1] + vy * gap)


def stitch_gap_score(gap: int, max_gap: int) -> float:
    if gap < 1 or gap > max_gap:
        return 0.0
    return float(1.0 - ((gap - 1) / max(max_gap, 1)))


def score_tracklet_stitch(a: Dict, b: Dict, max_gap: int, image_shape=(576, 1024, 3)) -> Dict:
    gap = int(b["start_frame_idx"] - a["end_frame_idx"])

    if gap < 1 or gap > max_gap:
        return {"valid": False, "total": 0.0}

    fg_a = mean_track_feature(a, "fg")
    fg_b = mean_track_feature(b, "fg")
    concat_a = mean_track_feature(a, "concat")
    concat_b = mean_track_feature(b, "concat")

    if fg_a is None or fg_b is None or concat_a is None or concat_b is None:
        return {"valid": False, "total": 0.0}

    fg_sim = cosine_similarity(fg_a, fg_b)
    concat_sim = cosine_similarity(concat_a, concat_b)

    if fg_sim < MIN_STITCH_FG_SIM:
        return {"valid": False, "total": 0.0}

    if concat_sim < MIN_STITCH_CONCAT_SIM:
        return {"valid": False, "total": 0.0}

    pred_center = predict_next_center(a, gap)
    start_center = b["items"][0]["centroid"]

    center_score = normalize_center_distance(
        pred_center,
        start_center,
        image_shape,
        max_frac=0.25,
    )

    area_score = area_ratio_score(
        a["items"][-1]["bbox"],
        b["items"][0]["bbox"],
        tol=4.0,
    )

    gap_score = stitch_gap_score(gap, max_gap)

    total = (
        W_STITCH_FG * fg_sim +
        W_STITCH_CONCAT * concat_sim +
        W_STITCH_CENTER * center_score +
        W_STITCH_AREA * area_score +
        W_STITCH_GAP * gap_score
    )

    return {
        "valid": total >= MIN_STITCH_TOTAL_SCORE,
        "total": float(total),
        "fg_sim": float(fg_sim),
        "concat_sim": float(concat_sim),
        "center_score": float(center_score),
        "area_score": float(area_score),
        "gap_score": float(gap_score),
        "gap": gap,
    }


def merge_two_tracklets(a: Dict, b: Dict, stitch_metric: Dict) -> Dict:
    merged = dict(a)

    merged["items"] = a["items"] + b["items"]
    merged["end_frame_idx"] = b["end_frame_idx"]
    merged["end_frame_name"] = b["end_frame_name"]
    merged["num_items"] = len(merged["items"])

    merged.setdefault("stitch_metrics", [])
    merged["stitch_metrics"].append({
        "from_track_id": a["track_id"],
        "to_track_id": b["track_id"],
        **stitch_metric,
    })

    return merged


def stitch_tracklets_within_segment(segment_tracklets: List[Dict], segment_num_frames: int) -> List[Dict]:
    if len(segment_tracklets) <= 1:
        return segment_tracklets

    max_gap = max(
        STITCH_MAX_GAP_MIN,
        min(int(round(segment_num_frames * STITCH_MAX_GAP_RATIO)), STITCH_MAX_GAP_MAX)
    )

    tracklets = sorted(segment_tracklets, key=lambda x: (x["start_frame_idx"], x["end_frame_idx"]))

    changed = True

    while changed:
        changed = False
        used = set()
        new_tracklets = []

        for i, a in enumerate(tracklets):
            if i in used:
                continue

            best_j = None
            best_score = None

            for j, b in enumerate(tracklets):
                if i == j or j in used:
                    continue

                if b["start_frame_idx"] <= a["end_frame_idx"]:
                    continue

                metric = score_tracklet_stitch(a, b, max_gap=max_gap)

                if not metric["valid"]:
                    continue

                if best_score is None or metric["total"] > best_score["total"]:
                    best_score = metric
                    best_j = j

            if best_j is not None:
                merged = merge_two_tracklets(a, tracklets[best_j], best_score)
                new_tracklets.append(merged)
                used.add(i)
                used.add(best_j)
                changed = True
            else:
                new_tracklets.append(a)
                used.add(i)

        tracklets = sorted(new_tracklets, key=lambda x: (x["start_frame_idx"], x["end_frame_idx"]))

    return tracklets

# [MODIFIED] converts NumPy/Python objects into JSON-safe types
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    return obj


# [MODIFIED] safe JSON saver
def save_json(obj, path: str) -> None:
    obj = make_json_serializable(obj)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_tracklets_json(tracklets: List[Dict], path: str) -> None:
    payload = []

    for i, tr in enumerate(tracklets):
        payload.append({
            "tracklet_id": i,
            "segment_id": tr["segment_id"],
            "category_name": tr["category_name"],
            "category_str": tr["category_str"],
            "phrase": tr["phrase"],
            "num_items": tr["num_items"],
            "start_frame_idx": tr["start_frame_idx"],
            "end_frame_idx": tr["end_frame_idx"],
            "start_frame_name": tr["start_frame_name"],
            "end_frame_name": tr["end_frame_name"],
            "items": tr["items"],
        })

    save_json(payload, path)


def save_tracks_json(tracklets: List[Dict], path: str) -> None:
    payload = []

    for gid, tr in enumerate(tracklets):
        payload.append({
            "global_track_id": gid,
            "segment_id": tr["segment_id"],
            "category_name": tr["category_name"],
            "category_str": tr["category_str"],
            "phrase": tr["phrase"],
            "begin_time": tr["begin_time"],
            "end_time": tr["end_time"],
            "num_items": tr["num_items"],
            "start_frame_idx": tr["start_frame_idx"],
            "end_frame_idx": tr["end_frame_idx"],
            "start_frame_name": tr["start_frame_name"],
            "end_frame_name": tr["end_frame_name"],
            "items": tr["items"],
        })

    save_json(payload, path)


def visualize_tracks(frame_index: List[Dict], tracklets: List[Dict], overlay_vis_dir: str, compare_vis_dir: str) -> None:
    if not SAVE_VISUALIZATIONS:
        return

    ensure_dir(overlay_vis_dir)
    ensure_dir(compare_vis_dir)

    frame_to_track_items = {}

    for gid, tr in enumerate(tracklets):
        for item in tr["items"]:
            frame_to_track_items.setdefault(item["frame_name"], []).append({
                "track_id": gid,
                "mask_index": item["mask_index"],
                "mask_archive_path": item["mask_archive_path"],
                "mask_archive_key": item.get("mask_archive_key", "masks"),
                "segment_id": tr["segment_id"],
            })

    for fd in frame_index:
        frame_name = fd["frame_name"]

        image = cv2.imread(fd["image_path"])
        if image is None:
            continue

        overlay = image.copy()

        for item in frame_to_track_items.get(frame_name, []):
            tid = item["track_id"]
            mask = load_mask_from_item(item)
            color = get_color(tid)

            overlay = overlay_mask(overlay, mask, color, MASK_ALPHA)

            ys, xs = np.where(mask > 0)
            if len(xs) > 0 and DRAW_TEXT:
                draw_text(
                    overlay,
                    f"G{tid}/S{item['segment_id']}",
                    (int(xs.mean()), int(ys.mean())),
                    color,
                    0.5,
                )

        compare = np.concatenate([image, overlay], axis=1)

        cv2.imwrite(os.path.join(overlay_vis_dir, f"{frame_name}_overlay.jpg"), overlay)
        cv2.imwrite(os.path.join(compare_vis_dir, f"{frame_name}_compare.jpg"), compare)


# ============================================================
# [NEW] TRACK ONE CATEGORY SEGMENT
# ============================================================

def track_one_segment(segment: Dict) -> List[Dict]:
    frame_index = segment["frame_index"]

    if len(frame_index) == 0:
        return []

    temporal_params = get_adaptive_temporal_params(len(frame_index))

    print(
        f"  Segment {segment['segment_id']} | "
        f"{segment['category_str']} | "
        f"frames={len(frame_index)} | "
        f"params={temporal_params}"
    )

    tracks: Dict[int, Dict] = {}
    finished_tracks: List[Dict] = []
    next_track_id = 0

    first = frame_index[0]
    _, first_objects = load_frame_objects(first)
    first_objects = suppress_intra_frame_duplicates(first_objects)

    for obj in first_objects:
        tracks[next_track_id] = init_track(next_track_id, obj, first["frame_idx"])
        next_track_id += 1

    for t in range(len(frame_index) - 1):
        curr_fd = frame_index[t + 1]

        image_shape, curr_objects = load_frame_objects(curr_fd)
        curr_objects = suppress_intra_frame_duplicates(curr_objects)

        matches, used_object_indices = associate_objects_to_tracks_hungarian(
            tracks,
            curr_objects,
            image_shape,
        )

        matched_track_ids = set()

        for m in matches:
            tid = m["track_id"]
            obj_idx = m["obj_idx"]
            update_track(
                tracks[tid],
                curr_objects[obj_idx],
                curr_fd["frame_idx"],
                m,
                temporal_params,
            )
            matched_track_ids.add(tid)

        for tid, tr in list(tracks.items()):
            if tid not in matched_track_ids:
                mark_track_missed(tr)

        for obj_idx, obj in enumerate(curr_objects):
            if obj_idx not in used_object_indices:
                tracks[next_track_id] = init_track(next_track_id, obj, curr_fd["frame_idx"])
                next_track_id += 1

        prune_dead_tracks(tracks, finished_tracks, temporal_params)

        del curr_objects
        gc.collect()

    all_tracks = finished_tracks + list(tracks.values())

    tracklets = [
        summarize_track(tr, segment)
        for tr in all_tracks
        if len(tr["items"]) > 0
    ]

    return sorted(tracklets, key=lambda x: (x["start_frame_idx"], x["track_id"]))


# ============================================================
# PROCESS ONE GX
# ============================================================

def process_one_job(job: Dict) -> None:
    ensure_dir(job["output_root"])
    ensure_dir(job["overlay_vis_dir"])
    ensure_dir(job["compare_vis_dir"])

    print("=" * 90)
    print(f"Camera{job['camera_id']} | {job['gx_id']}")
    print(f"Frames    : {job['frames_dir']}")
    print(f"Meta      : {job['meta_dir']}")
    print(f"Transcript: {job['transcript_json']}")
    print(f"Selected summary: {job['selected_summary_json']}")
    print(f"Output    : {job['output_root']}")

    full_frame_index = build_frame_index(job["frames_dir"], job["meta_dir"])

    if len(full_frame_index) == 0:
        print(f"Skipping {job['gx_id']}: no frames with proposal meta.")
        return

    # [NEW] prefer selected_frame_summary.json, because it gives exact saved frame lists
    if job["selected_summary_json"] is not None:
        segments = load_segments_from_selected_summary(
            job["selected_summary_json"],
            full_frame_index,
        )
    else:
        segments = load_segments_from_transcript(
            job["transcript_json"],
            full_frame_index,
        )

    segment_info_payload = []
    for seg in segments:
        segment_info_payload.append({
            "segment_id": seg["segment_id"],
            "source": seg["source"],
            "category_name": seg["category_name"],
            "category_str": seg["category_str"],
            "phrase": seg.get("phrase", ""),
            "begin_time": seg.get("begin_time", None),
            "end_time": seg.get("end_time", None),
            "num_frames": len(seg["frame_index"]),
            "saved_frames": seg.get("saved_frames", []),
            "adaptive_temporal_params": get_adaptive_temporal_params(len(seg["frame_index"])),
        })

    save_json(segment_info_payload, job["segment_info_json_path"])

    print(f"Total frames with meta: {len(full_frame_index)}")
    print(f"Segments to track: {len(segments)}")

    if len(segments) == 0:
        print(f"Skipping {job['gx_id']}: no valid category segments.")
        return

    all_tracklets_before_stitching = []
    all_tracklets_after_stitching = []

    for seg in segments:
        seg_tracklets = track_one_segment(seg)
        all_tracklets_before_stitching.extend(seg_tracklets)

        # [NEW] stitch only within the same segment/category
        if ENABLE_WITHIN_SEGMENT_STITCHING:
            seg_tracklets = stitch_tracklets_within_segment(
                segment_tracklets=seg_tracklets,
                segment_num_frames=len(seg["frame_index"]),
            )

        all_tracklets_after_stitching.extend(seg_tracklets)

    all_tracklets = sorted(
        all_tracklets_after_stitching,
        key=lambda x: (x["segment_id"], x["start_frame_idx"], x["track_id"])
    )

    print(f"Total segment-level tracklets: {len(all_tracklets)}")

    save_tracklets_json(all_tracklets_before_stitching, job["tracklet_json_path"])
    save_tracks_json(all_tracklets, job["track_json_path"])
    save_json(all_tracklets, job["segment_json_path"])

    if SAVE_VISUALIZATIONS:
        vis_frame_map = {}
        for seg in segments:
            for fd in seg["frame_index"]:
                vis_frame_map[fd["frame_name"]] = fd

        vis_frames = list(vis_frame_map.values())
        vis_frames = sorted(vis_frames, key=lambda x: x["frame_num"])

        num_vis = int(np.ceil(len(vis_frames) * (MAX_VIS_FRAMES / 100.0)))
        num_vis = min(num_vis, len(vis_frames))
        if num_vis > 0:
            random_frame_index = random.sample(vis_frames, num_vis)
            visualize_tracks(
                random_frame_index,
                all_tracklets,
                job["overlay_vis_dir"],
                job["compare_vis_dir"],
            )

    print(f"Saved segment info: {job['segment_info_json_path']}")
    print(f"Saved tracklets   : {job['tracklet_json_path']}")
    print(f"Saved tracks      : {job['track_json_path']}")


# ============================================================
# MAIN
# ============================================================

def run():
    jobs = find_jobs()

    if len(jobs) == 0:
        raise RuntimeError("No valid jobs found. Check paths, proposal_meta, and transcript JSON files.")

    print(f"Found {len(jobs)} jobs.")

    for job in jobs:
        process_one_job(job)

    print("All done.")


if __name__ == "__main__":
    run()