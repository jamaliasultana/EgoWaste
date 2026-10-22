### extract_bbox_crop_dino_fg_bg_features_compressed.py

import gc
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


# ============================================================
# PATHS
# ============================================================

PREPROC_ROOT = "/media/jisha/D/trash_dataset/Brookhaven MRF 20250611 Preproc"
# [NEW] save SAM outputs on E drive
OUTPUT_PREPROC_ROOT = "/media/jisha/E/trash_dataset/Brookhaven MRF 20250611 Preproc"

# [CHANGED] process Camera1 to Camera5
CAMERA_IDS = [1, 2, 3]

FRAMES_FOLDER_NAME = "frames_resized_1024"
MASKS_FOLDER_NAME = "sam2_masks"
OUTPUT_FOLDER_NAME = "dino_bbox_crop_feature_fg_ctx_cache"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_NAME = "dinov2_vitb14"

CROP_SIZE = 518
BBOX_EXPAND_RATIO = 0.50
MIN_CROP_SIZE = 50

FILL_VALUE_BGR = (0, 0, 0)
SKIP_IF_META_EXISTS = True

# [CHANGED] compressed feature archive per frame
FEATURE_ARCHIVE_NAME = "features.npz"

# [CHANGED] compressed mask archive per frame
MASK_ARCHIVE_NAME = "masks.npz"


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_compressed_mask(mask_dir: str, mask_index: int) -> np.ndarray:
    # [CHANGED] load one mask from masks.npz
    mask_npz_path = os.path.join(mask_dir, MASK_ARCHIVE_NAME)

    if not os.path.isfile(mask_npz_path):
        raise FileNotFoundError(f"Missing compressed mask archive: {mask_npz_path}")

    data = np.load(mask_npz_path)
    masks = data["masks"]

    if mask_index < 0 or mask_index >= masks.shape[0]:
        raise IndexError(f"mask_index {mask_index} out of range for {mask_npz_path}")

    return (masks[mask_index] > 0).astype(np.uint8)


def compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def compute_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def expand_bbox(box, image_shape, expand_ratio: float) -> Optional[Tuple[int, int, int, int]]:
    h, w = image_shape[:2]

    x, y, bw, bh = box
    x, y, bw, bh = int(round(x)), int(round(y)), int(round(bw)), int(round(bh))

    if bw <= 0 or bh <= 0:
        return None

    x1, y1 = x, y
    x2, y2 = x + bw - 1, y + bh - 1

    pad_w = int(round(bw * expand_ratio))
    pad_h = int(round(bh * expand_ratio))

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w - 1, x2 + pad_w)
    y2 = min(h - 1, y2 + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_expanded_region_and_mask(
    image_bgr: np.ndarray,
    full_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:

    expanded = expand_bbox(bbox, image_bgr.shape, BBOX_EXPAND_RATIO)
    if expanded is None:
        return None

    x1, y1, x2, y2 = expanded

    crop_bgr = image_bgr[y1:y2 + 1, x1:x2 + 1].copy()
    crop_mask = full_mask[y1:y2 + 1, x1:x2 + 1].copy()

    if crop_bgr.shape[0] < MIN_CROP_SIZE or crop_bgr.shape[1] < MIN_CROP_SIZE:
        return None

    return crop_bgr, crop_mask, expanded


def make_masked_foreground_crop(crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    fg = np.full_like(crop_bgr, FILL_VALUE_BGR, dtype=np.uint8)
    fg[crop_mask > 0] = crop_bgr[crop_mask > 0]
    return fg


def make_context_crop_object_removed(crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    ctx = crop_bgr.copy()
    ctx[crop_mask > 0] = np.array(FILL_VALUE_BGR, dtype=np.uint8)
    return ctx


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


# ============================================================
# FIND ALL GX JOBS
# ============================================================

def find_gx_jobs() -> List[Dict]:
    jobs = []

    for cam_id in CAMERA_IDS:
        camera_dir = Path(PREPROC_ROOT) / f"Camera {cam_id}"

        if not camera_dir.is_dir():
            camera_dir = Path(PREPROC_ROOT) / f"Camera{cam_id}"

        if not camera_dir.is_dir():
            print(f"Missing camera directory for Camera{cam_id}")
            continue

        for gx_dir in sorted(camera_dir.glob("GX*")):
            frames_dir = gx_dir / FRAMES_FOLDER_NAME

            # [CHANGED] SAM masks are stored on E drive
            masks_dir = (
                Path(OUTPUT_PREPROC_ROOT)
                / f"Camera {cam_id}"
                / gx_dir.name
                / MASKS_FOLDER_NAME
            )

            # [CHANGED] DINO features also stored on E drive
            output_root = (
                Path(OUTPUT_PREPROC_ROOT)
                / f"Camera {cam_id}"
                / gx_dir.name
                / OUTPUT_FOLDER_NAME
            )

            if not frames_dir.is_dir():
                continue

            if not masks_dir.is_dir():
                print(f"Skipping {gx_dir.name}: no sam2_masks folder")
                continue

            jobs.append({
                "camera_id": cam_id,
                "gx_id": gx_dir.name,
                "gx_dir": str(gx_dir),
                "frames_dir": str(frames_dir),
                "masks_dir": str(masks_dir),
                "output_root": str(output_root),
            })

    return jobs


def build_frame_index(frames_dir: str, masks_dir: str) -> List[Dict]:
    frame_paths = sorted([
        p for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    frame_index = []

    for fp in frame_paths:
        frame_name = fp.stem
        mask_dir = os.path.join(masks_dir, frame_name)
        masks_json = os.path.join(mask_dir, "masks.json")
        masks_npz = os.path.join(mask_dir, MASK_ARCHIVE_NAME)

        # [CHANGED] require compressed mask archive
        if os.path.isdir(mask_dir) and os.path.isfile(masks_json) and os.path.isfile(masks_npz):
            frame_index.append({
                "frame_name": frame_name,
                "image_path": str(fp),
                "mask_dir": mask_dir,
                "masks_json": masks_json,
                "masks_npz": masks_npz,
            })

    return frame_index


# ============================================================
# DINO
# ============================================================

def load_dino_model():
    model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL_NAME)
    model.eval().to(DEVICE)
    return model


def build_dino_transform():
    return T.Compose([
        T.Resize((CROP_SIZE, CROP_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def extract_dino_feature(model, transform, pil_img: Image.Image) -> np.ndarray:
    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        feat = model(x)

    if isinstance(feat, (tuple, list)):
        feat = feat[0]

    feat = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return l2_normalize(feat)


# ============================================================
# CACHE PATHS
# ============================================================

def make_output_dirs(output_root: str) -> Dict[str, str]:
    # [CHANGED] no fg_features/frame/*.npy, ctx_features/frame/*.npy, concat_features/frame/*.npy
    dirs = {
        "features": os.path.join(output_root, "compressed_features"),
        "meta": os.path.join(output_root, "proposal_meta"),
    }

    for d in dirs.values():
        ensure_dir(d)

    return dirs


def feature_archive_path(output_dirs: Dict[str, str], frame_name: str) -> str:
    # [CHANGED] one compressed feature archive per frame
    return os.path.join(output_dirs["features"], f"{frame_name}.npz")


def meta_json_path(output_dirs: Dict[str, str], frame_name: str) -> str:
    return os.path.join(output_dirs["meta"], f"{frame_name}.json")


# ============================================================
# PROCESS ONE GX
# ============================================================

def process_one_gx(model, transform, job: Dict) -> None:
    camera_id = job["camera_id"]
    gx_id = job["gx_id"]
    frames_dir = job["frames_dir"]
    masks_dir = job["masks_dir"]
    output_root = job["output_root"]

    output_dirs = make_output_dirs(output_root)
    frame_index = build_frame_index(frames_dir, masks_dir)

    if len(frame_index) == 0:
        print(f"No frames with compressed masks found for Camera{camera_id}/{gx_id}")
        return

    print("=" * 90)
    print(f"Camera{camera_id} | {gx_id}")
    print(f"Frames: {frames_dir}")
    print(f"Masks : {masks_dir}")
    print(f"Output: {output_root}")
    print(f"Frames with masks: {len(frame_index)}")

    for k, fd in enumerate(frame_index):
        frame_name = fd["frame_name"]
        meta_path = meta_json_path(output_dirs, frame_name)
        feat_archive = feature_archive_path(output_dirs, frame_name)

        if SKIP_IF_META_EXISTS and os.path.isfile(meta_path) and os.path.isfile(feat_archive):
            print(f"Skipping existing compressed features: {gx_id}/{frame_name}")
            continue

        print(f"[{k + 1}/{len(frame_index)}] {gx_id}/{frame_name}")

        image = cv2.imread(fd["image_path"])
        if image is None:
            print(f"Skipping unreadable frame: {fd['image_path']}")
            continue

        with open(fd["masks_json"], "r") as f:
            metadata = json.load(f)

        frame_meta = []

        # [CHANGED] collect all features into arrays, then save one .npz
        fg_features = []
        ctx_features = []
        concat_features = []
        kept_local_indices = []

        for idx, ann in enumerate(metadata):
            mask_index = int(ann.get("mask_index", idx))

            try:
                mask = load_compressed_mask(fd["mask_dir"], mask_index)
            except Exception as e:
                print(f"[WARNING] Could not load mask {frame_name}/{mask_index}: {e}")
                continue

            bbox = ann.get("bbox", None)
            if bbox is None:
                bbox = compute_bbox_from_mask(mask)
            else:
                bbox = tuple(int(round(v)) for v in bbox)

            if bbox is None:
                continue

            centroid = compute_centroid(mask)
            expanded_xyxy = expand_bbox(bbox, image.shape, BBOX_EXPAND_RATIO)

            crop_result = crop_expanded_region_and_mask(image, mask, bbox)
            if crop_result is None:
                continue

            expanded_crop_bgr, expanded_crop_mask, expanded_xyxy = crop_result

            fg_crop_bgr = make_masked_foreground_crop(
                expanded_crop_bgr,
                expanded_crop_mask
            )

            ctx_crop_bgr = make_context_crop_object_removed(
                expanded_crop_bgr,
                expanded_crop_mask
            )

            fg_feat = extract_dino_feature(model, transform, bgr_to_pil(fg_crop_bgr))
            ctx_feat = extract_dino_feature(model, transform, bgr_to_pil(ctx_crop_bgr))

            concat_feat = np.concatenate([fg_feat, ctx_feat], axis=0).astype(np.float32)
            concat_feat = l2_normalize(concat_feat)

            feature_index = len(fg_features)

            fg_features.append(fg_feat)
            ctx_features.append(ctx_feat)
            concat_features.append(concat_feat)
            kept_local_indices.append(idx)

            # [CHANGED] metadata points into compressed feature archive
            frame_meta.append({
                "local_idx": idx,
                "feature_index": feature_index,

                "frame_name": frame_name,

                "mask_index": mask_index,
                "mask_key": ann.get("mask_key", f"mask_{mask_index:03d}"),
                "mask_archive_path": fd["masks_npz"],
                "mask_archive_key": "masks",

                # [CHANGED] old individual mask paths removed
                "mask_file": None,
                "mask_path": None,

                # [CHANGED] new compressed feature references
                "feature_archive_path": feat_archive,
                "fg_feature_key": "fg_features",
                "ctx_feature_key": "ctx_features",
                "concat_feature_key": "concat_features",

                # [CHANGED] old individual .npy paths removed
                "fg_feature_path": None,
                "ctx_feature_path": None,
                "concat_feature_path": None,

                "bbox": list(bbox),
                "expanded_bbox_xyxy": list(expanded_xyxy) if expanded_xyxy is not None else None,
                "centroid": [float(centroid[0]), float(centroid[1])],
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
            })

        if len(frame_meta) == 0:
            with open(meta_path, "w") as f:
                json.dump([], f, indent=2)
            continue

        fg_arr = np.stack(fg_features, axis=0).astype(np.float32)
        ctx_arr = np.stack(ctx_features, axis=0).astype(np.float32)
        concat_arr = np.stack(concat_features, axis=0).astype(np.float32)

        # [CHANGED] one compressed DINO feature file per frame
        np.savez_compressed(
            feat_archive,
            fg_features=fg_arr,
            ctx_features=ctx_arr,
            concat_features=concat_arr,
            local_indices=np.array(kept_local_indices, dtype=np.int32),
        )

        with open(meta_path, "w") as f:
            json.dump(frame_meta, f, indent=2)

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def run():
    print(f"Using device: {DEVICE}")
    print("Finding GX folders...")

    jobs = find_gx_jobs()

    if len(jobs) == 0:
        raise RuntimeError("No valid GX jobs found. Check PREPROC_ROOT, frames, and sam2_masks folders.")

    print(f"Found {len(jobs)} GX folders to process.")

    print("Loading DINOv2...")
    model = load_dino_model()
    transform = build_dino_transform()

    for job in jobs:
        process_one_gx(model, transform, job)

    print("Done.")


if __name__ == "__main__":
    run()