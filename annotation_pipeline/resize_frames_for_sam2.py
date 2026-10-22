import os
import shutil 
from pathlib import Path
from typing import Tuple, List

import cv2


# ============================================================
# PATHS
# ============================================================

# [MODIFIED] parent directory containing GX030023, GX010023, etc.
BASE_DIR = "/media/jisha/D/trash_dataset/Brookhaven MRF 20250611"

# [MODIFIED] process all GX*/frames folders
INPUT_PATTERN = "Camera*/GX*/frames"

# [MODIFIED] output folder name inside each GX folder
OUTPUT_FOLDER_NAME = "frames_resized_1024"


# ============================================================
# SETTINGS
# ============================================================

TARGET_LONG_SIDE = 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SKIP_IF_EXISTS = True

DELETE_ORIGINAL_FRAMES = False  # [CHANGED] delete frames folder after successful resize/copy
SUMMARY_JSON_NAME = "selected_frame_summary.json"  # [CHANGED]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_new_size(h: int, w: int, target_long_side: int) -> Tuple[int, int]:
    long_side = max(h, w)
    scale = target_long_side / float(long_side)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    new_w = max(new_w, 1)
    new_h = max(new_h, 1)

    return new_h, new_w


# [MODIFIED] find all folders like GX030023/frames
def find_frame_dirs(base_dir: str, input_pattern: str) -> List[Path]:
    frame_dirs = sorted(Path(base_dir).glob(input_pattern))
    frame_dirs = [p for p in frame_dirs if p.is_dir()]
    return frame_dirs


def resize_one_folder(input_dir: Path) -> None:
    # [MODIFIED] output is sibling folder: GX030023/frames_resized_1024
    gx_dir = input_dir.parent
    output_dir = gx_dir / OUTPUT_FOLDER_NAME

    ensure_dir(str(output_dir))

    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    )

    if len(image_paths) == 0:
        print(f"No images found in: {input_dir}")
        return

    print("=" * 80)
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(image_paths)} images")

    saved_count = 0  # [CHANGED]

    for img_path in image_paths:
        out_path = output_dir / img_path.name

        if SKIP_IF_EXISTS and out_path.is_file():
            saved_count += 1  # [CHANGED]
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        h, w = image.shape[:2]
        new_h, new_w = compute_new_size(h, w, TARGET_LONG_SIDE)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok = cv2.imwrite(str(out_path), resized)
        if not ok:
            print(f"Failed to save: {out_path}")
            continue

        saved_count += 1  # [CHANGED]

    # ========================================================
    # [CHANGED] copy selected_frame_summary.json to resized folder
    # ========================================================

    summary_src = input_dir / SUMMARY_JSON_NAME
    summary_dst = output_dir / SUMMARY_JSON_NAME

    if summary_src.is_file():
        shutil.copy2(str(summary_src), str(summary_dst))
        print(f"Copied summary JSON to: {summary_dst}")
    else:
        print(f"[WARNING] No {SUMMARY_JSON_NAME} found in: {input_dir}")

    # ========================================================
    # [CHANGED] delete original frames folder only after resize
    # ========================================================

    if DELETE_ORIGINAL_FRAMES:
        output_images = [
            p for p in output_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(output_images) >= len(image_paths):
            shutil.rmtree(str(input_dir))
            print(f"Deleted original frames folder: {input_dir}")
        else:
            print(
                f"[WARNING] Not deleting {input_dir}: "
                f"only {len(output_images)}/{len(image_paths)} resized images found."
            )

    print(f"Finished: {gx_dir.name}")


def main() -> None:
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError(f"BASE_DIR does not exist: {BASE_DIR}")

    frame_dirs = find_frame_dirs(BASE_DIR, INPUT_PATTERN)

    if len(frame_dirs) == 0:
        raise RuntimeError(f"No frame directories found using pattern: {BASE_DIR}/{INPUT_PATTERN}")

    print(f"Found {len(frame_dirs)} GX frame folders.")

    for input_dir in frame_dirs:
        resize_one_folder(input_dir)

    print("All done.")


if __name__ == "__main__":
    main()