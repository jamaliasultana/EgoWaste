# EgoWaste baselines: Tables 4 and 5

This directory contains the historical training/evaluation code associated with
Tables 4 and 5, its data builders, and its local Python dependencies. Files have
been renamed for publication and imports updated together. Training logic,
metrics, defaults, and legacy output basenames have been preserved.
`SOURCE_MAP.json` records original filenames and checksums.

**Status:** this is a source-code release, not a verified end-to-end reproduction
of every number in the paper. Dataset/split availability and known differences
between saved runs and paper tables are described below. No weights, datasets,
predictions, or unrelated annotation-pipeline experiments are included.

## Files and table mapping

| Task | Entry point |
|---|---|
| Table 4: ResNet50, ConvNeXt-T, Swin-T, ViT-B classification | `train_classification.py` |
| Table 4: YOLO11-S, RetinaNet, Faster R-CNN detection | `train_detection.py` |
| Table 4: Mask R-CNN, DeepLabV3+, YOLO segmentation | `train_segmentation.py` |
| Table 4: TridentNet and Cascade Mask R-CNN | `train_detectron2.py` |
| Table 5: direct transfer and in-domain train/test matrix | `run_cross_dataset_matrix.py` |
| Table 5(c): EgoWaste pretraining | `pretrain_egowaste.py` |
| Table 5(b,c): ZeroWaste control/fine-tuning | `finetune_zerowaste.py` |
| Table 5(e,f): EgoWaste control/fine-tuning | `finetune_egowaste.py` |

Data builders: `prepare_classification_data.py`, `prepare_detection_data.py`,
and `prepare_segmentation_data.py`.

Shared dependencies: `annotation_utils.py` and `taxonomy.py`.
`pretrain_shared_materials.py` is retained because `finetune_zerowaste.py` imports
its five-class DeepLabV3+ model builder. It is not an additional claimed table row.
Do not replace that import with the all-material builder: their signatures differ.

The original classifier also exposes DenseNet121, but it is not a Table 4 model;
select the four model names above instead of `--all_models` for that table.
The separate later long-tail ablation script is outside this package's scope.

## Installation

Python 3.10 was observed in the existing research environments. The dependency
pins in `requirements.txt` were read from the current `trash_seg_env2` environment
on 2026-10-22; they are not a recovered lockfile for every historical experiment.
The corresponding observed PyTorch pair was torch 2.7.1+cu128 and torchvision
0.22.1+cu128.

Use a dedicated environment. Install the matching PyTorch/torchvision build for
your machine first, following the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/).
For a compatible Linux NVIDIA system, the observed pair can be installed with:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`segmentation-models-pytorch` is required for DeepLabV3+. Some historical code paths
fall back to torchvision DeepLabV3 if it is missing; that fallback is a different
architecture and must not be reported as DeepLabV3+.

### Detectron2 and TridentNet

Run `train_detectron2.py` in a separately configured compatible environment.
The observed existing Detectron2 environment used Python 3.10, torch 2.0.0+cu118,
torchvision 0.15.1+cu118, Detectron2 0.6, and NumPy 1.26.4. Do not apply the main
requirements file blindly to that older environment.

Follow the [official Detectron2 installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
TridentNet also requires the [Detectron2 TridentNet project](https://github.com/facebookresearch/detectron2/tree/main/projects/TridentNet):

```bash
git clone https://github.com/facebookresearch/detectron2.git "$HOME/detectron2_src"
```

Build/install a Detectron2 revision compatible with the chosen PyTorch environment,
and keep the project checkout at a compatible revision. Supply its
`projects/TridentNet` directory using `--trident_path`. Cascade Mask R-CNN uses
Detectron2's model-zoo configuration. Third-party repositories and pretrained
weights are external dependencies, not vendored into this directory.

## Required data

These scripts consume the original experiment source layout, not the exported
`datasets/v0.2/data/<recording>/annotations.csv` layout directly. The eight-recording
public subset is also not the full benchmark data used in the paper.

Expected EgoWaste source structure:

```text
<session-root>/
  Camera 1/
    GX.../
      data/integrated_corrections/meta2_integrated_corrected.csv
      frames_resized_1024/000001.jpg
```

Frames may be located in separate matching session roots supplied with
`--frames_roots` (Table 4) or `--ego_frames_roots` (Table 5). Source CSVs must retain
usable mask and image paths, frame names, object IDs, and labels. Original absolute
paths must be relocated when moving the source data. The classification/detection/
segmentation builders read `corrected_mask_path` or `mask_path` literally; they do
not resolve the exported release's relative mask paths.

Table 4 builders can fall back to older metadata when integrated CSVs are absent.
Their historical deletion filtering and discovery rules are not the validated
release exporter's rules. Use an explicitly selected, reviewed source snapshot.
Prepare split/index files and record the exact input snapshot for any reproduction.
Existing historical split/index files are not bundled here.

The ZeroWaste root must contain `train/`, `val/`, and `test/`, with image and semantic
mask directories. Supported names include `data` or `images` for images and
`sem_seg`, `semantic`, `masks`, or `labels` for masks. The shared label IDs are:
0 background, 1 rigid_plastic, 2 cardboard, 3 metal, 4 soft_plastic.

## Table 4 usage

Run these commands from this directory. They illustrate the interfaces, not a
recovered exact hyperparameter recipe for every paper result. Replace the input
paths and match epochs, sampling, losses, and splits to the original run logs.

```bash
EGO_ROOTS=("/path/to/landfill/session" "/path/to/mrf/session")
EGO_FRAMES=("/path/to/landfill/frames" "/path/to/mrf/frames")

python prepare_classification_data.py --roots "${EGO_ROOTS[@]}" --frames_roots "${EGO_FRAMES[@]}" --modes crop_ctx --out_dir outputs/classification_data
python prepare_detection_data.py --roots "${EGO_ROOTS[@]}" --frames_roots "${EGO_FRAMES[@]}" --out_dir outputs/detection_data
python prepare_segmentation_data.py --roots "${EGO_ROOTS[@]}" --frames_roots "${EGO_FRAMES[@]}" --out_dir outputs/segmentation_data

python train_classification.py --data_dir outputs/classification_data --model resnet50 --label_level material --mode crop_ctx --out_dir outputs/classification
python train_detection.py --data_dir outputs/detection_data --model yolo11s --label_mode agnostic --out_dir outputs/detection
python train_segmentation.py --data_dir outputs/segmentation_data --model mask_rcnn --label_mode agnostic --out_dir outputs/segmentation
```

Classification model options for Table 4: `resnet50`, `convnext_tiny`, `swin_tiny`,
`vit_base`; label levels: `material`, `waste_class`.
Detection options: `yolo11s`, `retinanet`, `faster_rcnn`.
Segmentation options: `mask_rcnn`, `deeplabv3plus`, `yolo9_seg`.
Detection/segmentation label modes: `agnostic`, `material`.

Train the Detectron2 models in the Detectron2 environment, using absolute paths
when switching environments/directories:

```bash
python train_detectron2.py --model trident --task det --data_dir outputs/detection_data --label_mode agnostic --trident_path "$HOME/detectron2_src/projects/TridentNet" --out_dir outputs/trident_agnostic
python train_detectron2.py --model cascade --task seg --data_dir outputs/segmentation_data --label_mode agnostic --out_dir outputs/cascade_agnostic
```

Evaluate their predictions with the common metrics in the main environment:

```bash
python train_detection.py --data_dir outputs/detection_data --label_mode agnostic --import_preds outputs/trident_agnostic/inference/ego_test_agnostic/coco_instances_results.json --import_name tridentnet --out_dir outputs/detection
python train_segmentation.py --data_dir outputs/segmentation_data --label_mode agnostic --import_preds outputs/cascade_agnostic/inference/ego_test_agnostic/coco_instances_results.json --import_name cascade_mask_rcnn --out_dir outputs/segmentation
```

For material runs, train with `--label_mode material`, use separate output folders,
and import the matching `ego_test_material` predictions. Never import agnostic
predictions as a material-aware run.

## Table 5 usage

The cross-matrix script's `--all` runs three training regimes (EgoWaste, ZeroWaste,
and joint training), each tested on two datasets. Its six cells are not the final
paper's six sequential-transfer rows. Fine-tuning uses the separate entry points.

Using the source roots defined above:

```bash
ZERO_ROOT="/path/to/zerowaste/splits_final_deblurred"

# (a) EgoWaste -> ZeroWaste direct transfer
python run_cross_dataset_matrix.py --ego_roots "${EGO_ROOTS[@]}" --ego_frames_roots "${EGO_FRAMES[@]}" --zero_root "$ZERO_ROOT" --train ego --test zero --out_dir outputs/ego_to_zero

# (d) ZeroWaste -> EgoWaste direct transfer
python run_cross_dataset_matrix.py --ego_roots "${EGO_ROOTS[@]}" --ego_frames_roots "${EGO_FRAMES[@]}" --zero_root "$ZERO_ROOT" --train zero --test ego --out_dir outputs/zero_to_ego

# (b) ZeroWaste-only control; also produces a checkpoint for reverse transfer
python finetune_zerowaste.py --zero_root "$ZERO_ROOT" --init imagenet --encoder resnet101 --tag zerowaste_only --out_dir outputs/zerowaste_control

# (c) EgoWaste pretraining followed by encoder transfer to ZeroWaste
python pretrain_egowaste.py --ego_roots "${EGO_ROOTS[@]}" --ego_frames_roots "${EGO_FRAMES[@]}" --encoder resnet101 --out_dir outputs/egowaste_pretrain
python finetune_zerowaste.py --zero_root "$ZERO_ROOT" --init outputs/egowaste_pretrain/ego_all_materials_resnet101.pt --encoder resnet101 --transfer encoder --tag ego_pretrained_encoder --out_dir outputs/zerowaste_transfer

# (e) EgoWaste-only control with the same fine-tuning architecture/settings as (f)
python finetune_egowaste.py --ego_roots "${EGO_ROOTS[@]}" --ego_frames_roots "${EGO_FRAMES[@]}" --init imagenet --encoder resnet101 --tag egowaste_only --out_dir outputs/egowaste_control

# (f) ZeroWaste pretraining followed by encoder transfer to EgoWaste
python finetune_egowaste.py --ego_roots "${EGO_ROOTS[@]}" --ego_frames_roots "${EGO_FRAMES[@]}" --init outputs/zerowaste_control/zerowaste_zerowaste_only.pt --encoder resnet101 --transfer encoder --tag zero_pretrained_encoder --out_dir outputs/egowaste_transfer
```

The archived Table 5(e) result was found in cross-matrix outputs; the matching
fine-tuning control above is a controlled comparison for a new run, not a claim
that it reproduces that archived row. The matrix uses ResNet50; the fine-tuning
scripts default to ResNet101. For each claimed pretraining benefit, match the
control and transfer architecture, split, schedule, evaluated classes, and seeds.

## Reproduction notes

- The original classifier defaults to `--loss logit_adjust`, filters classes with
  fewer than 10 samples, and removes classes absent from training. Saved runs
  contained 50 fine classes, despite the paper's 56-class description.
- The YOLO segmentation checkpoint is `yolov9c-seg.pt`; the paper labels the model
  YOLO9-M. The name discrepancy has not been resolved by renaming these files.
- Segmentation output `top1_mask_iou` is mean IoU, while `top1_hit@0.5` is a success
  rate. Detection `top1_loc` ignores material correctness; `top1_loc_cls` requires it.
  Keep these metrics distinct when preparing tables.
- Table 5 defaults average the subset background/rigid_plastic/cardboard. The model
  head still has five classes. Record `--eval_classes` explicitly in run provenance;
  a five-class head does not imply a five-class reported average.
- Several saved table entries do not match the current result summaries. This
  package does not alter the numbers or certify that historical tables reproduce.
- Cascade Mask R-CNN foreground mIoU imported from score-filtered predictions is
  threshold-sensitive. Use matched thresholds before comparing that metric.

Use `python <entry-point>.py --help` for the complete interface. Write generated
artifacts under `outputs/`; it is ignored by Git. No full training was performed
as part of preparing this code package.
