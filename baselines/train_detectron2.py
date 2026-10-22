#!/usr/bin/env python3
"""
train_detectron2.py
===============
Trains the two detectron2-only baselines on the EgoWaste active-object data and
writes predictions in COCO format, ready to import into the existing
evaluation scripts.

  Cascade Mask R-CNN   config ships inside detectron2 -- works immediately
  TridentNet           lives in detectron2's projects/ folder, which pip does
                       NOT install, so the source tree must be cloned once
                       (see --trident_path below; no rebuild is needed, the
                       module simply imports against the installed detectron2)

Run this inside the environment where detectron2 works (for you: hands23).
It deliberately imports nothing from the rebuttal codebase, so the numpy and
pillow pins in that environment are not disturbed.

USAGE
  # Cascade Mask R-CNN, instance segmentation, 15 material categories
  python train_detectron2.py --model cascade --task seg \
      --data_dir /abs/path/active_seg_data --label_mode material --iters 8000

  # TridentNet, detection, class-agnostic
  python train_detectron2.py --model trident --task det \
      --data_dir /abs/path/active_det_data --label_mode agnostic --iters 8000 \
      --trident_path ~/detectron2_src/projects/TridentNet

Afterwards the script prints the exact --import_preds command to score the
result with the same metrics as every other model.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def register(data_dir, label_mode):
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import DatasetCatalog

    d = Path(data_dir)
    names = {}
    for split in ("train", "test"):
        ann = d / "coco" / f"{split}_{label_mode}.json"
        img = d / "images" / split
        if not ann.is_file():
            raise SystemExit(f"missing {ann} -- run the build script first")
        if not img.is_dir():
            raise SystemExit(f"missing {img}")
        name = f"ego_{split}_{label_mode}"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        register_coco_instances(name, {}, str(ann), str(img))
        names[split] = name
    with open(d / "coco" / f"train_{label_mode}.json") as f:
        n_classes = len(json.load(f)["categories"])
    return names, n_classes


def build_cfg(args, names, n_classes):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    cfg = get_cfg()

    if args.model == "trident":
        # the projects/ tree is not pip-installed; add it to the path so the
        # tridentnet module and its config extensions become importable
        tp = os.path.expanduser(args.trident_path)
        if not os.path.isdir(tp):
            raise SystemExit(
                f"TridentNet not found at {tp}\n"
                f"  git clone https://github.com/facebookresearch/detectron2.git "
                f"~/detectron2_src\n"
                f"then pass --trident_path ~/detectron2_src/projects/TridentNet")
        sys.path.insert(0, tp)
        try:
            from tridentnet import add_tridentnet_config
        except ImportError as e:
            raise SystemExit(f"could not import tridentnet from {tp}: {e}")
        add_tridentnet_config(cfg)
        cfg_file = os.path.join(tp, "configs", args.config or
                                "tridentnet_fast_R_50_C4_1x.yaml")
        cfg.merge_from_file(cfg_file)
        if not cfg.MODEL.WEIGHTS:
            cfg.MODEL.WEIGHTS = ("detectron2://ImageNetPretrained/MSRA/"
                                 "R-50.pkl")
    else:
        cfg_name = args.config or "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(cfg_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_name)

    cfg.DATASETS.TRAIN = (names["train"],)
    cfg.DATASETS.TEST = (names["test"],)
    cfg.DATALOADER.NUM_WORKERS = args.workers

    # detectron2 counts foreground classes only -- no background slot, unlike
    # torchvision, so this must NOT be n_classes + 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.NUM_CLASSES = n_classes
    cfg.MODEL.MASK_ON = (args.task == "seg")

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.STEPS = (int(args.iters * 0.7), int(args.iters * 0.9))
    cfg.SOLVER.CHECKPOINT_PERIOD = max(args.iters // 4, 1000)
    cfg.INPUT.MIN_SIZE_TRAIN = (args.min_size,)
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    cfg.INPUT.MAX_SIZE_TRAIN = args.max_size
    cfg.INPUT.MAX_SIZE_TEST = args.max_size
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    cfg.OUTPUT_DIR = os.path.abspath(args.out_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cascade", "trident"], required=True)
    ap.add_argument("--task", choices=["det", "seg"], required=True)
    ap.add_argument("--data_dir", required=True,
                    help="active_det_data or active_seg_data (absolute path)")
    ap.add_argument("--label_mode", choices=["agnostic", "material"],
                    default="agnostic")
    ap.add_argument("--config", default=None, help="override the config file")
    ap.add_argument("--trident_path",
                    default="~/detectron2_src/projects/TridentNet")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min_size", type=int, default=576)
    ap.add_argument("--max_size", type=int, default=1024)
    ap.add_argument("--score_thresh", type=float, default=0.01)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--weights", default=None,
                    help="checkpoint for --eval_only")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    args.data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if args.out_dir is None:
        args.out_dir = f"./d2_{args.model}_{args.task}_{args.label_mode}"

    from detectron2.engine import DefaultTrainer, DefaultPredictor, default_setup
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    print("=" * 74)
    print(f"detectron2 :: {args.model} | task={args.task} | "
          f"labels={args.label_mode}")
    print("=" * 74)

    names, n_classes = register(args.data_dir, args.label_mode)
    print(f"  datasets: {names}")
    print(f"  foreground classes: {n_classes}")

    cfg = build_cfg(args, names, n_classes)
    print(f"  output: {cfg.OUTPUT_DIR}")
    print(f"  mask_on: {cfg.MODEL.MASK_ON}   iters: {cfg.SOLVER.MAX_ITER}")

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference",
                                             dataset_name)
            os.makedirs(output_folder, exist_ok=True)
            # COCOEvaluator writes coco_instances_results.json with the
            # ORIGINAL category ids restored, which is what the import step
            # expects
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    if args.eval_only:
        if args.weights:
            cfg.MODEL.WEIGHTS = os.path.abspath(args.weights)
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        out = os.path.join(cfg.OUTPUT_DIR, "inference", names["test"])
        os.makedirs(out, exist_ok=True)
        loader = build_detection_test_loader(cfg, names["test"])
        inference_on_dataset(model, loader,
                             COCOEvaluator(names["test"], output_dir=out))
    else:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        trainer.test(cfg, trainer.model)

    res = os.path.join(cfg.OUTPUT_DIR, "inference", names["test"],
                       "coco_instances_results.json")
    print("\n" + "=" * 74)
    if os.path.isfile(res):
        with open(res) as f:
            n = len(json.load(f))
        print(f"  predictions: {n} instances")
        print(f"  file: {res}")
        script = ("train_segmentation.py" if args.task == "seg"
                  else "train_detection.py")
        tag = "cascade_mask_rcnn" if args.model == "cascade" else "tridentnet"
        print("\n  Score it with the same metrics as every other model:")
        print(f"    python {script} \\")
        print(f"        --data_dir {args.data_dir} \\")
        print(f"        --label_mode {args.label_mode} \\")
        print(f"        --import_preds {res} \\")
        print(f"        --import_name {tag}")
    else:
        print(f"  [!] no results file at {res}")
        print("      check the log above for a training or evaluation failure")


if __name__ == "__main__":
    main()
