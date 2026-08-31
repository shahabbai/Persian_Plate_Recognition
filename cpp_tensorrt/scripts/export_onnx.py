#!/usr/bin/env python3
"""
Export the YOLO26 plate weights (.pt) to ONNX for the C++/TensorRT pipeline.

The export produces NMS-free end-to-end ONNX models (YOLO26 one-to-one head,
max 300 detections), which is the layout the cpp_tensorrt pipeline expects.

Usage:
    python3 cpp_tensorrt/scripts/export_onnx.py                          # both models
    python3 cpp_tensorrt/scripts/export_onnx.py --weights weights/cars_best_v26.pt
    python3 cpp_tensorrt/scripts/export_onnx.py --weights weights/chars_best_v26.pt --imgsz 640

Equivalent manual command for a single model:
    yolo export model=weights/cars_best_v26.pt format=onnx end2end=True
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# Repo root = two levels above this script (cpp_tensorrt/scripts/)
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WEIGHTS = ["weights/cars_best_v26.pt", "weights/chars_best_v26.pt"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO26 weights to ONNX (end-to-end, NMS-free)")
    parser.add_argument(
        "--weights", nargs="+", default=DEFAULT_WEIGHTS,
        help="List of .pt weights to export (default: %(default)s, "
             "resolved against the repo root)")
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Export image size (default: %(default)s)")
    return parser.parse_args()


def export_one(weights_path: Path, imgsz: int) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"[export] {weights_path} -> ONNX (imgsz={imgsz}, end2end=True)")
    model = YOLO(str(weights_path))
    onnx_path = model.export(format="onnx", imgsz=imgsz, end2end=True,
                             simplify=True)
    print(f"[export] done: {onnx_path}")


def main():
    args = parse_args()
    for weights in args.weights:
        path = Path(weights)
        # Default paths are relative to the repo root, user-provided paths
        # are relative to the current working directory.
        if not path.is_absolute() and str(weights) in DEFAULT_WEIGHTS:
            path = REPO_ROOT / path
        export_one(path, args.imgsz)
    print("\nAll models exported.")
    print("Run the C++ app once to build and cache the TensorRT .engine "
          "files next to the ONNX models.")


if __name__ == "__main__":
    main()
