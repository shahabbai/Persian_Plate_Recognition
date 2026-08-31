#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Luyao.zhang
   @Date:          2023/5/16
   @Description: configuration file
-------------------------------------------------
"""
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

print(ROOT)
# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]


# DL model config (YOLO26)
DETECTION_MODEL_DIR = ROOT / 'weights'
CARS_MODEL = DETECTION_MODEL_DIR / "cars_best_v26.pt"    # car/plate detection (data.yaml, 2 classes)
CHARS_MODEL = DETECTION_MODEL_DIR / "chars_best_v26.pt"  # plate char detection (ir_lpr.yaml, 39 classes)



DETECTION_MODEL_LIST = [
    "cars_best_v26.pt",
    "chars_best_v26.pt"]


OBJECT_COUNTER = None
OBJECT_COUNTER1 = None