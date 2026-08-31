# Persian Plate Recognition

Recognize Persian (Iran) vehicle license plates with **YOLO26**.

This repo detects cars and plates and then recognizes every persian character on the plate with a two-stage pipeline:

1. **Cars & plates detection** — YOLO26 model trained on `data.yaml` (2 classes: `Car`, `Plate`)
2. **Plate characters recognition** — YOLO26 model trained on `ir_lpr.yaml` (39 classes), run on every plate crop; the characters are sorted left→right to build the plate string

Two inference options are included:

| | **C++ / CUDA / TensorRT** (`cpp_tensorrt/`) | **Python / Ultralytics** (`main.py`, `app.py`) |
|---|---|---|
| Backend | TensorRT engines + full CUDA pre/post-processing | PyTorch (Ultralytics runtime) |
| Head | NMS-free end-to-end YOLO26 (one-to-one) | Ultralytics runtime |
| Best for | Deployment, maximum FPS, edge devices | Quick experiments |

## Demo


https://github.com/shahabbai/PersianPlateRecog/assets/133869713/ba761742-4dfa-4f29-a016-e3876cf0d8cf


## Models

| Model | Weights | Classes | Dataset config |
|---|---|---|---|
| Cars & plates | `weights/cars_best_v26.pt` | 2 (`Car`, `Plate`) | `data.yaml` |
| Plate characters | `weights/chars_best_v26.pt` | 39 (persian chars) | `ir_lpr.yaml` |

Both models are NMS-free **end-to-end** YOLO26 (one-to-one head, max 300 detections).

### Plate character classes (`ir_lpr.yaml`)

| id | class | latin display | id | class | latin display |
|---|---|---|---|---|---|
| 0–9 | `0` … `9` | `0` … `9` | 20 | `س` | `S` |
| 10 | `D` | `D` | 21 | `ش` | `SH` |
| 11 | `S` | `S` | 22 | `ص` | `SAD` |
| 12 | `الف` | `A` | 23 | `ط` | `T` |
| 13 | `ب` | `B` | 24 | `ظ` | `ZA` |
| 14 | `ت` | `T` | 25 | `ع` | `EIN` |
| 15 | `تشریفات` | `CER` | 26 | `ف` | `F` |
| 16 | `ث` | `S` | 27 | `ق` | `GH` |
| 17 | `ج` | `J` | 28 | `ل` | `L` |
| 18 | `د` | `D` | 29 | `م` | `M` |
| 19 | `ز` | `Z` | 30 | `ن` | `N` |
| 31 | `ه` | `H` | 35 | `ژ (معلولین و جانبازان)` | `ZHE` |
| 32 | `ه‌` | `H2` | 36 | `ک` | `K` |
| 33 | `و` | `V` | 37 | `گ` | `G` |
| 34 | `پ` | `P` | 38 | `ی` | `Y` |

The latin names are only used for drawing (`cv::putText` / console), since the OpenCV built-in fonts cannot render persian glyphs — the full persian string is always printed to the console.

## Option 1 — C++ inference with CUDA + TensorRT (recommended)

The `cpp_tensorrt/` folder contains a self-contained C++ port of this pipeline, built on the YOLO26 C++ TensorRT inference library:

- Two-stage `PersianPlateRecognizer` (cars/plates detection → plate char recognition)
- TensorRT engines, FP16 automatically enabled when the GPU supports it
- Full CUDA pre-processing (GPU letterbox + BGR→RGB + normalization) and post-processing (GPU confidence filtering + coordinate scaling)
- NMS-free end-to-end YOLO26 head (max 300 detections)
- Image / video / webcam input, annotated output saved to `output/`, recognized plates logged to the console

### 1. Requirements

- NVIDIA GPU + CUDA Toolkit
- TensorRT >= 8.5
- OpenCV >= 4.0 (with video I/O)
- CMake >= 3.15 and a C++17 compiler
- Python + `ultralytics` (only for the one-time ONNX export)

### 2. Export the weights to ONNX

```bash
pip install ultralytics
python3 cpp_tensorrt/scripts/export_onnx.py
# -> weights/cars_best_v26.onnx
# -> weights/chars_best_v26.onnx
```

or manually:

```bash
yolo export model=weights/cars_best_v26.pt format=onnx end2end=True
yolo export model=weights/chars_best_v26.pt format=onnx end2end=True
```

### 3. Build

```bash
cd cpp_tensorrt
./build.sh --tensorrt-root /path/to/TensorRT-x.x.x
# binary: cpp_tensorrt/build/persian_plate_recognizer
```

If TensorRT is installed in a default system location you can omit `--tensorrt-root` (or set the `TENSORRT_ROOT` env var).

### 4. Run

Run from the repo root so the default `weights/`, `assets/` and `output/` paths resolve:

```bash
# demo video
./cpp_tensorrt/build/persian_plate_recognizer --source assets/video.mp4

# single image
./cpp_tensorrt/build/persian_plate_recognizer --source assets/car.jpg

# webcam
./cpp_tensorrt/build/persian_plate_recognizer --source 0

# headless + explicitly set models
./cpp_tensorrt/build/persian_plate_recognizer --source assets/car.jpg --no-display \
    --cars-model weights/cars_best_v26.engine --chars-model weights/chars_best_v26.engine
```

The first run converts each `.onnx` to a `.engine` (FP16 when supported) and caches it next to the ONNX file — subsequent runs load the engine directly and start instantly. Press `q` (or `Esc`) in the preview window to quit.

### Options

| Option | Default | Description |
|---|---|---|
| `--source PATH/INDEX` | `assets/video.mp4` | Image file, video file or webcam index |
| `--cars-model PATH` | auto | Cars/plates model (`.engine` preferred, else `.onnx`) |
| `--chars-model PATH` | auto | Plate chars model (`.engine` preferred, else `.onnx`) |
| `--cars-conf FLOAT` | `0.70` | Car/plate confidence threshold |
| `--chars-conf FLOAT` | `0.35` | Char confidence threshold |
| `--input-size INT` | `640` | Square network input (letterboxed) |
| `--expected-chars N` | `8` | Chars needed to report a complete plate |
| `--output-dir DIR` | `output` | Where annotated results are saved |
| `--show-partial` | off | Also draw plates with a partial char read |
| `--no-save` | off | Don't save annotated results |
| `--no-display` | off | Headless mode (no preview window) |

Annotated results are written to `output/output_<timestamp>.jpg` (images) and `output/output_<timestamp>.mp4` (videos), matching the behaviour of the original python script.

## Option 2 — Python (Ultralytics)

```bash
pip install ultralytics
python main.py
```

or with the streamlit app:

```bash
pip install streamlit
streamlit run app.py
```

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shahabbai-persianplaterecog-app-84qgn4.streamlit.app/)

The python scripts now load the same YOLO26 weights (`weights/cars_best_v26.pt`, `weights/chars_best_v26.pt`).

## Datasets

1. Dataset for detection cars and plates [Link](https://universe.roboflow.com/shahab-jafari-1vorv/persian-car)
2. Dataset for detection chars of the plates [Link](https://universe.roboflow.com/shahab-jafari-1vorv/persian-plate-characters-mvinj)

## Project structure

```
Persian_Plate_Recognition/
├── main.py                    # python two-stage inference (Ultralytics)
├── app.py                     # streamlit demo app
├── config.py                  # python model/source config
├── data.yaml                  # cars & plates dataset config (2 classes)
├── ir_lpr.yaml                # plate chars dataset config (39 classes)
├── weights/
│   ├── cars_best_v26.pt       # YOLO26 cars/plates detector
│   └── chars_best_v26.pt      # YOLO26 plate chars detector
├── cpp_tensorrt/              # C++ / CUDA / TensorRT pipeline
│   ├── include/               # yolo26 backend headers + plate_pipeline.hpp
│   ├── src/                   # yolo26 TensorRT backend + main.cpp
│   ├── scripts/
│   │   └── export_onnx.py     # .pt -> ONNX (end2end) exporter
│   ├── CMakeLists.txt
│   └── build.sh
├── assets/                    # sample image / video
└── output/                    # annotated results
```

## Troubleshooting

- **`TensorRT not found` at cmake time** — pass `-DTENSORRT_ROOT_DIR=/path/to/TensorRT-x.x.x` (or `./build.sh --tensorrt-root ...`).
- **First run is slow** — TensorRT builds each engine once (1–2 minutes per model); the engines are cached as `.engine` next to the ONNX files and every later run loads them instantly.
- **`Failed to deserialize engine`** — TensorRT engines are specific to the GPU and TensorRT version that built them. Delete the cached `.engine` file and run again to rebuild it.
- **Weights not found** — run `python3 cpp_tensorrt/scripts/export_onnx.py` first, and start the app from the repo root (or pass `--cars-model` / `--chars-model` explicitly).
- **No preview window on a headless machine** — add `--no-display`; results are still saved to `output/` and logged to the console.

## Legacy YOLOv8 training results (previous version of this repo)

1. yolov8s model for cars and plates detection

![results (1)](https://github.com/shahabbai/PersianPlateRecog/assets/133869713/8cb0e04b-edc9-4f2a-b560-3daec538af6c)

2. yolov8n model for characters detection

![yolov8n_char_new_small](https://github.com/shahabbai/PersianPlateRecog/assets/133869713/59db56cf-94a4-4289-ad60-b8f58225b7c2)

## Acknowledgments

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- YOLO26 C++ TensorRT inference library (CUDA pre/post-processing backend)

