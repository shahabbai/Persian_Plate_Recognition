#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo26_base.h"

#ifndef USE_TENSORRT
#error "PersianPlateRecognizer requires TensorRT support (build with -DUSE_TENSORRT)"
#endif

#include "yolo26_tensorrt.h"

/**
 * Result of the character recognition stage for a single detected plate.
 */
struct PlateResult {
    cv::Rect plate_box;                    // Plate bounding box in frame coordinates
    float plate_conf = 0.0f;               // Plate detection confidence
    std::vector<yolo26::Detection> chars;  // Char detections sorted left to right
    std::string text_latin;                // Left-to-right latin transliteration (for cv::putText)
    std::string text_persian;              // Left-to-right persian text (UTF-8, for console/files)
    bool complete = false;                 // True when `expected_chars` characters were read
};

/**
 * Result of the full two-stage pipeline for a single frame.
 */
struct FrameResult {
    std::vector<yolo26::Detection> objects;  // Stage 1: cars + plates
    std::vector<PlateResult> plates;         // Stage 2: per-plate char recognition
};

/**
 * Two-stage Persian license plate recognizer running on YOLO26 + TensorRT:
 *
 *   stage 1: car/plate detection       (2 classes, see data.yaml)
 *   stage 2: plate character detection on every plate crop
 *                                      (39 classes, see ir_lpr.yaml)
 *
 * Both stages use the YOLO26 C++ TensorRT backend with full CUDA
 * pre/post-processing (GPU letterbox + NMS-free end-to-end head).
 */
class PersianPlateRecognizer {
public:
    struct Params {
        std::string cars_model;    // .engine (preferred) or .onnx path
        std::string chars_model;   // .engine (preferred) or .onnx path
        int input_size = 640;      // Square network input (letterboxed)
        float cars_conf = 0.70f;   // Car/plate confidence threshold
        float chars_conf = 0.35f;  // Character confidence threshold
        int expected_chars = 8;    // A plate is "complete" with this many chars
        int device_id = 0;
    };

    // Class ids of the car/plate model (order matches data.yaml)
    static constexpr int kCarClassId = 0;
    static constexpr int kPlateClassId = 1;

    /**
     * Load both TensorRT engines. When given an .onnx path, the engine is
     * built once (FP16 when the GPU supports it) and cached next to the
     * ONNX file, so subsequent runs load the .engine directly.
     */
    bool initialize(const Params& params);

    // Run the full pipeline on one frame (no drawing).
    FrameResult process(const cv::Mat& frame) const;

    const Params& params() const { return params_; }

    // Latin transliteration of a char class id (for cv::putText overlays).
    // Returns "?" when the class id is out of range.
    std::string charNameLatin(int class_id) const;

    // Persian name of a char class id (UTF-8). Returns "?" when out of range.
    std::string charNamePersian(int class_id) const;

private:
    Params params_;
    std::unique_ptr<yolo26::YOLO26TensorRT> cars_detector_;
    std::unique_ptr<yolo26::YOLO26TensorRT> chars_detector_;

    // Character class names, index == class id (ir_lpr.yaml order, 39 classes)
    std::vector<std::string> char_names_persian_;
    std::vector<std::string> char_names_latin_;
};
