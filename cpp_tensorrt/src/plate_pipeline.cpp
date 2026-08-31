#include "plate_pipeline.hpp"

#include <algorithm>
#include <iostream>

namespace {

// ---------------------------------------------------------------------------
// Character classes of the plate char model, exactly in ir_lpr.yaml order
// (39 classes).
//
// `kCharNamesLatin` is used for cv::putText overlays because the OpenCV
// Hershey fonts cannot render Persian/Arabic glyphs.
// ---------------------------------------------------------------------------
const std::vector<std::string> kCharNamesPersian = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "D", "S", "الف", "ب", "ت", "تشریفات", "ث", "ج", "د", "ز",
    "س", "ش", "ص", "ط", "ظ", "ع", "ف", "ق", "ل", "م",
    "ن", "ه", "ه‌", "و", "پ", "ژ (معلولین و جانبازان)", "ک", "گ", "ی"};

const std::vector<std::string> kCharNamesLatin = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "D", "S", "A", "B", "T", "CER", "S", "J", "D", "Z",
    "S", "SH", "SAD", "T", "ZA", "EIN", "F", "GH", "L", "M",
    "N", "H", "H2", "V", "P", "ZHE", "K", "G", "Y"};

// Car/plate class names in data.yaml order
const std::vector<std::string> kObjectClassNames = {"Car", "Plate"};

yolo26::ModelConfig makeModelConfig(
    const std::string& model_path,
    int num_classes,
    const std::vector<std::string>& class_names,
    const PersianPlateRecognizer::Params& params,
    float conf_threshold) {
    yolo26::ModelConfig config;
    config.model_path = model_path;
    config.input_width = params.input_size;
    config.input_height = params.input_size;
    config.conf_threshold = conf_threshold;
    config.iou_threshold = 0.45f;
    config.max_detections = 300;     // YOLO26 end-to-end head
    config.end2end = true;           // one-to-one head, NMS-free
    config.use_cuda_prepost = true;  // CUDA letterbox + postprocess
    config.num_classes = num_classes;
    config.class_names = class_names;
    config.device_id = params.device_id;
    return config;
}

}  // namespace

bool PersianPlateRecognizer::initialize(const Params& params) {
    params_ = params;

    char_names_persian_ = kCharNamesPersian;
    char_names_latin_ = kCharNamesLatin;

    // ---- Stage 1: car + plate detection -----------------------------------
    cars_detector_ = std::make_unique<yolo26::YOLO26TensorRT>();
    const yolo26::ModelConfig cars_config = makeModelConfig(
        params_.cars_model, static_cast<int>(kObjectClassNames.size()),
        kObjectClassNames, params_, params_.cars_conf);

    std::cout << "[PersianPlate] Loading cars/plates model: "
              << params_.cars_model << std::endl;
    if (!cars_detector_->initialize(cars_config)) {
        std::cerr << "[PersianPlate] Failed to initialize cars/plates model: "
                  << params_.cars_model << std::endl;
        return false;
    }

    // ---- Stage 2: plate character recognition ------------------------------
    chars_detector_ = std::make_unique<yolo26::YOLO26TensorRT>();
    const yolo26::ModelConfig chars_config = makeModelConfig(
        params_.chars_model, static_cast<int>(char_names_persian_.size()),
        char_names_persian_, params_, params_.chars_conf);

    std::cout << "[PersianPlate] Loading plate chars model: "
              << params_.chars_model << std::endl;
    if (!chars_detector_->initialize(chars_config)) {
        std::cerr << "[PersianPlate] Failed to initialize chars model: "
                  << params_.chars_model << std::endl;
        return false;
    }

    return true;
}

FrameResult PersianPlateRecognizer::process(const cv::Mat& frame) const {
    FrameResult result;
    if (frame.empty() || !cars_detector_ || !chars_detector_) {
        return result;
    }

    // Stage 1: cars + plates
    result.objects = cars_detector_->detect(frame);

    // Stage 2: recognize the characters of every detected plate
    for (const auto& det : result.objects) {
        if (det.class_id != kPlateClassId) {
            continue;
        }

        PlateResult plate;
        plate.plate_conf = det.confidence;
        // Clip the box to the frame before cropping
        plate.plate_box = det.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (plate.plate_box.width < 2 || plate.plate_box.height < 2) {
            continue;
        }

        const cv::Mat plate_img = frame(plate.plate_box).clone();
        plate.chars = chars_detector_->detect(plate_img);

        // Read the plate from left to right
        std::sort(plate.chars.begin(), plate.chars.end(),
                  [](const yolo26::Detection& a, const yolo26::Detection& b) {
                      return a.bbox.x < b.bbox.x;
                  });

        for (const auto& ch : plate.chars) {
            plate.text_latin += charNameLatin(ch.class_id);
            plate.text_persian += charNamePersian(ch.class_id);
        }
        plate.complete =
            static_cast<int>(plate.chars.size()) == params_.expected_chars;

        result.plates.push_back(std::move(plate));
    }

    // Left-most plate first
    std::sort(result.plates.begin(), result.plates.end(),
              [](const PlateResult& a, const PlateResult& b) {
                  return a.plate_box.x < b.plate_box.x;
              });

    return result;
}

std::string PersianPlateRecognizer::charNameLatin(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(char_names_latin_.size())) {
        return char_names_latin_[class_id];
    }
    return "?";
}

std::string PersianPlateRecognizer::charNamePersian(int class_id) const {
    if (class_id >= 0 &&
        class_id < static_cast<int>(char_names_persian_.size())) {
        return char_names_persian_[class_id];
    }
    return "?";
}
