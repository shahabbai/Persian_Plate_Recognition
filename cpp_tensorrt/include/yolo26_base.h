#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace yolo26 {

// Detection result structure
struct Detection {
    cv::Rect bbox;           // Bounding box
    float confidence;
    size_t output_size;        // Detection confidence
    int class_id;           // Class ID
    std::string class_name; // Class name
    cv::Mat mask;           // Instance segmentation mask (if available)
};

// Model configuration
struct ModelConfig {
    std::string model_path;
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_detections = 300;  // YOLO26 default for end-to-end inference
    bool end2end = true;        // Use one-to-one head (no NMS required)
    bool use_cuda_prepost = true;       
    int num_classes = 80;
    std::vector<std::string> class_names;
    
    // GPU settings
    int device_id = 0;
    
    ModelConfig() {
        // Initialize with COCO class names
        class_names = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        };
    }
};

// Base class for YOLO26 inference
class YOLO26Base {
public:
    virtual ~YOLO26Base() = default;
    
    // Initialize the model
    virtual bool initialize(const ModelConfig& config) = 0;
    
    // Run inference on a single image
    virtual std::vector<Detection> detect(const cv::Mat& image) = 0;
    
    // Run inference on multiple images (batch processing)
    virtual std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& images) = 0;
    
    // Get model information
    virtual std::string getBackendName() const = 0;
    virtual bool isInitialized() const = 0;

protected:
    ModelConfig config_;
    bool initialized_ = false;
    
    // Preprocessing utilities
    cv::Mat preprocessImage(const cv::Mat& image, cv::Size& original_size);
    
    // Postprocessing utilities
    std::vector<Detection> postprocessDetections(
        const float* output_data,
        size_t output_size,
        const cv::Size& original_size,
        const cv::Size& input_size
    );
    
    // Non-Maximum Suppression (for one-to-many head)
    void nms(std::vector<Detection>& detections, float iou_threshold);
    
    // Calculate IoU between two boxes
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    
    // Scale coordinates back to original image size
    cv::Rect scaleCoords(const cv::Rect& box, const cv::Size& original_size, const cv::Size& input_size);
};

} // namespace yolo26
