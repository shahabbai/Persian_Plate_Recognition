#include "yolo26_base.h"
#include <algorithm>
#include <cmath>

namespace yolo26 {

cv::Mat YOLO26Base::preprocessImage(const cv::Mat& image, cv::Size& original_size) {
    original_size = image.size();
    
    // Calculate scaling factor to maintain aspect ratio
    float scale = std::min(
        static_cast<float>(config_.input_width) / image.cols,
        static_cast<float>(config_.input_height) / image.rows
    );
    
    int new_width = static_cast<int>(image.cols * scale);
    int new_height = static_cast<int>(image.rows * scale);
    
    // Resize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));
    
    // Create padded image (letterbox)
    cv::Mat padded = cv::Mat::zeros(config_.input_height, config_.input_width, CV_8UC3);
    int offset_x = (config_.input_width - new_width) / 2;
    int offset_y = (config_.input_height - new_height) / 2;
    
    resized.copyTo(padded(cv::Rect(offset_x, offset_y, new_width, new_height)));
    
    // Convert BGR to RGB
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    
    // Normalize to [0, 1]
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    
    return padded;
}

std::vector<Detection> YOLO26Base::postprocessDetections(
    const float* output_data,
    size_t output_size,
    const cv::Size& original_size,
    const cv::Size& input_size
) {
    std::vector<Detection> detections;
    
    if (config_.end2end) {
        // Output is flat [x1, y1, x2, y2, conf, class_id, ...] for valid detections
        size_t stride = 6;
        size_t num_entries = output_size / stride;
        
        for (size_t i = 0; i < num_entries; ++i) {
            size_t offset = i * stride;
            
            float confidence = output_data[offset + 4];
            // No need to check threshold here (already done in runInference)
            
            Detection det;
            det.confidence = confidence;
            det.class_id = static_cast<int>(output_data[offset + 5]);
            
            if (det.class_id >= 0 && det.class_id < config_.class_names.size()) {
                det.class_name = config_.class_names[det.class_id];
            } else {
                det.class_name = "Unknown";
            }
            
            // Extract bounding box (xyxy, already scaled to input_size)
            float x1 = output_data[offset + 0];
            float y1 = output_data[offset + 1];
            float x2 = output_data[offset + 2];
            float y2 = output_data[offset + 3];
            
            cv::Rect box(
                static_cast<int>(x1),
                static_cast<int>(y1),
                static_cast<int>(x2 - x1),
                static_cast<int>(y2 - y1)
            );
            
            det.bbox = scaleCoords(box, original_size, input_size);
            detections.push_back(det);
        }
        
        // Optional: Apply NMS if overlaps occur (one-to-one head may not need it, but test)
        // nms(detections, config_.iou_threshold);
    }else {
        // One-to-many head: output shape is [batch, num_classes + 4, num_anchors]
        // This requires NMS post-processing
        // For simplicity, this example focuses on the end2end mode
        // Implementation would be similar to traditional YOLO postprocessing
    }
    
    return detections;
}

void YOLO26Base::nms(std::vector<Detection>& detections, float iou_threshold) {
    // Sort by confidence in descending order
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            if (detections[i].class_id == detections[j].class_id) {
                float iou = calculateIoU(detections[i].bbox, detections[j].bbox);
                if (iou > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    // Remove suppressed detections
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
            [&suppressed, &detections](const Detection& det) {
                return suppressed[&det - &detections[0]];
            }),
        detections.end()
    );
}

float YOLO26Base::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int area1 = box1.width * box1.height;
    int area2 = box2.width * box2.height;
    int union_area = area1 + area2 - intersection;
    
    return static_cast<float>(intersection) / union_area;
}

cv::Rect YOLO26Base::scaleCoords(const cv::Rect& box, const cv::Size& original_size, const cv::Size& input_size) {
    // Calculate scaling factors
    float scale = std::min(
        static_cast<float>(input_size.width) / original_size.width,
        static_cast<float>(input_size.height) / original_size.height
    );
    
    // Calculate padding
    int pad_x = (input_size.width - static_cast<int>(original_size.width * scale)) / 2;
    int pad_y = (input_size.height - static_cast<int>(original_size.height * scale)) / 2;
    
    // Remove padding and scale back to original size
    cv::Rect scaled_box;
    scaled_box.x = static_cast<int>((box.x - pad_x) / scale);
    scaled_box.y = static_cast<int>((box.y - pad_y) / scale);
    scaled_box.width = static_cast<int>(box.width / scale);
    scaled_box.height = static_cast<int>(box.height / scale);
    
    // Clip to image boundaries
    scaled_box.x = std::max(0, std::min(scaled_box.x, original_size.width - 1));
    scaled_box.y = std::max(0, std::min(scaled_box.y, original_size.height - 1));
    scaled_box.width = std::min(scaled_box.width, original_size.width - scaled_box.x);
    scaled_box.height = std::min(scaled_box.height, original_size.height - scaled_box.y);
    
    return scaled_box;
}

} // namespace yolo26
