#pragma once

#ifdef USE_TENSORRT

#include <cuda_runtime.h>
#include <vector>

namespace yolo26 {
namespace cuda {

/**
 * CUDA-accelerated postprocessing for YOLO26
 * Performs confidence filtering and coordinate scaling on GPU
 */
class CudaPostprocessor {
public:
    CudaPostprocessor();
    ~CudaPostprocessor();
    
    /**
     * Initialize CUDA postprocessing
     * @param max_detections Maximum detections per image (300 for YOLO26)
     * @param num_classes Number of object classes
     * @param max_batch_size Maximum batch size to support
     */
    bool initialize(int max_detections, int num_classes, int max_batch_size = 8);
    
    /**
     * Detection structure (matches CPU version)
     */
    struct Detection {
        float x1, y1, x2, y2;  // Bounding box coordinates
        float confidence;       // Detection confidence
        int class_id;          // Class ID
    };
    
    /**
     * Postprocess detections on GPU for end-to-end YOLO26
     * @param logits_device Model logits output on device [batch, 300, 80]
     * @param boxes_device Model boxes output on device [batch, 300, 4] cxcywh normalized
     * @param batch_size Number of images in batch
     * @param conf_threshold Confidence threshold for filtering
     * @param scale Scaling factor from preprocessing
     * @param pad_x Horizontal padding from preprocessing
     * @param pad_y Vertical padding from preprocessing
     * @param orig_width Original image width
     * @param orig_height Original image height
     * @param stream CUDA stream for async execution
     * @return Vector of detections (host memory)
     */
    std::vector<std::vector<Detection>> postprocess(
        const float* logits_device,
        const float* boxes_device,
        int batch_size,
        float conf_threshold,
        const std::vector<float>& scales,
        const std::vector<int>& pad_x,
        const std::vector<int>& pad_y,
        const std::vector<int>& orig_widths,
        const std::vector<int>& orig_heights,
        cudaStream_t stream
    );
    
    /**
     * Postprocess single image
     */
    std::vector<Detection> postprocessSingle(
        const float* logits_device,
        const float* boxes_device,
        float conf_threshold,
        float scale,
        int pad_x,
        int pad_y,
        int orig_width,
        int orig_height,
        cudaStream_t stream
    );
    
    // Get buffer sizes
    size_t getDetectionBufferSize() const;
    
private:
    int max_detections_;
    int num_classes_;
    int max_batch_size_;
    
    // Device buffers
    Detection* d_detections_;          // Filtered detections (GPU)
    int* d_detection_counts_;          // Number of valid detections per image (GPU)
    Detection* h_detections_;          // Pinned host memory for fast transfer
    int* h_detection_counts_;          // Pinned host memory for counts
    
    size_t detection_buffer_size_;
    
    bool initialized_;
};

/**
 * CUDA kernel for postprocessing YOLO26 output
 * Filters by confidence and scales coordinates back to original image size
 */
void launchPostprocessKernel(
    const float* logits,           // [batch, max_det, num_classes]
    const float* boxes,            // [batch, max_det, 4] cxcywh
    int batch_size,
    int max_detections,
    int num_classes,
    float conf_threshold,
    float input_width,
    float input_height,
    const float* scales,           // [batch]
    const int* pad_x,              // [batch]
    const int* pad_y,              // [batch]
    const int* orig_widths,        // [batch]
    const int* orig_heights,       // [batch]
    CudaPostprocessor::Detection* output_detections,  // [batch * max_det]
    int* detection_counts,         // [batch]
    cudaStream_t stream
);

/**
 * CUDA kernel for NMS (Non-Maximum Suppression) if using one-to-many head
 * This is a parallel NMS implementation on GPU
 */
void launchNMSKernel(
    CudaPostprocessor::Detection* detections,
    int* detection_counts,
    int batch_size,
    int max_detections,
    float iou_threshold,
    cudaStream_t stream
);

} // namespace cuda
} // namespace yolo26

#endif // USE_TENSORRT