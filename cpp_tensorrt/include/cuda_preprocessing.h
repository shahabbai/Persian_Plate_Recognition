#pragma once

#ifdef USE_TENSORRT

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace yolo26 {
namespace cuda {

/**
 * CUDA-accelerated preprocessing for YOLO26
 * Performs letterbox resize, BGR to RGB conversion, and normalization on GPU
 */
class CudaPreprocessor {
public:
    CudaPreprocessor();
    ~CudaPreprocessor();
    
    /**
     * Initialize CUDA preprocessing with input/output dimensions
     * @param input_width Target width after preprocessing
     * @param input_height Target height after preprocessing
     * @param max_batch_size Maximum batch size to support
     */
    bool initialize(int input_width, int input_height, int max_batch_size = 8);
    
    /**
     * Preprocess a single image on GPU
     * @param image Input image (CV_8UC3, BGR)
     * @param output_device Output device pointer (float*, CHW format, RGB, normalized [0,1])
     * @param stream CUDA stream for async execution
     * @return Scaling parameters for postprocessing
     */
    struct PreprocessParams {
        float scale;      // Scaling factor used
        int pad_x;        // Horizontal padding
        int pad_y;        // Vertical padding
        int orig_width;   // Original image width
        int orig_height;  // Original image height
    };
    
    PreprocessParams preprocess(const cv::Mat& image, float* output_device, cudaStream_t stream);
    
    /**
     * Preprocess multiple images on GPU (batch)
     * @param images Vector of input images
     * @param output_device Output device pointer for batch
     * @param stream CUDA stream for async execution
     * @return Vector of preprocessing parameters for each image
     */
    std::vector<PreprocessParams> preprocessBatch(
        const std::vector<cv::Mat>& images, 
        float* output_device, 
        cudaStream_t stream
    );
    
    // Get buffer sizes
    size_t getInputBufferSize() const;
    size_t getOutputBufferSize() const;
    
private:
    int input_width_;
    int input_height_;
    int max_batch_size_;
    
    // Device buffers for intermediate processing
    uint8_t* d_input_buffer_;      // Raw image data (GPU)
    float* d_temp_buffer_;         // Temporary processing buffer
    
    size_t input_buffer_size_;
    size_t temp_buffer_size_;
    
    bool initialized_;
};

/**
 * CUDA kernel for letterbox preprocessing
 * Combines resize, padding, BGR->RGB, and normalization in one kernel
 */
void launchPreprocessKernel(
    const uint8_t* input,
    int input_width,
    int input_height,
    float* output,
    int output_width,
    int output_height,
    float scale,
    int pad_x,
    int pad_y,
    cudaStream_t stream
);

} // namespace cuda
} // namespace yolo26

#endif // USE_TENSORRT
