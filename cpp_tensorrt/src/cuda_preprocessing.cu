#ifdef USE_TENSORRT

#include "cuda_preprocessing.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

namespace yolo26 {
namespace cuda {

// CUDA kernel for letterbox preprocessing
// Combines resize, padding, BGR->RGB conversion, and normalization
__global__ void preprocessKernel(
    const uint8_t* __restrict__ input,
    int input_width,
    int input_height,
    float* __restrict__ output,
    int output_width,
    int output_height,
    float scale,
    int pad_x,
    int pad_y
) {
    // Calculate output coordinates
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;  // 0=R, 1=G, 2=B
    
    if (out_x >= output_width || out_y >= output_height || channel >= 3) {
        return;
    }
    
    // Calculate output index (CHW format)
    int out_idx = channel * output_height * output_width + out_y * output_width + out_x;
    
    // Check if this pixel is in the padded region
    if (out_x < pad_x || out_x >= (pad_x + input_width * scale) ||
        out_y < pad_y || out_y >= (pad_y + input_height * scale)) {
        // Padding region - set to 0
        output[out_idx] = 0.0f;
        return;
    }
    
    // Calculate corresponding input coordinates
    float in_x_f = (out_x - pad_x) / scale;
    float in_y_f = (out_y - pad_y) / scale;
    
    int in_x = static_cast<int>(in_x_f);
    int in_y = static_cast<int>(in_y_f);
    
    // Clamp to image boundaries
    in_x = min(max(in_x, 0), input_width - 1);
    in_y = min(max(in_y, 0), input_height - 1);
    
    // BGR to RGB conversion (channel mapping)
    // OpenCV uses BGR: input channel 0=B, 1=G, 2=R
    // We want RGB: output channel 0=R, 1=G, 2=B
    int bgr_channel = 2 - channel;  // Reverse channel order
    
    // Calculate input index (HWC format from OpenCV)
    int in_idx = in_y * input_width * 3 + in_x * 3 + bgr_channel;
    
    // Read, convert to float, and normalize to [0, 1]
    uint8_t pixel = input[in_idx];
    output[out_idx] = static_cast<float>(pixel) / 255.0f;
}

// Bilinear interpolation version for higher quality (optional)
__global__ void preprocessKernelBilinear(
    const uint8_t* __restrict__ input,
    int input_width,
    int input_height,
    float* __restrict__ output,
    int output_width,
    int output_height,
    float scale,
    int pad_x,
    int pad_y
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z;
    
    if (out_x >= output_width || out_y >= output_height || channel >= 3) {
        return;
    }
    
    int out_idx = channel * output_height * output_width + out_y * output_width + out_x;
    
    // Check if in padding region
    if (out_x < pad_x || out_x >= (pad_x + input_width * scale) ||
        out_y < pad_y || out_y >= (pad_y + input_height * scale)) {
        output[out_idx] = 0.0f;
        return;
    }
    
    // Calculate input coordinates
    float in_x_f = (out_x - pad_x) / scale;
    float in_y_f = (out_y - pad_y) / scale;
    
    int x0 = static_cast<int>(floorf(in_x_f));
    int y0 = static_cast<int>(floorf(in_y_f));
    int x1 = min(x0 + 1, input_width - 1);
    int y1 = min(y0 + 1, input_height - 1);
    
    x0 = max(x0, 0);
    y0 = max(y0, 0);
    
    float dx = in_x_f - x0;
    float dy = in_y_f - y0;
    
    int bgr_channel = 2 - channel;
    
    // Bilinear interpolation
    uint8_t v00 = input[y0 * input_width * 3 + x0 * 3 + bgr_channel];
    uint8_t v01 = input[y0 * input_width * 3 + x1 * 3 + bgr_channel];
    uint8_t v10 = input[y1 * input_width * 3 + x0 * 3 + bgr_channel];
    uint8_t v11 = input[y1 * input_width * 3 + x1 * 3 + bgr_channel];
    
    float v0 = v00 * (1 - dx) + v01 * dx;
    float v1 = v10 * (1 - dx) + v11 * dx;
    float value = v0 * (1 - dy) + v1 * dy;
    
    output[out_idx] = value / 255.0f;
}

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
) {
    // Use 16x16 thread blocks for 2D processing
    dim3 block(16, 16, 1);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        3  // Process all 3 channels
    );
    
    // Use bilinear interpolation for better quality
    preprocessKernelBilinear<<<grid, block, 0, stream>>>(
        input, input_width, input_height,
        output, output_width, output_height,
        scale, pad_x, pad_y
    );
}

// CudaPreprocessor implementation
CudaPreprocessor::CudaPreprocessor()
    : input_width_(0)
    , input_height_(0)
    , max_batch_size_(0)
    , d_input_buffer_(nullptr)
    , d_temp_buffer_(nullptr)
    , input_buffer_size_(0)
    , temp_buffer_size_(0)
    , initialized_(false)
{
}

CudaPreprocessor::~CudaPreprocessor() {
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
    }
    if (d_temp_buffer_) {
        cudaFree(d_temp_buffer_);
    }
}

bool CudaPreprocessor::initialize(int input_width, int input_height, int max_batch_size) {
    input_width_ = input_width;
    input_height_ = input_height;
    max_batch_size_ = max_batch_size;
    
    // Allocate buffers for max possible input size
    // Assume max input image size is 4K (3840x2160)
    const int max_img_width = 3840;
    const int max_img_height = 2160;
    
    input_buffer_size_ = max_batch_size * max_img_width * max_img_height * 3;
    temp_buffer_size_ = max_batch_size * input_width * input_height * 3 * sizeof(float);
    
    cudaError_t err;
    
    err = cudaMalloc(&d_input_buffer_, input_buffer_size_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate input buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_temp_buffer_, temp_buffer_size_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate temp buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input_buffer_);
        d_input_buffer_ = nullptr;
        return false;
    }
    
    initialized_ = true;
    std::cout << "CUDA Preprocessor initialized successfully" << std::endl;
    return true;
}

CudaPreprocessor::PreprocessParams CudaPreprocessor::preprocess(
    const cv::Mat& image, 
    float* output_device, 
    cudaStream_t stream
) {
    PreprocessParams params;
    params.orig_width = image.cols;
    params.orig_height = image.rows;
    
    // Calculate scaling and padding
    params.scale = std::min(
        static_cast<float>(input_width_) / image.cols,
        static_cast<float>(input_height_) / image.rows
    );
    
    int new_width = static_cast<int>(image.cols * params.scale);
    int new_height = static_cast<int>(image.rows * params.scale);
    
    params.pad_x = (input_width_ - new_width) / 2;
    params.pad_y = (input_height_ - new_height) / 2;
    
    // Copy image to device
    size_t image_size = image.rows * image.cols * 3;
    cudaMemcpyAsync(d_input_buffer_, image.data, image_size, 
                    cudaMemcpyHostToDevice, stream);
    
    // Launch preprocessing kernel
    launchPreprocessKernel(
        d_input_buffer_,
        image.cols, image.rows,
        output_device,
        input_width_, input_height_,
        params.scale,
        params.pad_x, params.pad_y,
        stream
    );
    
    return params;
}

std::vector<CudaPreprocessor::PreprocessParams> CudaPreprocessor::preprocessBatch(
    const std::vector<cv::Mat>& images,
    float* output_device,
    cudaStream_t stream
) {
    std::vector<PreprocessParams> params_vec;
    
    for (size_t i = 0; i < images.size(); ++i) {
        float* output_offset = output_device + i * input_width_ * input_height_ * 3;
        auto params = preprocess(images[i], output_offset, stream);
        params_vec.push_back(params);
    }
    
    return params_vec;
}

size_t CudaPreprocessor::getInputBufferSize() const {
    return input_buffer_size_;
}

size_t CudaPreprocessor::getOutputBufferSize() const {
    return max_batch_size_ * input_width_ * input_height_ * 3 * sizeof(float);
}

} // namespace cuda
} // namespace yolo26

#endif // USE_TENSORRT
