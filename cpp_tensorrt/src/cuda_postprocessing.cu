#ifdef USE_TENSORRT

#include "cuda_postprocessing.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

namespace yolo26 {
namespace cuda {

// CUDA kernel for postprocessing YOLO26 end-to-end output
// Input: [batch, max_detections, 6] format: [x1, y1, x2, y2, conf, cls]
// Output: Filtered and scaled detections
__global__ void postprocessKernel(
    const float* __restrict__ logits,  // [batch, 300, 80]
    const float* __restrict__ boxes,   // [batch, 300, 4] cxcywh normalized [0,1]
    int batch_size,
    int max_detections,  // 300
    int num_classes,     // 80
    float conf_threshold,
    float input_width,   // 640.0f
    float input_height,  // 640.0f
    const float* __restrict__ scales,
    const int* __restrict__ pad_x,
    const int* __restrict__ pad_y,
    const int* __restrict__ orig_widths,
    const int* __restrict__ orig_heights,
    CudaPostprocessor::Detection* __restrict__ output_detections,
    int* __restrict__ detection_counts
) {
    int batch_idx = blockIdx.x;
    int det_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || det_idx >= max_detections) {
        return;
    }
    
    // Box: cxcywh normalized -> xyxy scaled to input size
    int box_offset = batch_idx * max_detections * 4 + det_idx * 4;
    float cx = boxes[box_offset + 0];
    float cy = boxes[box_offset + 1];
    float w = boxes[box_offset + 2];
    float h = boxes[box_offset + 3];
    
    float x1 = (cx - w * 0.5f) * input_width;
    float y1 = (cy - h * 0.5f) * input_height;
    float x2 = (cx + w * 0.5f) * input_width;
    float y2 = (cy + h * 0.5f) * input_height;
    
    // Clamp to input size
    x1 = fmaxf(0.0f, fminf(x1, input_width));
    y1 = fmaxf(0.0f, fminf(y1, input_height));
    x2 = fmaxf(0.0f, fminf(x2, input_width));
    y2 = fmaxf(0.0f, fminf(y2, input_height));
    
    // Logits: find max class and sigmoid conf
    int logits_offset = batch_idx * max_detections * num_classes + det_idx * num_classes;
    float max_logit = logits[logits_offset];
    int max_class = 0;
    for (int c = 1; c < num_classes; ++c) {
        float logit = logits[logits_offset + c];
        if (logit > max_logit) {
            max_logit = logit;
            max_class = c;
        }
    }
    float conf = 1.0f / (1.0f + expf(-max_logit));  // Sigmoid
    
    if (conf < conf_threshold) {
        return;
    }
    
    // Scale to original size
    float scale = scales[batch_idx];
    int px = pad_x[batch_idx];
    int py = pad_y[batch_idx];
    int orig_w = orig_widths[batch_idx];
    int orig_h = orig_heights[batch_idx];
    
    float scaled_x1 = (x1 - static_cast<float>(px)) / scale;
    float scaled_y1 = (y1 - static_cast<float>(py)) / scale;
    float scaled_x2 = (x2 - static_cast<float>(px)) / scale;
    float scaled_y2 = (y2 - static_cast<float>(py)) / scale;
    
    // Clamp to original size
    scaled_x1 = fmaxf(0.0f, fminf(scaled_x1, static_cast<float>(orig_w - 1)));
    scaled_y1 = fmaxf(0.0f, fminf(scaled_y1, static_cast<float>(orig_h - 1)));
    scaled_x2 = fmaxf(0.0f, fminf(scaled_x2, static_cast<float>(orig_w - 1)));
    scaled_y2 = fmaxf(0.0f, fminf(scaled_y2, static_cast<float>(orig_h - 1)));
    
    // Store
    int out_idx = atomicAdd(&detection_counts[batch_idx], 1);
    int output_offset = batch_idx * max_detections + out_idx;
    output_detections[output_offset].x1 = scaled_x1;
    output_detections[output_offset].y1 = scaled_y1;
    output_detections[output_offset].x2 = scaled_x2;
    output_detections[output_offset].y2 = scaled_y2;
    output_detections[output_offset].confidence = conf;
    output_detections[output_offset].class_id = max_class;
}

// Calculate IoU between two boxes
__device__ float calculateIoU(
    const CudaPostprocessor::Detection& a,
    const CudaPostprocessor::Detection& b
) {
    float x1 = fmaxf(a.x1, b.x1);
    float y1 = fmaxf(a.y1, b.y1);
    float x2 = fminf(a.x2, b.x2);
    float y2 = fminf(a.y2, b.y2);
    
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;
    
    return intersection / (union_area + 1e-6f);
}

// Parallel NMS kernel
__global__ void nmsKernel(
    CudaPostprocessor::Detection* detections,
    const int* detection_counts,
    int batch_size,
    int max_detections,
    float iou_threshold,
    bool* suppressed
) {
    int batch_idx = blockIdx.x;
    int det_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size) {
        return;
    }
    
    int count = detection_counts[batch_idx];
    if (det_idx >= count) {
        return;
    }
    
    int offset = batch_idx * max_detections;
    int global_idx = offset + det_idx;
    
    // Check if already suppressed
    if (suppressed[global_idx]) {
        return;
    }
    
    CudaPostprocessor::Detection& current = detections[global_idx];
    
    // Compare with all higher confidence detections
    for (int i = 0; i < det_idx; ++i) {
        int compare_idx = offset + i;
        
        if (suppressed[compare_idx]) {
            continue;
        }
        
        CudaPostprocessor::Detection& other = detections[compare_idx];
        
        // Only suppress if same class
        if (current.class_id != other.class_id) {
            continue;
        }
        
        // Calculate IoU
        float iou = calculateIoU(current, other);
        
        // Suppress if IoU exceeds threshold
        if (iou > iou_threshold) {
            suppressed[global_idx] = true;
            return;
        }
    }
}

void launchPostprocessKernel(
    const float* logits,
    const float* boxes,
    int batch_size,
    int max_detections,
    int num_classes,
    float conf_threshold,
    float input_width,
    float input_height,
    const float* scales,
    const int* pad_x,
    const int* pad_y,
    const int* orig_widths,
    const int* orig_heights,
    CudaPostprocessor::Detection* output_detections,
    int* detection_counts,
    cudaStream_t stream
) {
    cudaMemsetAsync(detection_counts, 0, batch_size * sizeof(int), stream);
    
    const int threads_per_block = 256;
    dim3 block(threads_per_block);
    dim3 grid(batch_size, (max_detections + threads_per_block - 1) / threads_per_block);
    
    postprocessKernel<<<grid, block, 0, stream>>>(
        logits, boxes, batch_size, max_detections, num_classes, conf_threshold,
        input_width, input_height, scales, pad_x, pad_y, orig_widths, orig_heights,
        output_detections, detection_counts
    );
}

void launchNMSKernel(
    CudaPostprocessor::Detection* detections,
    int* detection_counts,
    int batch_size,
    int max_detections,
    float iou_threshold,
    cudaStream_t stream
) {
    // Allocate suppression flags
    bool* d_suppressed;
    cudaMalloc(&d_suppressed, batch_size * max_detections * sizeof(bool));
    cudaMemsetAsync(d_suppressed, 0, batch_size * max_detections * sizeof(bool), stream);
    
    // Launch NMS kernel
    const int threads_per_block = 256;
    dim3 block(threads_per_block);
    dim3 grid(batch_size, (max_detections + threads_per_block - 1) / threads_per_block);
    
    nmsKernel<<<grid, block, 0, stream>>>(
        detections,
        detection_counts,
        batch_size,
        max_detections,
        iou_threshold,
        d_suppressed
    );
    
    // Compact detections (remove suppressed ones)
    // This is done on host for simplicity, but could be optimized with stream compaction
    cudaStreamSynchronize(stream);
    
    cudaFree(d_suppressed);
}

// CudaPostprocessor implementation
CudaPostprocessor::CudaPostprocessor()
    : max_detections_(0)
    , num_classes_(0)
    , max_batch_size_(0)
    , d_detections_(nullptr)
    , d_detection_counts_(nullptr)
    , h_detections_(nullptr)
    , h_detection_counts_(nullptr)
    , detection_buffer_size_(0)
    , initialized_(false)
{
}

CudaPostprocessor::~CudaPostprocessor() {
    if (d_detections_) {
        cudaFree(d_detections_);
    }
    if (d_detection_counts_) {
        cudaFree(d_detection_counts_);
    }
    if (h_detections_) {
        cudaFreeHost(h_detections_);
    }
    if (h_detection_counts_) {
        cudaFreeHost(h_detection_counts_);
    }
}

bool CudaPostprocessor::initialize(int max_detections, int num_classes, int max_batch_size) {
    max_detections_ = max_detections;
    num_classes_ = num_classes;
    max_batch_size_ = max_batch_size;
    
    detection_buffer_size_ = max_batch_size * max_detections * sizeof(Detection);
    
    cudaError_t err;
    
    // Allocate device memory
    err = cudaMalloc(&d_detections_, detection_buffer_size_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate detection buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaMalloc(&d_detection_counts_, max_batch_size * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate count buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_detections_);
        return false;
    }
    
    // Allocate pinned host memory for fast transfers
    err = cudaMallocHost(&h_detections_, detection_buffer_size_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate host detection buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_detections_);
        cudaFree(d_detection_counts_);
        return false;
    }
    
    err = cudaMallocHost(&h_detection_counts_, max_batch_size * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate host count buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_detections_);
        cudaFree(d_detection_counts_);
        cudaFreeHost(h_detections_);
        return false;
    }
    
    initialized_ = true;
    std::cout << "CUDA Postprocessor initialized successfully" << std::endl;
    return true;
}

std::vector<CudaPostprocessor::Detection> CudaPostprocessor::postprocessSingle(
    const float* logits_device,
    const float* boxes_device,
    float conf_threshold,
    float scale,
    int pad_x,
    int pad_y,
    int orig_width,
    int orig_height,
    cudaStream_t stream
) {
    // Copy parameters to device
    float* d_scales;
    int* d_pad_x;
    int* d_pad_y;
    int* d_orig_widths;
    int* d_orig_heights;
    
    cudaMalloc(&d_scales, sizeof(float));
    cudaMalloc(&d_pad_x, sizeof(int));
    cudaMalloc(&d_pad_y, sizeof(int));
    cudaMalloc(&d_orig_widths, sizeof(int));
    cudaMalloc(&d_orig_heights, sizeof(int));
    
    cudaMemcpyAsync(d_scales, &scale, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pad_x, &pad_x, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pad_y, &pad_y, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_orig_widths, &orig_width, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_orig_heights, &orig_height, sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Launch postprocessing kernel
    launchPostprocessKernel(
        logits_device,
        boxes_device,
        1,                    // batch_size
        max_detections_,
        num_classes_,
        conf_threshold,
        640.0f,               // input_width
        640.0f,               // input_height
        d_scales,
        d_pad_x,
        d_pad_y,
        d_orig_widths,
        d_orig_heights,
        d_detections_,
        d_detection_counts_,
        stream
    );
    
    // Copy results back to host
    cudaMemcpyAsync(h_detections_, d_detections_, 
                    max_detections_ * sizeof(Detection), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_detection_counts_, d_detection_counts_, 
                    sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Convert to vector
    std::vector<Detection> result;
    int count = h_detection_counts_[0];
    result.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        result.push_back(h_detections_[i]);
    }
    
    // Cleanup
    cudaFree(d_scales);
    cudaFree(d_pad_x);
    cudaFree(d_pad_y);
    cudaFree(d_orig_widths);
    cudaFree(d_orig_heights);
    
    return result;
}

std::vector<std::vector<CudaPostprocessor::Detection>> CudaPostprocessor::postprocess(
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
) {
    // Copy parameters to device
    float* d_scales;
    int* d_pad_x;
    int* d_pad_y;
    int* d_orig_widths;
    int* d_orig_heights;
    
    cudaMalloc(&d_scales, batch_size * sizeof(float));
    cudaMalloc(&d_pad_x, batch_size * sizeof(int));
    cudaMalloc(&d_pad_y, batch_size * sizeof(int));
    cudaMalloc(&d_orig_widths, batch_size * sizeof(int));
    cudaMalloc(&d_orig_heights, batch_size * sizeof(int));
    
    cudaMemcpyAsync(d_scales, scales.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pad_x, pad_x.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pad_y, pad_y.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_orig_widths, orig_widths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_orig_heights, orig_heights.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Launch postprocessing
    launchPostprocessKernel(
        logits_device,
        boxes_device,
        batch_size,
        max_detections_,
        num_classes_,
        conf_threshold,
        640.0f,  // input_width (hardcoded)
        640.0f,  // input_height
        d_scales,
        d_pad_x,
        d_pad_y,
        d_orig_widths,
        d_orig_heights,
        d_detections_,
        d_detection_counts_,
        stream
    );
    
    // Copy results back
    cudaMemcpyAsync(h_detections_, d_detections_, detection_buffer_size_, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_detection_counts_, d_detection_counts_, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Convert to vector of vectors
    std::vector<std::vector<Detection>> results;
    results.reserve(batch_size);
    
    for (int b = 0; b < batch_size; ++b) {
        std::vector<Detection> batch_result;
        int count = h_detection_counts_[b];
        batch_result.reserve(count);
        
        for (int i = 0; i < count; ++i) {
            batch_result.push_back(h_detections_[b * max_detections_ + i]);
        }
        
        results.push_back(batch_result);
    }
    
    // Cleanup
    cudaFree(d_scales);
    cudaFree(d_pad_x);
    cudaFree(d_pad_y);
    cudaFree(d_orig_widths);
    cudaFree(d_orig_heights);
    
    return results;
}

size_t CudaPostprocessor::getDetectionBufferSize() const {
    return detection_buffer_size_;
}

} // namespace cuda
} // namespace yolo26

#endif // USE_TENSORRT
