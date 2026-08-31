#pragma once

#ifdef USE_TENSORRT

#include "yolo26_base.h"
#include "cuda_preprocessing.h"
#include "cuda_postprocessing.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <iterator> // for std::begin/std::end
#include <vector>
namespace yolo26 {

// Logger for TensorRT
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YOLO26TensorRT : public YOLO26Base {
public:
    YOLO26TensorRT();
    ~YOLO26TensorRT() override;
    
    bool initialize(const ModelConfig& config) override;
    std::vector<Detection> detect(const cv::Mat& image) override;
    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& images) override;
    
    std::string getBackendName() const override { return "TensorRT"; }
    bool isInitialized() const override { return initialized_; }
    
    // Build TensorRT engine from ONNX model
    bool buildEngineFromOnnx(const std::string& onnx_path, const std::string& engine_path);
    
private:
    TensorRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // CUDA stream
    cudaStream_t stream_;
    
    // CUDA preprocessing and postprocessing
    std::unique_ptr<cuda::CudaPreprocessor> cuda_preprocessor_;
    std::unique_ptr<cuda::CudaPostprocessor> cuda_postprocessor_;
    
    // Device buffers
    std::vector<void*> device_buffers_;
    
    // Input/Output information
    int input_index_;
    int output_index_;
    int logits_index_;
    int boxes_index_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    nvinfer1::Dims logits_dims_;
    nvinfer1::Dims boxes_dims_;
    size_t logits_size_;
    size_t boxes_size_; 
    size_t input_size_;
    size_t output_size_;

    // Fused one-to-one output mode: single output [batch, max_det, 6]
    // ([x1, y1, x2, y2, confidence, class_id]) instead of logits+boxes
    bool fused_output_ = false;
    int fused_index_ = -1;
    nvinfer1::Dims fused_dims_{};
    size_t fused_size_ = 0;
    
    // Load engine from file
    bool loadEngine(const std::string& engine_path);
    
    // Serialize engine to file
    bool serializeEngine(const std::string& engine_path);
    
    // Allocate device buffers
    void allocateBuffers();
    
    // Free device buffers
    void freeBuffers();
    
    // Run inference
    std::vector<float> runInference(const std::vector<cv::Mat>& preprocessed_images);
};

// CUDA deleter for smart pointers
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

// TensorRT deleter for smart pointers
template<typename T>
struct TensorRTDeleter {
    void operator()(T* obj) const {
        if (obj) {
            delete obj;
        }
    }
};

} // namespace yolo26

#endif // USE_TENSORRT
