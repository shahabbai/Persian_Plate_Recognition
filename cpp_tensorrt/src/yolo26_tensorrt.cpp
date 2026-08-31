#ifdef USE_TENSORRT

#include "yolo26_tensorrt.h"
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>  // Added for std::exp
#include <algorithm>  // Added for std::min, std::max

namespace yolo26 {

YOLO26TensorRT::YOLO26TensorRT() : stream_(nullptr) {
    cudaStreamCreate(&stream_);
    cuda_preprocessor_ = std::make_unique<cuda::CudaPreprocessor>();
    cuda_postprocessor_ = std::make_unique<cuda::CudaPostprocessor>();
}

YOLO26TensorRT::~YOLO26TensorRT() {
    freeBuffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

bool YOLO26TensorRT::initialize(const ModelConfig& config) {
    try {
        config_ = config;
        
        // Check if engine file exists
        std::string engine_path = config_.model_path;
        if (engine_path.substr(engine_path.find_last_of(".") + 1) == "onnx") {
            // Build engine from ONNX
            engine_path = engine_path.substr(0, engine_path.find_last_of(".")) + ".engine";
            if (!buildEngineFromOnnx(config_.model_path, engine_path)) {
                return false;
            }
        }
        
        // Load engine
        if (!loadEngine(engine_path)) {
            return false;
        }
        
        // Allocate buffers
        allocateBuffers();
        
        
        if (config_.use_cuda_prepost) {
        // Initialize CUDA preprocessor
        if (!cuda_preprocessor_->initialize(config_.input_width, config_.input_height, 8)) {
        std::cerr << "Failed to initialize CUDA preprocessor" << std::endl;
        return false;
        }
        // Initialize CUDA postprocessor
        if (!cuda_postprocessor_->initialize(config_.max_detections, config_.num_classes, 8)) {
        std::cerr << "Failed to initialize CUDA postprocessor" << std::endl;
        return false;
        }
    }
        
        initialized_ = true;
        std::cout << "YOLO26 TensorRT initialized successfully" << std::endl;
        std::cout << "Backend: " << getBackendName() << std::endl;
        std::cout << "CUDA Acceleration: Enabled for preprocessing and postprocessing" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing TensorRT: " << e.what() << std::endl;
        return false;
    }
}

bool YOLO26TensorRT::buildEngineFromOnnx(const std::string& onnx_path, const std::string& engine_path) {
    std::cout << "Building TensorRT engine from ONNX model..." << std::endl;
    
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_)
    );
    
    const auto explicit_batch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
    );
    
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicit_batch)
    );
    
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_)
    );
    
    // Parse ONNX model
    if (!parser->parseFromFile(onnx_path.c_str(), 
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX model" << std::endl;
        return false;
    }
    
    // Build engine configuration
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig()
    );
    
    // Set max workspace size (e.g., 1GB)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    
    // Enable FP16 if supported
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled" << std::endl;
    }
    
    // Optimization profile.
    // If the ONNX input has fully static dimensions (e.g. the export fixed the
    // batch size to 1), the profile dimensions must match the ONNX input
    // exactly, otherwise buildSerializedNetwork fails with an API usage
    // error. A dynamic 1..8 batch profile is used only for dynamic inputs.
    const char* input_name = network->getInput(0)->getName();
    const nvinfer1::Dims onnx_input_dims = network->getInput(0)->getDimensions();
    bool static_input = true;
    for (int i = 0; i < onnx_input_dims.nbDims; ++i) {
        if (onnx_input_dims.d[i] < 0) {
            static_input = false;
            break;
        }
    }
    if (static_input && onnx_input_dims.nbDims == 4) {
        // Keep the runtime input size consistent with the static engine input
        config_.input_width = onnx_input_dims.d[3];
        config_.input_height = onnx_input_dims.d[2];
        std::cout << "Static ONNX input ["
                  << onnx_input_dims.d[0] << ", " << onnx_input_dims.d[1]
                  << ", " << config_.input_height << ", " << config_.input_width
                  << "] - using a static optimization profile" << std::endl;
    }
    auto profile = builder->createOptimizationProfile();
    
    if (static_input) {
        // Static input: the profile must match the ONNX dimensions exactly
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN,
                               onnx_input_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT,
                               onnx_input_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX,
                               onnx_input_dims);
    } else {
        // Dynamic input: allow a variable batch size from 1 to 8
        profile->setDimensions(
            input_name,
            nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, 3, config_.input_height, config_.input_width});
        profile->setDimensions(
            input_name,
            nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{1, 3, config_.input_height, config_.input_width});
        profile->setDimensions(
            input_name,
            nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{8, 3, config_.input_height, config_.input_width});
    }
    
    config->addOptimizationProfile(profile);
    
    // Build engine
    std::cout << "Building engine... This may take a while." << std::endl;
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config)
    );
    
    if (!serialized_engine) {
        std::cerr << "Failed to build engine" << std::endl;
        return false;
    }
    
    // Save engine to file
    std::ofstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open engine file for writing" << std::endl;
        return false;
    }
    
    engine_file.write(
        static_cast<const char*>(serialized_engine->data()),
        serialized_engine->size()
    );
    engine_file.close();
    
    std::cout << "Engine built and saved to: " << engine_path << std::endl;
    return true;
}

bool YOLO26TensorRT::loadEngine(const std::string& engine_path) {
    std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;
    
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open engine file" << std::endl;
        return false;
    }
    
    engine_file.seekg(0, std::ios::end);
    size_t file_size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    
    std::vector<char> engine_data(file_size);
    engine_file.read(engine_data.data(), file_size);
    engine_file.close();
    
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), file_size));
    
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    // Get input/output information
    int num_bindings = engine_->getNbIOTensors();
    int output_count = 0;
    for (int i = 0; i < num_bindings; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);
        
        std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output") 
                  << " " << i << ": " << name << " - Shape: [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_index_ = i;
            input_dims_ = dims;
        } else {
            if (output_count == 0) {
                logits_index_ = i;
                logits_dims_ = dims;
            } else if (output_count == 1) {
                boxes_index_ = i;
                boxes_dims_ = dims;
            }
            output_count++;
        }
    }
    
    if (output_count == 1) {
        // Single fused one-to-one output: [batch, max_det, 6] =
        // [x1, y1, x2, y2, confidence, class_id] (current ultralytics
        // end2end export format)
        fused_output_ = true;
        fused_index_ = logits_index_;  // set above for the first (only) output
        fused_dims_ = logits_dims_;
        if (fused_dims_.nbDims < 2 || fused_dims_.d[fused_dims_.nbDims - 1] != 6) {
            std::cerr << "Unsupported single-output engine: expected a fused "
                      << "one-to-one output [batch, 300, 6]" << std::endl;
            return false;
        }
        std::cout << "Fused one-to-one output detected: [batch, "
                  << fused_dims_.d[1] << ", 6]" << std::endl;
    } else if (output_count == 2) {
        // Split layout: separate logits [batch, 300, nc] and boxes
        // [batch, 300, 4] outputs
        fused_output_ = false;
    } else {
        std::cerr << "Expected 2 outputs (logits and boxes) or 1 fused output "
                  << "[batch, 300, 6], found " << output_count << std::endl;
        return false;
    }
    
    return true;
}

void YOLO26TensorRT::allocateBuffers() {
    // Calculate sizes (assuming batch=1 for now; extend for dynamic)
    input_size_ = 1;
    for (int i = 0; i < input_dims_.nbDims; ++i) {
        input_size_ *= std::abs(input_dims_.d[i]);  // Handle dynamic (-1)
    }
    
    if (fused_output_) {
        fused_size_ = 1;
        for (int i = 0; i < fused_dims_.nbDims; ++i) {
            fused_size_ *= std::abs(fused_dims_.d[i]);
        }

        // Allocate device memory (2 buffers: input, fused output)
        device_buffers_.resize(2);
        cudaMalloc(&device_buffers_[0], input_size_ * sizeof(float));  // input
        cudaMalloc(&device_buffers_[1], fused_size_ * sizeof(float));  // fused output

        std::cout << "Buffers allocated - Input: " << input_size_ * sizeof(float)
                  << " bytes, Fused output: " << fused_size_ * sizeof(float)
                  << " bytes" << std::endl;
        return;
    }

    logits_size_ = 1;
    for (int i = 0; i < logits_dims_.nbDims; ++i) {
        logits_size_ *= std::abs(logits_dims_.d[i]);
    }
    
    boxes_size_ = 1;
    for (int i = 0; i < boxes_dims_.nbDims; ++i) {
        boxes_size_ *= std::abs(boxes_dims_.d[i]);
    }
    
    // Allocate device memory (3 buffers: input, logits, boxes)
    device_buffers_.resize(3);
    cudaMalloc(&device_buffers_[0], input_size_ * sizeof(float));  // input
    cudaMalloc(&device_buffers_[1], logits_size_ * sizeof(float)); // logits
    cudaMalloc(&device_buffers_[2], boxes_size_ * sizeof(float));  // boxes
    
    std::cout << "Buffers allocated - Input: " << input_size_ * sizeof(float) 
              << " bytes, Logits: " << logits_size_ * sizeof(float) 
              << " bytes, Boxes: " << boxes_size_ * sizeof(float) << " bytes" << std::endl;
}

void YOLO26TensorRT::freeBuffers() {
    for (auto& buffer : device_buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    device_buffers_.clear();
}

std::vector<Detection> YOLO26TensorRT::detect(const cv::Mat& image) {
    if (!initialized_) {
        std::cerr << "Model not initialized" << std::endl;
        return {};
    }
    
    // With CUDA preprocessing, we pass the original image directly
    // No need for CPU preprocessing
    cv::Size original_size = image.size();
    void* input_device = device_buffers_[0];
    
    cuda::CudaPreprocessor::PreprocessParams preprocess_params;
    if (config_.use_cuda_prepost) {  // Use CUDA if initialized
        preprocess_params = cuda_preprocessor_->preprocess(image, static_cast<float*>(input_device), stream_);
    } else {  // Use CPU
    original_size = image.size();
    cv::Mat preprocessed = preprocessImage(image, original_size);
    
    // Convert HWC to CHW format for TensorRT
    std::vector<float> chw_data(config_.input_width * config_.input_height * 3);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < config_.input_height; ++h) {
            for (int w = 0; w < config_.input_width; ++w) {
                size_t dst_idx = c * config_.input_height * config_.input_width + 
                               h * config_.input_width + w;
                chw_data[dst_idx] = preprocessed.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
    
    cudaMemcpyAsync(input_device, chw_data.data(), 
                    chw_data.size() * sizeof(float), 
                    cudaMemcpyHostToDevice, stream_);
    
    preprocess_params.scale = std::min(static_cast<float>(config_.input_width) / original_size.width, 
                                       static_cast<float>(config_.input_height) / original_size.height);
    preprocess_params.pad_x = (config_.input_width - static_cast<int>(original_size.width * preprocess_params.scale)) / 2;
    preprocess_params.pad_y = (config_.input_height - static_cast<int>(original_size.height * preprocess_params.scale)) / 2;
    preprocess_params.orig_width = original_size.width;
    preprocess_params.orig_height = original_size.height;
    }
    
    // Set input shape
    nvinfer1::Dims4 input_dims{1, 3, config_.input_height, config_.input_width};
    context_->setInputShape(engine_->getIOTensorName(input_index_), input_dims);
    
    // Set tensor addresses
    context_->setTensorAddress(engine_->getIOTensorName(input_index_), device_buffers_[0]);
    if (fused_output_) {
        context_->setTensorAddress(engine_->getIOTensorName(fused_index_), device_buffers_[1]);
    } else {
        context_->setTensorAddress(engine_->getIOTensorName(logits_index_), device_buffers_[1]);
        context_->setTensorAddress(engine_->getIOTensorName(boxes_index_), device_buffers_[2]);
    }
    
    // Execute inference
    if (!context_->enqueueV3(stream_)) {
        std::cerr << "Failed to enqueue inference" << std::endl;
        return {};
    }

    if (fused_output_) {
        // Fused one-to-one output: [batch, max_det, 6] with
        // [x1, y1, x2, y2, confidence, class_id] in input-scale pixels.
        // Copy to host, filter by confidence and reuse the base end-to-end
        // postprocessing (letterbox-aware coordinate mapping).
        std::vector<float> output(fused_size_);
        cudaMemcpyAsync(output.data(), device_buffers_[1],
                        fused_size_ * sizeof(float), cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);

        const int num_dets = fused_dims_.nbDims >= 2 ? fused_dims_.d[1] : 0;
        const size_t stride = 6;
        std::vector<float> combined;
        combined.reserve(static_cast<size_t>(std::max(num_dets, 0)) * stride);
        for (int i = 0; i < num_dets; ++i) {
            const float* det = &output[static_cast<size_t>(i) * stride];
            if (det[4] < config_.conf_threshold) {
                continue;
            }
            combined.insert(combined.end(), det, det + stride);
        }

        return postprocessDetections(
            combined.data(), combined.size(), original_size,
            cv::Size(config_.input_width, config_.input_height));
    }
    
    // CUDA postprocessing
    std::vector<Detection> detections;
    if (config_.use_cuda_prepost) {  // Use CUDA if initialized
        auto cuda_detections = cuda_postprocessor_->postprocessSingle(
            static_cast<const float*>(device_buffers_[1]),  // logits
            static_cast<const float*>(device_buffers_[2]),  // boxes
            config_.conf_threshold,
            preprocess_params.scale,
            preprocess_params.pad_x,
            preprocess_params.pad_y,
            preprocess_params.orig_width,
            preprocess_params.orig_height,
            stream_
        );
        
        // Convert CUDA detections to standard Detection format
        detections.reserve(cuda_detections.size());
        
        for (const auto& cuda_det : cuda_detections) {
            Detection det;
            det.bbox = cv::Rect(
                static_cast<int>(cuda_det.x1),
                static_cast<int>(cuda_det.y1),
                static_cast<int>(cuda_det.x2 - cuda_det.x1),
                static_cast<int>(cuda_det.y2 - cuda_det.y1)
            );
            det.confidence = cuda_det.confidence;
            det.class_id = cuda_det.class_id;
            
            if (det.class_id >= 0 && det.class_id < config_.class_names.size()) {
                det.class_name = config_.class_names[det.class_id];
            } else {
                det.class_name = "Unknown";
            }
            
            detections.push_back(det);
        }
    } else {
        // Use base postprocess: copy outputs to host, then postprocessDetections
        std::vector<float> logits(logits_size_);
        std::vector<float> boxes(boxes_size_);
        cudaMemcpyAsync(logits.data(), device_buffers_[1], logits_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(boxes.data(), device_buffers_[2], boxes_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        // Combine into flat output for postprocessDetections (using max class per proposal)
        std::vector<float> combined;

        int batch_size = 1;  // detect() uses single image
        int num_detections = logits_dims_.d[1];   // usually 300
        int num_classes    = logits_dims_.d[2];   // usually 80

        float input_w = static_cast<float>(config_.input_width);
        float input_h = static_cast<float>(config_.input_height);

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < num_detections; ++i) {

                // ---- boxes (cx,cy,w,h normalized) ----
                int box_offset = b * num_detections * 4 + i * 4;

                float cx = boxes[box_offset + 0];
                float cy = boxes[box_offset + 1];
                float w  = boxes[box_offset + 2];
                float h  = boxes[box_offset + 3];

                // convert to xyxy in input resolution
                float x1 = (cx - w * 0.5f) * input_w;
                float y1 = (cy - h * 0.5f) * input_h;
                float x2 = (cx + w * 0.5f) * input_w;
                float y2 = (cy + h * 0.5f) * input_h;

                // Clamp boxes
                x1 = std::max(0.0f, std::min(x1, input_w));
                y1 = std::max(0.0f, std::min(y1, input_h));
                x2 = std::max(0.0f, std::min(x2, input_w));
                y2 = std::max(0.0f, std::min(y2, input_h));

                // ---- logits: find max class and conf ----
                int logit_offset = b * num_detections * num_classes + i * num_classes;

                float max_logit = logits[logit_offset];
                int max_class = 0;
                for (int c = 1; c < num_classes; ++c) {
                    float logit = logits[logit_offset + c];
                    if (logit > max_logit) {
                        max_logit = logit;
                        max_class = c;
                    }
                }
                float conf = 1.f / (1.f + std::exp(-max_logit));  // sigmoid

                if (conf < config_.conf_threshold)
                    continue;

                combined.push_back(x1);
                combined.push_back(y1);
                combined.push_back(x2);
                combined.push_back(y2);
                combined.push_back(conf);
                combined.push_back(static_cast<float>(max_class));
            }
        }

        detections = postprocessDetections(combined.data(), combined.size(), original_size, cv::Size(config_.input_width, config_.input_height));

    }
    
    return detections;
}

std::vector<std::vector<Detection>> YOLO26TensorRT::detectBatch(const std::vector<cv::Mat>& images) {
    if (!initialized_) {
        std::cerr << "Model not initialized" << std::endl;
        return {};
    }
    
    std::vector<std::vector<Detection>> all_detections;
    
    // For simplicity, process images one by one
    for (const auto& image : images) {
        all_detections.push_back(detect(image));
    }
    
    return all_detections;
}

std::vector<float> YOLO26TensorRT::runInference(const std::vector<cv::Mat>& preprocessed_images) {
    try {
        size_t batch_size = preprocessed_images.size();
        
        // CUDA preprocessing - directly to device memory
        std::vector<cuda::CudaPreprocessor::PreprocessParams> preprocess_params = 
            cuda_preprocessor_->preprocessBatch(preprocessed_images, 
                                               static_cast<float*>(device_buffers_[0]), 
                                               stream_);
        
        // Set input shape for dynamic batch
        nvinfer1::Dims4 input_dims{
            static_cast<int>(batch_size),
            3,
            config_.input_height,
            config_.input_width
        };
        
        context_->setInputShape(engine_->getIOTensorName(input_index_), input_dims);
        
        // Set tensor addresses
        context_->setTensorAddress(engine_->getIOTensorName(input_index_), device_buffers_[0]);
        context_->setTensorAddress(engine_->getIOTensorName(logits_index_), device_buffers_[1]);
        context_->setTensorAddress(engine_->getIOTensorName(boxes_index_), device_buffers_[2]);
        
        // Execute inference
        if (!context_->enqueueV3(stream_)) {
            std::cerr << "Failed to enqueue inference" << std::endl;
            return {};
        }
        
        // Prepare parameters
        std::vector<float> scales;
        std::vector<int> pad_x, pad_y, orig_widths, orig_heights;
        
        for (const auto& params : preprocess_params) {
            scales.push_back(params.scale);
            pad_x.push_back(params.pad_x);
            pad_y.push_back(params.pad_y);
            orig_widths.push_back(params.orig_width);
            orig_heights.push_back(params.orig_height);
        }
        
        // CUDA postprocessing
        auto cuda_detections = cuda_postprocessor_->postprocess(
            static_cast<const float*>(device_buffers_[1]),  // logits
            static_cast<const float*>(device_buffers_[2]),  // boxes
            batch_size,
            config_.conf_threshold,
            scales,
            pad_x,
            pad_y,
            orig_widths,
            orig_heights,
            stream_
        );
        
        // For compatibility, return raw output (but detections are processed)
        auto output_dims = context_->getTensorShape(engine_->getIOTensorName(logits_index_));  // or boxes, doesn't matter
        size_t output_size = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_size *= output_dims.d[i];
        }
        
        std::vector<float> output_data(output_size);
        cudaMemcpyAsync(output_data.data(), device_buffers_[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        
        return output_data;
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return {};
    }
}
} // namespace yolo26

#endif // USE_TENSORRT