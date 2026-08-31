// Persian Plate Recognition - YOLO26 + CUDA/TensorRT inference application.
//
// C++ port of the original python main.py pipeline on top of the YOLO26
// TensorRT backend (full CUDA pre/post-processing, NMS-free end-to-end head):
//
//   stage 1: cars + plates detection -> weights/cars_best_v26  (data.yaml, 2 classes)
//   stage 2: plate char recognition  -> weights/chars_best_v26 (ir_lpr.yaml, 39 classes)
//
// Supports images, video files and webcams. Results are drawn on the frame,
// printed to the console and optionally saved under output/.

#include "plate_pipeline.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace {

const char* kWindowName = "Persian Plate Recognition (YOLO26 + TensorRT)";

struct Options {
    std::string source = "assets/video.mp4";
    std::string cars_model;      // empty -> auto-discover in weights/
    std::string chars_model;     // empty -> auto-discover in weights/
    float cars_conf = 0.70f;
    float chars_conf = 0.35f;
    int input_size = 640;
    int expected_chars = 8;
    bool show_partial = false;   // also draw plates with a partial read
    bool display = true;
    bool save = true;
    std::string output_dir = "output";
};

void printUsage(const char* argv0) {
    std::cout << "Persian Plate Recognition - YOLO26 + TensorRT\n\n"
              << "Usage: " << argv0 << " [options]\n\n"
              << "Options:\n"
              << "  --source PATH        Image/video file or webcam index\n"
              << "                       (default: assets/video.mp4)\n"
              << "  --cars-model PATH    Cars/plates model (.engine or .onnx)\n"
              << "                       (default: auto -> weights/cars_best_v26.engine|onnx)\n"
              << "  --chars-model PATH   Plate chars model (.engine or .onnx)\n"
              << "                       (default: auto -> weights/chars_best_v26.engine|onnx)\n"
              << "  --cars-conf FLOAT    Car/plate confidence threshold (default: 0.70)\n"
              << "  --chars-conf FLOAT   Char confidence threshold (default: 0.35)\n"
              << "  --input-size INT     Network input size, square letterbox (default: 640)\n"
              << "  --expected-chars N   Plate is reported complete with N chars (default: 8)\n"
              << "  --output-dir DIR     Directory for saved results (default: output)\n"
              << "  --show-partial       Also draw plates with a partial char read\n"
              << "  --no-save            Do not save annotated images/videos\n"
              << "  --no-display         Do not open the preview window (headless)\n"
              << "  --help               Show this help\n";
}

Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << std::endl;
                std::exit(-1);
            }
            return argv[++i];
        };

        if (arg == "--source") {
            opt.source = next("--source");
        } else if (arg == "--cars-model") {
            opt.cars_model = next("--cars-model");
        } else if (arg == "--chars-model") {
            opt.chars_model = next("--chars-model");
        } else if (arg == "--cars-conf") {
            opt.cars_conf = std::stof(next("--cars-conf"));
        } else if (arg == "--chars-conf") {
            opt.chars_conf = std::stof(next("--chars-conf"));
        } else if (arg == "--input-size") {
            opt.input_size = std::stoi(next("--input-size"));
        } else if (arg == "--expected-chars") {
            opt.expected_chars = std::stoi(next("--expected-chars"));
        } else if (arg == "--output-dir") {
            opt.output_dir = next("--output-dir");
        } else if (arg == "--show-partial") {
            opt.show_partial = true;
        } else if (arg == "--no-save") {
            opt.save = false;
        } else if (arg == "--no-display") {
            opt.display = false;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(-1);
        }
    }
    return opt;
}

// Prefer a pre-built TensorRT engine; fall back to ONNX (an engine is then
// built automatically on the first run and cached next to the ONNX file).
std::string resolveModelPath(const std::string& provided,
                             const std::string& base_name) {
    if (!provided.empty()) {
        return provided;
    }
    for (const char* ext : {".engine", ".onnx"}) {
        const std::string candidate = "weights/" + base_name + ext;
        std::ifstream file(candidate, std::ios::binary);
        if (file.good()) {
            return candidate;
        }
    }
    return "weights/" + base_name + ".onnx";
}

bool isWebcamIndex(const std::string& source) {
    return !source.empty() &&
           std::all_of(source.begin(), source.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
}

bool isImageFile(const std::string& path) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower.size() > 4 &&
           (lower.rfind(".jpg") == lower.size() - 4 ||
            lower.rfind(".jpeg") == lower.size() - 5 ||
            lower.rfind(".png") == lower.size() - 4 ||
            lower.rfind(".bmp") == lower.size() - 4);
}

void ensureDir(const std::string& path) { ::mkdir(path.c_str(), 0755); }

std::string timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return ss.str();
}

// Rolling-average FPS counter (over the last N frames)
class FpsCounter {
public:
    void update(double frame_ms) {
        times_.push_back(frame_ms);
        if (times_.size() > 30) {
            times_.pop_front();
        }
    }

    double fps() const {
        if (times_.empty()) {
            return 0.0;
        }
        const double avg = std::accumulate(times_.begin(), times_.end(), 0.0) /
                           static_cast<double>(times_.size());
        return avg > 0.0 ? 1000.0 / avg : 0.0;
    }

private:
    std::deque<double> times_;
};

void drawObjects(cv::Mat& frame, const std::vector<yolo26::Detection>& objects) {
    for (const auto& det : objects) {
        const bool is_plate = det.class_id == PersianPlateRecognizer::kPlateClassId;
        const cv::Scalar color = is_plate ? cv::Scalar(0, 140, 255)   // plates: orange
                                          : cv::Scalar(255, 0, 0);    // cars: blue
        cv::rectangle(frame, det.bbox, color, 2);

        std::ostringstream label;
        label << det.class_name << " " << std::fixed << std::setprecision(2)
              << det.confidence;
        cv::putText(frame, label.str(),
                    cv::Point(std::max(0, det.bbox.x), std::max(15, det.bbox.y - 5)),
                    cv::FONT_HERSHEY_TRIPLEX, 0.5, color, 1, cv::LINE_AA);
    }
}

void drawPlateResults(cv::Mat& frame, const std::vector<PlateResult>& plates,
                      bool only_complete) {
    for (const auto& plate : plates) {
        if (plate.text_latin.empty()) {
            continue;
        }
        if (only_complete && !plate.complete) {
            continue;
        }

        // Black banner above the plate (like the original python demo)
        const int y_line = std::max(10, plate.plate_box.y - 10);
        cv::line(frame,
                 cv::Point(std::max(0, plate.plate_box.x - 25), y_line),
                 cv::Point(std::min(frame.cols - 1, plate.plate_box.br().x + 25),
                           y_line),
                 cv::Scalar(0, 0, 0), 20, cv::LINE_AA);

        // cv::putText cannot render persian glyphs, so draw the latin
        // transliteration (the full persian string goes to the console).
        const std::string text = "Plate: " + plate.text_latin;
        cv::putText(frame, text,
                    cv::Point(std::max(0, plate.plate_box.x - 15),
                              std::max(12, plate.plate_box.y - 5)),
                    cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(10, 50, 255), 1,
                    cv::LINE_AA);
    }
}

void drawFps(cv::Mat& frame, double fps) {
    std::ostringstream text;
    text << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, text.str(), cv::Point(std::max(10, frame.cols - 220), 40),
                cv::FONT_HERSHEY_TRIPLEX, 1.0, cv::Scalar(10, 50, 255), 2,
                cv::LINE_AA);
}

int runImage(const Options& opt, const PersianPlateRecognizer& recognizer) {
    cv::Mat frame = cv::imread(opt.source);
    if (frame.empty()) {
        std::cerr << "Failed to load image: " << opt.source << std::endl;
        return -1;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    const FrameResult result = recognizer.process(frame);
    const double infer_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0)
                                .count();

    drawObjects(frame, result.objects);
    drawPlateResults(frame, result.plates, !opt.show_partial);
    drawFps(frame, infer_ms > 0.0 ? 1000.0 / infer_ms : 0.0);

    for (const auto& plate : result.plates) {
        std::cout << "Plate: " << plate.text_persian << "  ("
                  << plate.text_latin << ")  chars=" << plate.chars.size()
                  << (plate.complete ? "  [complete]" : "  [partial]")
                  << std::endl;
    }
    std::cout << "Inference: " << infer_ms << " ms, plates: "
              << result.plates.size() << std::endl;

    if (opt.save) {
        ensureDir(opt.output_dir);
        const std::string out_path =
            opt.output_dir + "/output_" + timestamp() + ".jpg";
        cv::imwrite(out_path, frame);
        std::cout << "Saved: " << out_path << std::endl;
    }

    if (opt.display) {
        cv::imshow(kWindowName, frame);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}

int runVideo(const Options& opt, const PersianPlateRecognizer& recognizer) {
    cv::VideoCapture cap;
    if (isWebcamIndex(opt.source)) {
        cap.open(std::stoi(opt.source));
    } else {
        cap.open(opt.source);
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open source: " << opt.source << std::endl;
        return -1;
    }

    const double source_fps = cap.get(cv::CAP_PROP_FPS);
    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer;
    std::string out_path;
    if (opt.save) {
        ensureDir(opt.output_dir);
        out_path = opt.output_dir + "/output_" + timestamp() + ".mp4";
        writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    source_fps > 0.0 ? source_fps : 25.0,
                    cv::Size(width, height));
        if (writer.isOpened()) {
            std::cout << "Recording to: " << out_path << std::endl;
        } else {
            std::cerr << "Failed to open video writer: " << out_path
                      << std::endl;
        }
    }

    FpsCounter fps_counter;
    std::string last_logged;
    long frame_idx = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            break;
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        const FrameResult result = recognizer.process(frame);
        const auto t1 = std::chrono::high_resolution_clock::now();
        fps_counter.update(
            std::chrono::duration<double, std::milli>(t1 - t0).count());

        drawObjects(frame, result.objects);
        drawPlateResults(frame, result.plates, !opt.show_partial);

        // Log newly recognized plates to the console
        for (const auto& plate : result.plates) {
            if (plate.complete && plate.text_persian != last_logged) {
                last_logged = plate.text_persian;
                std::cout << "[frame " << frame_idx << "] Plate: "
                          << plate.text_persian << "  (" << plate.text_latin
                          << ")" << std::endl;
            }
        }

        drawFps(frame, fps_counter.fps());

        if (writer.isOpened()) {
            writer.write(frame);
        }

        if (opt.display) {
            cv::imshow(kWindowName, frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                std::cout << "Quit requested." << std::endl;
                break;
            }
        }

        ++frame_idx;
    }

    if (writer.isOpened()) {
        writer.release();
        std::cout << "Video saved to: " << out_path << std::endl;
    }
    std::cout << "Frames processed: " << frame_idx << std::endl;

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);

    PersianPlateRecognizer::Params params;
    params.cars_model = resolveModelPath(opt.cars_model, "cars_best_v26");
    params.chars_model = resolveModelPath(opt.chars_model, "chars_best_v26");
    params.input_size = opt.input_size;
    params.cars_conf = opt.cars_conf;
    params.chars_conf = opt.chars_conf;
    params.expected_chars = opt.expected_chars;

    std::cout << "==============================================" << std::endl;
    std::cout << " Persian Plate Recognition - YOLO26 + TensorRT" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Source      : " << opt.source << std::endl;
    std::cout << "Cars model  : " << params.cars_model << std::endl;
    std::cout << "Chars model : " << params.chars_model << std::endl;

    PersianPlateRecognizer recognizer;
    if (!recognizer.initialize(params)) {
        std::cerr << "Failed to initialize the recognizer" << std::endl;
        return -1;
    }

    if (isImageFile(opt.source)) {
        return runImage(opt, recognizer);
    }
    return runVideo(opt, recognizer);
}



