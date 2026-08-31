#!/bin/bash

# Build script for the Persian Plate Recognition C++ / CUDA / TensorRT pipeline

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Persian Plate Recognition - C++/TensorRT${NC}"
echo -e "${GREEN}========================================${NC}"

BUILD_TYPE=Release
TENSORRT_ROOT="${TENSORRT_ROOT:-}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tensorrt-root)
            TENSORRT_ROOT="$2"
            shift 2
            ;;
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --clean)
            rm -rf build
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--tensorrt-root PATH] [--debug] [--clean]"
            echo ""
            echo "Options:"
            echo "  --tensorrt-root PATH  Path to the TensorRT installation"
            echo "                        (or set the TENSORRT_ROOT env var)"
            echo "  --debug               Build in Debug mode (default: Release)"
            echo "  --clean               Remove the build directory first"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}CMake not found. Please install CMake 3.15 or higher.${NC}"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}CUDA compiler (nvcc) not found. Please install the CUDA toolkit.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Build Configuration:${NC}"
echo "  Build Type : $BUILD_TYPE"
if [ -n "$TENSORRT_ROOT" ]; then
    echo "  TensorRT   : $TENSORRT_ROOT"
else
    echo "  TensorRT   : (auto-detect: /opt/tensorrt*, /usr/local/tensorrt*, system paths)"
fi
echo ""

# Create build directory
mkdir -p build
cd build

# Build CMake command
CMAKE_CMD="cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE"
if [ -n "$TENSORRT_ROOT" ]; then
    CMAKE_CMD="$CMAKE_CMD -DTENSORRT_ROOT_DIR=$TENSORRT_ROOT"
fi

echo -e "${YELLOW}Running CMake...${NC}"
echo "$CMAKE_CMD"
eval $CMAKE_CMD

echo ""
echo -e "${YELLOW}Building project...${NC}"
make -j$(nproc)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Application: $(pwd)/persian_plate_recognizer"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Export the YOLO26 weights to ONNX (one-time):"
echo "     python3 ../scripts/export_onnx.py"
echo ""
echo "2. Run the recognizer (run from the repo root so the default"
echo "   weights/ and assets/ paths resolve):"
echo "     cd .."
echo "     ./cpp_tensorrt/build/persian_plate_recognizer --source assets/car.jpg"
echo "   (the first run builds and caches the .engine files next to the ONNX models)"
