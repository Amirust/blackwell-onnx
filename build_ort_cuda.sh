#!/usr/bin/env bash
#
# Build ONNX Runtime with CUDA support for Blackwell (sm_120 / RTX 50 series).
# The build runs inside an NVIDIA CUDA "devel" container — only Docker is
# required on the host, and no GPU is needed to compile.
#
# Output (./onnxruntime-libs/):
#   libonnxruntime.so(.*)
#   libonnxruntime_providers_shared.so
#   libonnxruntime_providers_cuda.so
#
# Environment overrides: ORT_VERSION, CUDA_IMAGE, CUDA_ARCH, JOBS, OUT_DIR, WORK_DIR
#
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-v1.24.4}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"
CUDA_ARCH="${CUDA_ARCH:-120-real}"

if [ -z "${JOBS:-}" ]; then
    _cores="$(nproc)"
    _ram_gb="$(free -g | awk '/^Mem:/{print $2}')"
    JOBS=$(( _ram_gb / 6 ))
    [ "$JOBS" -lt 1 ] && JOBS=1
    [ "$JOBS" -gt "$_cores" ] && JOBS="$_cores"
fi

OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")" && pwd)/onnxruntime-libs}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "$0")" && pwd)/ort-build}"

echo ">> ONNX Runtime $ORT_VERSION | arch sm_${CUDA_ARCH%-real} | jobs $JOBS"
echo ">> CUDA build image: $CUDA_IMAGE"
echo ">> Work dir (host): $WORK_DIR"
echo ">> Output dir:      $OUT_DIR"

mkdir -p "$OUT_DIR" "$WORK_DIR"

docker run --rm \
    -e ORT_VERSION="$ORT_VERSION" \
    -e CUDA_ARCH="$CUDA_ARCH" \
    -e JOBS="$JOBS" \
    -v "$OUT_DIR:/out" \
    -v "$WORK_DIR:/work" \
    "$CUDA_IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends \
            git python3 python3-pip python3-dev build-essential \
            ninja-build ca-certificates curl unzip

        pip install --no-cache-dir --break-system-packages cmake psutil
        CMAKE_BIN="$(python3 -c "import cmake,os;print(os.path.join(cmake.CMAKE_BIN_DIR,\"cmake\"))")"
        echo ">> using cmake: $CMAKE_BIN"; "$CMAKE_BIN" --version

        cd /work
        if [ ! -d onnxruntime/.git ]; then
            git clone --recursive --depth 1 --branch "$ORT_VERSION" \
                https://github.com/microsoft/onnxruntime.git
        fi
        cd onnxruntime

        CUTLASS_URL="$(grep -E "^cutlass;" cmake/deps.txt | cut -d";" -f2)"
        echo ">> prefetching cutlass: $CUTLASS_URL"
        mkdir -p /work/deps
        curl -fL --retry 3 -o /work/deps/cutlass.zip "$CUTLASS_URL"
        ( cd /work/deps && rm -rf cutlass-* && unzip -q cutlass.zip )
        CUTLASS_DIR="$(find /work/deps -maxdepth 1 -type d -name "cutlass-*" | head -1)"
        echo ">> cutlass extracted to: $CUTLASS_DIR"

        python3 tools/ci_build/build.py \
            --build_dir build \
            --config Release \
            --update --build \
            --cmake_path "$CMAKE_BIN" \
            --parallel "$JOBS" \
            --build_shared_lib \
            --use_cuda \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cuda \
            --cmake_extra_defines \
                CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
                onnxruntime_USE_FLASH_ATTENTION=OFF \
                onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
                FETCHCONTENT_SOURCE_DIR_CUTLASS="$CUTLASS_DIR" \
            --skip_tests \
            --allow_running_as_root

        echo ">> copying artifacts to /out"
        cp -av build/Release/libonnxruntime.so*                 /out/ 2>/dev/null || true
        cp -av build/Release/libonnxruntime_providers_shared.so /out/ 2>/dev/null || true
        cp -av build/Release/libonnxruntime_providers_cuda.so   /out/ 2>/dev/null || true
        ls -la /out
    '

echo ">> Done. Libraries are in: $OUT_DIR"
