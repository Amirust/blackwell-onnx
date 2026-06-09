# ONNX Runtime (CUDA, Blackwell / sm_120)

Prebuilt ONNX Runtime shared libraries with the CUDA execution provider, compiled
for **Blackwell GPUs (sm_120 / RTX 50 series)**.

```
libonnxruntime.so(.*)               core library — point ORT_DYLIB_PATH here
libonnxruntime_providers_shared.so
libonnxruntime_providers_cuda.so    CUDA execution provider
```

## Usage

Load via the `ort` crate's `load-dynamic` feature and set the path:

```bash
export ORT_DYLIB_PATH=/path/to/onnx/libonnxruntime.so
```

## Building

Built from [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) with
[`./build_ort_cuda.sh`](./build_ort_cuda.sh). The build runs inside an NVIDIA CUDA
container, so only Docker is required on the host (no GPU needed to compile).

```bash
./build_ort_cuda.sh
```

## Special Thanks
I want to thank Huang, NVIDIA's CEO, for scamming me into buying a graphics card with 'AI' that can't even run AI properly
<img src="https://github.com/Amirust/blackwell-onnx/blob/master/message_to_nvidia.jpg?raw=true" alt="nvidia i dont like you" />
