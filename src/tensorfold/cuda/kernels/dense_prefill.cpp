#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void dense_prefill_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, bool, int);

// Prompt rows x (M, K) bf16 times an unquantized weight w (N, K) bf16: one fp32 chain over K, row-invariant.
void dense_prefill(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, bool f32, int64_t tile) {
    TORCH_CHECK(tile >= 0 && tile <= 4, "tile 0-4");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    const int64_t k = x.size(1);
    TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kBFloat16 && w.dim() == 2 &&
                w.size(1) == k && w.size(0) >= 1, "w: (N, K) bf16, contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "w must start on a 16-byte boundary");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == x.size(0) && out.size(1) == w.size(0) &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, N)");
    TORCH_CHECK(x.get_device() == w.get_device() && w.get_device() == out.get_device(), "one device");
    c10::cuda::CUDAGuard guard(x.device());
    dense_prefill_cuda(x, w, out, f32, static_cast<int>(tile));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_prefill", &dense_prefill);
}
