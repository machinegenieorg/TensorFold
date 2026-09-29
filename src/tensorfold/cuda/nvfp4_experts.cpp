#include <torch/extension.h>

void nvfp4_experts_cuda(int64_t m, int64_t epi, int64_t sk, int64_t upb, int64_t d, int64_t rt,
                        const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                        const at::Tensor& s2, int64_t kg, int64_t nb, int64_t rg, const at::Tensor& items,
                        const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                        int64_t max_units, int64_t dense);

static void check(const at::Tensor& t, const char* name, at::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype, name, ": expected a CUDA tensor of the right dtype");
}

void run(int64_t m, int64_t epi, int64_t sk, int64_t upb, int64_t d, int64_t rt, const at::Tensor& x, int64_t slots,
         const at::Tensor& w, const at::Tensor& s2, int64_t kg, int64_t nb, int64_t rg, const at::Tensor& items,
         const at::Tensor& counts, const at::Tensor& members, at::Tensor out, int64_t n, int64_t max_units,
         int64_t dense) {
  check(x, "x", at::kBFloat16);
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0 && x.size(1) == kg * 64,
              "x: rows of K = groups * 64 inputs, 16-byte aligned");
  check(w, "w", at::kInt);
  TORCH_CHECK(w.is_contiguous() && w.numel() % (kg * nb * m * 288) == 0, "w: a contiguous packed table");
  check(s2, "s2", at::kFloat);
  TORCH_CHECK(s2.is_contiguous() && s2.numel() * kg * nb * 288 == w.numel(), "s2: one scale an expert a matrix");
  for (const auto* t : {&items, &counts, &members}) check(*t, "plan", at::kInt);
  TORCH_CHECK(out.is_contiguous() && out.size(-1) == n && n == nb * 32, "out: contiguous rows of n = 32 nb columns");
  TORCH_CHECK(out.scalar_type() == (epi == 0 ? at::kFloat : at::kBFloat16), "out has the wrong dtype");
  TORCH_CHECK(sk >= 1 && kg % sk == 0, "the K slices must split the groups evenly");
  TORCH_CHECK(dense == 0 || (dense <= x.size(0) && out.size(0) >= dense), "dense: rows of x and out");
  nvfp4_experts_cuda(m, epi, sk, upb, d, rt, x, x.stride(0), slots, w, s2, kg, nb, rg, items, counts, members, out,
                     n, max_units, dense);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "grouped NVFP4 expert matmul (epilogue 0: fp32, 2: SwiGLU, 3: bf16), or dense rows");
}
