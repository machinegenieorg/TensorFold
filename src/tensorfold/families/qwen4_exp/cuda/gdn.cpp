#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void gdn_chain_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, double, int64_t, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, int64_t, bool);
void gdn_replay_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     int64_t, at::Tensor&);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == t, name, ": expected a contiguous CUDA tensor");
}

static bool supported(int64_t nk, int64_t nv) { return (nk == 16 && (nv == 48 || nv == 32)) || (nk == 8 && nv == 24); }

// gate 0: sigmoid(z) (Flash Next), 1: silu(z) (Qwen3.6). An empty xs skips the 32-group sums.

void chain(const at::Tensor& P, const at::Tensor& cs, const at::Tensor& cw, const at::Tensor& state_in,
           const at::Tensor& a_log, const at::Tensor& dt_bias, const at::Tensor& norm_w, double eps, int64_t rows,
           at::Tensor out, at::Tensor xs, at::Tensor state_out, at::Tensor k_save, at::Tensor v_save,
           at::Tensor g_save, at::Tensor b_save, int64_t gate) {
    check(P, at::kBFloat16, "P");
    check(cs, at::kBFloat16, "conv state");
    check(cw, at::kBFloat16, "conv weight");
    check(state_in, at::kFloat, "state");
    check(a_log, at::kFloat, "A_log");
    check(dt_bias, at::kFloat, "dt_bias");
    check(norm_w, at::kBFloat16, "norm");
    check(out, at::kBFloat16, "out");
    if (xs.numel()) check(xs, at::kFloat, "group sums");
    TORCH_CHECK(gate == 0 || gate == 1, "gate: 0 (sigmoid) or 1 (silu)");
    TORCH_CHECK(P.dim() == 2, "P must be [rows, width]");
    // P row: q | k (nk * 128 each) | v | z (nv * 128 each) | b | a (nv each)
    const int64_t nv = a_log.numel(), nk = (P.size(1) - 2 * nv * 128 - 2 * nv) / 256;
    TORCH_CHECK(supported(nk, nv), "(key, value) heads (16, 48), (8, 24) or (16, 32)");
    TORCH_CHECK(P.size(0) >= rows && P.size(1) == 2 * nk * 128 + 2 * nv * 128 + 2 * nv, "P width");
    TORCH_CHECK(!k_save.numel() || (k_save.dim() == 3 && k_save.size(0) >= rows && k_save.size(1) == nk),
                "k_save must be [>= rows, key heads, 128]");
    TORCH_CHECK(state_in.numel() == nv * 128 * 128, "state must be [nv, 128, 128]");
    TORCH_CHECK(out.size(0) >= rows && out.size(1) == nv * 128, "out must be [>= rows, nv * 128]");
    TORCH_CHECK(!xs.numel() || (xs.size(0) >= rows && xs.size(1) == nv * 128 / 32), "group sums: [>= rows, nv * 4]");
    c10::cuda::CUDAGuard guard(P.device());
    gdn_chain_cuda(P, cs, cw, state_in, a_log, dt_bias, norm_w, eps, rows, out, xs, state_out, k_save, v_save,
                   g_save, b_save, nk, gate == 1);
}

void replay(const at::Tensor& state_in, const at::Tensor& k_save, const at::Tensor& v_save, const at::Tensor& g_save,
            const at::Tensor& b_save, int64_t rows, at::Tensor state_out) {
    check(state_in, at::kFloat, "state");
    check(state_out, at::kFloat, "state out");
    TORCH_CHECK(k_save.dim() == 3 && g_save.dim() == 2 && supported(k_save.size(1), g_save.size(1)),
                "(key, value) heads (16, 48), (8, 24) or (16, 32)");
    TORCH_CHECK(rows <= k_save.size(0), "replay: more rows than the window saved");
    c10::cuda::CUDAGuard guard(state_in.device());
    gdn_replay_cuda(state_in, k_save, v_save, g_save, b_save, rows, state_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chain", &chain);
    m.def("replay", &replay);
}
