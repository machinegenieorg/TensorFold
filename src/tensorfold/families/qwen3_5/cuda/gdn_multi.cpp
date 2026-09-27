#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void gdn_preorder_multi_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, int);
void gdn_tree_multi_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                         const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int);
void gdn_replay_multi_cuda(const at::Tensor&, int, const at::Tensor&, const at::Tensor&, at::Tensor&, int, int, int);

static void check_int(const at::Tensor& t, const at::Tensor& like, const char* what) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt && t.device() == like.device(), what);
}

// q, k (R, hk, 128) bf16; v (R, hv, dv) bf16; g, beta (R, hv) fp32: every request's rows back to back.
// states: (items,) int64 device pointers to each request's (hv, dv, 128) fp32 state; parents (R,) item-local;
// offsets (items + 1,) row starts; chain (items,) 1 if that request's rows are a chain.
at::Tensor tree_multi(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                      const at::Tensor& beta, const at::Tensor& states, const at::Tensor& parents,
                      const at::Tensor& offsets, const at::Tensor& chain) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() && beta.is_contiguous(),
                "GDN inputs must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16 &&
                g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat, "q, k, v bf16; g, beta fp32");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && k.sizes() == q.sizes() && v.size(0) == q.size(0) &&
                g.size(0) == q.size(0) && v.size(1) == g.size(1) && v.size(1) % q.size(1) == 0, "invalid GDN shapes");
    TORCH_CHECK(states.is_cuda() && states.is_contiguous() && states.scalar_type() == at::kLong, "states: int64 pointers");
    check_int(parents, q, "parents: int32");
    check_int(offsets, q, "offsets: int32");
    check_int(chain, q, "chain: int32");
    const int items = static_cast<int>(chain.numel());
    TORCH_CHECK(states.numel() == items && offsets.numel() == items + 1 && parents.numel() == q.size(0), "one entry per item");
    c10::cuda::CUDAGuard guard(q.device());
    auto order = at::empty_like(parents);
    auto depths = at::empty_like(parents);
    gdn_preorder_multi_cuda(parents, offsets, chain, order, depths, items);
    auto out = at::empty({q.size(0), v.size(1), v.size(2)}, q.options());
    gdn_tree_multi_cuda(q, k, v, g, beta, states, parents, offsets, chain, order, depths, out, items);
    return out;
}

// table (6, pairs) int64 on the device: k, v, g, beta, state pointers and the request index of each
// (layer, request) pair; rows (requests, 128) int32 accepted rows; counts (requests,) int32.
at::Tensor replay_multi(const at::Tensor& table, const at::Tensor& rows, const at::Tensor& counts,
                        int64_t hk, int64_t hv, int64_t dv) {
    TORCH_CHECK(table.is_cuda() && table.is_contiguous() && table.scalar_type() == at::kLong && table.dim() == 2 &&
                table.size(0) == 6, "table: (6, pairs) int64");
    check_int(rows, table, "rows: int32");
    check_int(counts, table, "counts: int32");
    TORCH_CHECK(rows.dim() == 2 && rows.size(1) == 128 && counts.numel() == rows.size(0), "rows (requests, 128)");
    c10::cuda::CUDAGuard guard(table.device());
    const int pairs = static_cast<int>(table.size(1));
    auto out = at::empty({pairs, hv, dv, 128}, table.options().dtype(at::kFloat));
    gdn_replay_multi_cuda(table, pairs, rows, counts, out, static_cast<int>(hk), static_cast<int>(hv),
                          static_cast<int>(dv));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tree_multi", &tree_multi);
    m.def("replay_multi", &replay_multi);
}
