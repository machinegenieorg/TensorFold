// SPIKE: gdn_tree.cu's tree and replay for several requests in one launch (blockIdx.z picks the request).
// Every request keeps its own rows, parents and state; the arithmetic per node is gdn_tree.cu's, element for
// element (built with the same --fmad=false), so each request's outputs are bit-identical to its own launch.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_down_sync(0xffffffff, x, offset);
    return __shfl_sync(0xffffffff, x, 0);
}

// One thread per request: the depth-first preorder of its tree (item-local rows), as gdn_tree.cu's
// preorder_kernel. A chain is its own preorder at depth 0 (it uses one state slot).
__global__ void preorder_multi_kernel(const int* parents, const int* offsets, const int* chain,
                                      int* order, int* depths, int items) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= items) return;
    const int r0 = offsets[item], nodes = offsets[item + 1] - r0;
    const int* p = parents + r0;
    int* o = order + r0;
    int* dep = depths + r0;
    if (chain[item]) {
        for (int i = 0; i < nodes; ++i) { o[i] = i; dep[i] = 0; }
        return;
    }
    int at_depth[32], next_child[32];
    int depth = 0, emitted = 1;
    at_depth[0] = 0;
    next_child[0] = 1;
    o[0] = 0;
    dep[0] = 0;
    while (depth >= 0 && emitted < nodes) {
        const int parent = at_depth[depth];
        int child = -1;
        for (int i = next_child[depth]; i < nodes; ++i) {
            if (p[i] == parent) { child = i; next_child[depth] = i + 1; break; }
        }
        if (child < 0) {
            --depth;
        } else {
            ++depth;
            at_depth[depth] = child;
            next_child[depth] = child + 1;
            o[emitted++] = child;
            dep[child] = depth;
        }
    }
}

// tree_kernel<CHAIN or DFS> per request. A node's state is its parent's times decay plus its update, whatever
// the visiting order, so the DFS walk gives the same bits as the index-order walk for small trees.
__global__ void tree_multi_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                                  const float* g, const float* beta, const long long* states,
                                  const int* parents, const int* offsets, const int* chain,
                                  const int* order, const int* depths, __nv_bfloat16* y, int hk, int hv, int dv,
                                  const int* final_idx, float* finals) {
    const int value = blockIdx.x, head = blockIdx.y, item = blockIdx.z, lane = threadIdx.x;
    const int r0 = offsets[item], nodes = offsets[item + 1] - r0;
    const bool is_chain = chain[item] != 0;
    const float* state0 = reinterpret_cast<const float*>(states[item]);
    const int key_head = head / (hv / hk);
    const int state_base = (head * dv + value) * 128;
    float initial[4], slots[32][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) initial[i] = state0[state_base + lane * 4 + i];
    // Node j + 1's inputs are loaded while node j computes (the walk is latency-bound); the arithmetic is
    // unchanged. A chain keeps its state in registers.
    float nq[4], nk[4], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    int nnode = 0;
    auto fetch = [&](int step_index) {
        nnode = is_chain ? step_index : order[r0 + step_index];
        const int row = r0 + nnode;
        const int key_base = (row * hk + key_head) * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            nq[i] = __bfloat162float(q[key_base + i]);
            nk[i] = __bfloat162float(k[key_base + i]);
        }
        nv = __bfloat162float(v[(row * hv + head) * dv + value]);
        ng = g[row * hv + head];
        nb = beta[row * hv + head];
    };
    if (nodes > 0) fetch(0);
    float cur[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cur[i] = initial[i];
    for (int step_index = 0; step_index < nodes; ++step_index) {
        const int node = nnode;
        float qi[4], ki[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { qi[i] = nq[i]; ki[i] = nk[i]; }
        const float vv = nv, decay = ng, step = nb;
        if (step_index + 1 < nodes) fetch(step_index + 1);
        const int value_base = ((r0 + node) * hv + head) * dv + value;
        float s[4], mem = 0.0f;
        if (is_chain) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] = cur[i] * decay;
                mem += s[i] * ki[i];
            }
        } else {
            const int parent = parents[r0 + node];
            const int source = depths[r0 + node] - 1;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] = (parent < 0 ? initial[i] : slots[source][i]) * decay;
                mem += s[i] * ki[i];
            }
        }
        const float delta = (vv - warp_sum(mem)) * step;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] += ki[i] * delta;
            out += s[i] * qi[i];
        }
        if (is_chain) {
#pragma unroll
            for (int i = 0; i < 4; ++i) cur[i] = s[i];
        } else {
            const int destination = depths[r0 + node];
#pragma unroll
            for (int i = 0; i < 4; ++i) slots[destination][i] = s[i];
        }
        out = warp_sum(out);
        if (lane == 0) y[value_base] = __float2bfloat16_rn(out);
    }
    // a chain whose rows are all committed (a prompt chunk): its last state is replay's result, same arithmetic
    const int fi = final_idx[item];
    if (is_chain && fi >= 0 && nodes > 0) {
        float* dst = finals + static_cast<long long>(fi) * hv * dv * 128;
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[state_base + lane * 4 + i] = cur[i];
    }
}

// replay_many_kernel over (layer, request) pairs: table rows k, v, g, beta, state pointers and the request's
// index into rows/counts, one column per pair.
__global__ void replay_multi_kernel(const long long* table, int pairs, const int* rows, const int* counts,
                                   float* state_out, int hk, int hv, int dv) {
    const int value = blockIdx.x, head = blockIdx.y, pair = blockIdx.z, lane = threadIdx.x;
    const auto* k = reinterpret_cast<const __nv_bfloat16*>(table[0 * pairs + pair]);
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(table[1 * pairs + pair]);
    const auto* g = reinterpret_cast<const float*>(table[2 * pairs + pair]);
    const auto* beta = reinterpret_cast<const float*>(table[3 * pairs + pair]);
    const auto* state0 = reinterpret_cast<const float*>(table[4 * pairs + pair]);
    const int job = static_cast<int>(table[5 * pairs + pair]);
    const int* path = rows + job * 128;
    float* out = state_out + static_cast<long long>(pair) * hv * dv * 128;
    const int key_head = head / (hv / hk);
    const int state_base = (head * dv + value) * 128;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = state0[state_base + lane * 4 + i];
    const int count = counts[job];
    float nk[4], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int j) {
        const int node = path[j];
        const int key_base = (node * hk + key_head) * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) nk[i] = __bfloat162float(k[key_base + i]);
        nv = __bfloat162float(v[(node * hv + head) * dv + value]);
        ng = g[node * hv + head];
        nb = beta[node * hv + head];
    };
    if (count > 0) fetch(0);
    for (int j = 0; j < count; ++j) {
        float ki[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) ki[i] = nk[i];
        const float vv = nv, decay = ng, bb = nb;
        if (j + 1 < count) fetch(j + 1);
        float mem = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] *= decay;
            mem += s[i] * ki[i];
        }
        const float delta = (vv - warp_sum(mem)) * bb;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] += ki[i] * delta;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) out[state_base + lane * 4 + i] = s[i];
}

} // namespace

void gdn_preorder_multi_cuda(const at::Tensor& parents, const at::Tensor& offsets, const at::Tensor& chain,
                             at::Tensor& order, at::Tensor& depths, int items) {
    auto stream = at::cuda::getCurrentCUDAStream();
    preorder_multi_kernel<<<(items + 31) / 32, 32, 0, stream>>>(parents.data_ptr<int>(), offsets.data_ptr<int>(),
                                                                chain.data_ptr<int>(), order.data_ptr<int>(),
                                                                depths.data_ptr<int>(), items);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_tree_multi_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                         const at::Tensor& beta, const at::Tensor& states, const at::Tensor& parents,
                         const at::Tensor& offsets, const at::Tensor& chain, const at::Tensor& order,
                         const at::Tensor& depths, at::Tensor& out, int items, const at::Tensor& final_idx,
                         at::Tensor& finals) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(v.size(2), v.size(1), items);
    tree_multi_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        g.data_ptr<float>(), beta.data_ptr<float>(), reinterpret_cast<const long long*>(states.data_ptr<int64_t>()),
        parents.data_ptr<int>(), offsets.data_ptr<int>(), chain.data_ptr<int>(), order.data_ptr<int>(),
        depths.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        static_cast<int>(q.size(1)), static_cast<int>(v.size(1)), static_cast<int>(v.size(2)),
        final_idx.data_ptr<int>(), finals.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_replay_multi_cuda(const at::Tensor& table, int pairs, const at::Tensor& rows, const at::Tensor& counts,
                           at::Tensor& out, int hk, int hv, int dv) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(dv, hv, pairs);
    replay_multi_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const long long*>(table.data_ptr<int64_t>()), pairs, rows.data_ptr<int>(),
        counts.data_ptr<int>(), out.data_ptr<float>(), hk, hv, dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
