// Prompt matmul on unquantized bf16 weights: x (M, K) times w (N, K)^T, one fp32 chain over K in k16 steps, so a
// row's bits never depend on M, on the other rows or on the tile shape.

#include <ATen/ATen.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

// Tile shapes: BM rows by BN columns a block, WM x WN warps, each warp (BM / WM) x (BN / WN); 64 inputs a stage.
template <int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int BK = 64;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp (even: B fragments load in pairs)
    static constexpr int ROW = BK * 2;                    // bytes of one row's stage slice
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                    // stage bytes: inputs,
    static constexpr int W = BN * ROW;                    // weight rows
    static constexpr int STAGE = X + W;
    static constexpr int SMEM = STAGES * STAGE;
    static_assert(NT % 2 == 0, "B fragments load two n8 tiles at a time");
};

template <int BM, int BN, int WM, int WN, int STAGES, bool F32>
__global__ void __launch_bounds__(WM * WN * 32) dense_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w, void* __restrict__ out, int M, int N,
        int K, int ldx, int group) {
    using T = Tile<BM, BN, WM, WN, STAGES>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int steps = K / T::BK;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    // 64 inputs of every row and weight row the block reads into stage s; rows past M or N read as zeros
    auto load = [&](int s, int kb) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16,
                  x + static_cast<size_t>(row) * ldx + kb * T::BK + ch * 8, m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < BN * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int col = min(n0 + r, N - 1);
            cp16z(pw + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16,
                  w + static_cast<size_t>(col) * K + kb * T::BK + ch * 8, n0 + r < N);
        }
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < steps) load(s, s);
        commit();
    }
    for (int it = 0; it < steps; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < steps) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const unsigned char* pw = p + T::X;
#pragma unroll
        for (int kt = 0; kt < T::BK / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
            uint32_t b[T::NT][2];
#pragma unroll
            for (int jj = 0; jj < T::NT / 2; ++jj) {
                // matrices (n 0-7, k 0-7), (n 0-7, k 8-15), (n 8-15, k 0-7), (n 8-15, k 8-15): two n8 tiles' b0, b1
                const int r = wn * (BN / WN) + jj * 16 + (lane & 7) + (lane >> 4) * 8;
                const int ch = kt * 2 + ((lane >> 3) & 1);
                uint32_t t[4];
                ldmatrix4(t, pw + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                b[2 * jj][0] = t[0];
                b[2 * jj][1] = t[1];
                b[2 * jj + 1][0] = t[2];
                b[2 * jj + 1][1] = t[3];
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b[j][0], b[j][1]);
        }
    }
    wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (col < N) dst[0] = v0;
                    if (col + 1 < N) dst[1] = v1;
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

template <int BM, int BN, int WM, int WN, int STAGES, bool F32>
void launch(const at::Tensor& x, const at::Tensor& w, at::Tensor& out) {
    using T = Tile<BM, BN, WM, WN, STAGES>;
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    auto kernel = dense_kernel<BM, BN, WM, WN, STAGES, F32>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
        configured = true;
    }
    const int rows_t = (M + BM - 1) / BM;
    // a band of row tiles' inputs stays near 12 MB of L2 while its blocks sweep the column tiles
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
    kernel<<<rows_t * ((N + BN - 1) / BN), T::THREADS, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        out.data_ptr(), M, N, K, M == 1 ? K : static_cast<int>(x.stride(0)), group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool F32>
void dispatch(int tile, const at::Tensor& x, const at::Tensor& w, at::Tensor& out) {
    switch (tile) {
        case 1: launch<64, 128, 1, 4, 3, F32>(x, w, out); break;
        case 2: launch<128, 64, 2, 2, 3, F32>(x, w, out); break;
        case 3: launch<16, 32, 1, 2, 4, F32>(x, w, out); break;
        case 4: launch<32, 64, 2, 2, 4, F32>(x, w, out); break;
        default: launch<128, 128, 2, 4, 3, F32>(x, w, out); break;
    }
}

} // namespace

// ``tile`` (0: 128x128, 1: 64x128, 2: 128x64 in three stages; 3: 16x32, 4: 32x64 in four, many blocks for a few
// rows) never changes a row's bits.
void dense_prefill_cuda(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, bool f32, int tile) {
    if (f32) dispatch<true>(tile, x, w, out); else dispatch<false>(tile, x, w, out);
}
