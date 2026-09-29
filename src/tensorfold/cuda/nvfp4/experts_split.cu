// NVFP4 grouped experts with K split in slices: the MLX experts kernel's decode form (experts.cu) on NVIDIA's FP4
// format, and its prompt form (experts_prefill.cu: K in one slice, an item's rows and weights staged in shared
// memory). A proposed alternative to experts.cu: measured on an RTX PRO 6000 Max-Q and a GB10 with Qwen3.6's
// experts, see experts_split.py.
//
// A table keeps the checkpoint's E2M1 nibbles in the MLX experts' fragment order (experts.pack; the two formats pack
// nibbles alike) and, where an MLX group keeps its scales and biases, the 64-input group's e4m3 block scales: one
// word a lane, byte j the scale of the lane's B-fragment column in n8 tile j for its 16-input block (lane t of a quad
// holds inputs 16 t .. 16 t + 15 of the group, one NVFP4 block). A lane decodes its fragment to exact bf16 values,
// 2 x code x scale (at most six significant bits), so each (pair, column) is one fp32 mma chain over K: K is split
// into slices fixed by K, the slices summed in slice order, times the expert's per-tensor scale over two. A pair is
// one mma row, so no pair affects another's bits, whatever the rows of the call.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <torch/extension.h>

#include "../experts.cuh"

namespace {

constexpr int GS = 64;                     // inputs a group: four NVFP4 blocks
using G = Geo<GS>;                         // WV = 2 weight uint4 a lane a group, XV = 2 input uint4, BLOCK = 72
constexpr int SCALES = 32 * G::WV;         // a block's scale words follow its weights (uint4 offset)
constexpr uint32_t E2M1_LO = 0x03020100u;  // twice the E2M1 magnitudes by code: 0, 1, 2, 3 | 4, 6, 8, 12
constexpr uint32_t E2M1_HI = 0x0C080604u;

__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ld_s(const uint32_t* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];\n" : "=r"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ld_x(const __nv_bfloat16* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// the codes at bits sh .. sh + 3 and sh + 16 .. sh + 19 -> the exact bf16 pair (2 v0, 2 v1): the magnitude from a
// byte table ((128 + 2 v) - 128), the sign bit moved to bf16's
__device__ __forceinline__ uint32_t fp4x2(uint32_t w, int sh) {
  const uint32_t m = (w >> sh) & 0x00070007u;
  uint32_t v = __byte_perm(E2M1_LO, E2M1_HI, m | (m >> 8)) | K128;
  const uint32_t k = K128;
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<const __nv_bfloat162*>(&k));
  return *reinterpret_cast<uint32_t*>(&r) | ((w << (12 - sh)) & 0x80008000u);
}

// exact: 2 x code has at most two significant bits, an e4m3 scale four
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ float e4m3f(uint32_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

// byte j of a scale word as the bf16 pair (s, s): every e4m3 value is a bf16 value
__device__ __forceinline__ uint32_t scale_pair(uint32_t word, int j) {
  const __nv_bfloat16 s = __float2bfloat16_rn(e4m3f((word >> (8 * j)) & 0xFFu));
  const __nv_bfloat162 p = __halves2bfloat162(s, s);
  return *reinterpret_cast<const uint32_t*>(&p);
}

// A warp's rows: RT row tiles of 16, each lane rows gq and gq + 8 of every tile.
template <int RT>
struct Rows {
  const __nv_bfloat16* x0[RT];
  const __nv_bfloat16* x1[RT];
  bool v0[RT], v1[RT];
};

template <int M, int RT>
struct Stage {
  uint4 w[M][G::WV];
  uint32_t s[M];
  uint4 xa[RT][G::XV];  // row gq of each tile: the lane's 16 inputs of the group
  uint4 xb[RT][G::XV];  // row gq + 8
};

template <int M, int RT>
__device__ __forceinline__ void load_stage(Stage<M, RT>& st, const uint4* blk, int g, int lane, const Rows<RT>& rw) {
  const uint4* b = blk + (size_t)g * (M * G::BLOCK);
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int c = 0; c < G::WV; ++c) st.w[m][c] = ld_w(b + m * G::BLOCK + c * 32 + lane);
    st.s[m] = ld_s(reinterpret_cast<const uint32_t*>(b + m * G::BLOCK + SCALES) + lane);
  }
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  const int k0 = g * GS;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int c = 0; c < G::XV; ++c) {
      st.xa[r][c] = rw.v0[r] ? ld_x(rw.x0[r] + k0 + 8 * c) : zero;
      st.xb[r][c] = rw.v1[r] ? ld_x(rw.x1[r] + k0 + 8 * c) : zero;
    }
}

template <int M, int RT>
__device__ __forceinline__ void compute_stage(float (&acc)[M][RT][NTW][4], const Stage<M, RT>& st) {
#pragma unroll
  for (int m = 0; m < M; ++m) {
    uint32_t sp[NTW];
#pragma unroll
    for (int j = 0; j < NTW; ++j) sp[j] = scale_pair(st.s[m], j);
#pragma unroll
    for (int ks = 0; ks < G::KS; ++ks) {
      // k-step ks: the lane's inputs 4 ks .. 4 ks + 3 of its block (pairs at the mma's k positions 2t and 2t + 8)
      const int sh = (ks & 1) ? 8 : 0;
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const int wi = j * (GS / 32) + (ks >> 1);
        const uint32_t word = comp(st.w[m][wi >> 2], wi & 3);
        const uint32_t b0 = bmul(fp4x2(word, sh), sp[j]), b1 = bmul(fp4x2(word, sh + 4), sp[j]);
#pragma unroll
        for (int r = 0; r < RT; ++r)                   // one decoded fragment, every row tile's chain
          mma(acc[m][r][j], comp(st.xa[r][ks >> 1], 2 * (ks & 1)), comp(st.xb[r][ks >> 1], 2 * (ks & 1)),
              comp(st.xa[r][ks >> 1], 2 * (ks & 1) + 1), comp(st.xb[r][ks >> 1], 2 * (ks & 1) + 1), b0, b1);
      }
    }
  }
}

template <int M, int RT, int D>
__device__ __forceinline__ void k_loop(float (&acc)[M][RT][NTW][4], const uint4* blk, int KG, int lane,
                                       const Rows<RT>& rw) {
  Stage<M, RT> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M, RT>(st[d], blk, d, lane, rw);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M, RT>(acc, st[d]);
        if (g + D < KG) load_stage<M, RT>(st[d], blk, g + D, lane, rw);
      }
    }
  }
}

// EPI 0: fp32 out (decode down); 2: bf16 SwiGLU(matrix 0, matrix 1); 3: bf16 out (prompt down). Each sum times its
// matrix's scale first (the per-tensor scale over two), one fp32 rounding.
template <int M, int EPI, int RT>
__device__ __forceinline__ void store(const float (&acc)[M][RT][NTW][4], int r, const float* s2, void* out, int N,
                                      int col0, int pr0, int pr1, bool v0, bool v1) {
  float sc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) sc[m] = __ldg(s2 + m);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][r][j][2 * h] * sc[0], a1 = acc[0][r][j][2 * h + 1] * sc[0];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0 = a0, o1 = a1;
        if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[M - 1][r][j][2 * h] * sc[M - 1], 0.f);
          o1 = swiglu(a1, acc[M - 1][r][j][2 * h + 1] * sc[M - 1], 0.f);
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

// A unit is (item, RT row tiles of 16, 32 output columns); SK warps take its K slices and warp 0 adds them in order.
// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p (X holds a row a pair).
template <int M, int EPI, int SK, int UPB, int D, int RT>
__global__ void __launch_bounds__(UPB * SK * 32)
    fp4_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
               const float* __restrict__ S2, int KG, int NB, int RG, const int* __restrict__ items,
               const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N) {
  constexpr int ACC = M * RT * NTW * 4;
  __shared__ float red[SK > 1 ? UPB * (SK - 1) * ACC * 32 : 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, ul = warp / SK, sl = warp - ul * SK;
  const int gq = lane >> 2, t = lane & 3;
  const int per_item = NB * RG, per = KG / SK;
  const int units = __ldg(counts) * per_item;
  for (int u0 = blockIdx.x * UPB; u0 < units; u0 += gridDim.x * UPB) {
    const int unit = u0 + ul;
    float acc[M][RT][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
    bool live = false;
    int e = 0, cb = 0, first = 0, cnt = 0, row0 = 0;
    Rows<RT> rw;
    int pr0[RT], pr1[RT];
    if (unit < units) {
      const int it = unit / per_item, rest = unit - it * per_item, rg = rest / NB;
      cb = rest - rg * NB;
      e = __ldg(items + 3 * it);
      first = __ldg(items + 3 * it + 1);
      cnt = __ldg(items + 3 * it + 2);
      row0 = 16 * RT * rg;
      live = row0 < cnt;
    }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int base = row0 + 16 * r;
      rw.v0[r] = live && base + gq < cnt;
      rw.v1[r] = live && base + gq + 8 < cnt;
      pr0[r] = rw.v0[r] ? __ldg(members + first + base + gq) : 0;
      pr1[r] = rw.v1[r] ? __ldg(members + first + base + gq + 8) : 0;
      const int r0 = slots ? pr0[r] / slots : pr0[r], r1 = slots ? pr1[r] / slots : pr1[r];
      const size_t k0 = (size_t)sl * per * GS + t * (GS / 4);
      rw.x0[r] = X + (size_t)r0 * x_stride + k0;
      rw.x1[r] = X + (size_t)r1 * x_stride + k0;
    }
    if (live) {
      const uint4* blk = W + (((size_t)e * NB + cb) * KG + (size_t)sl * per) * (M * G::BLOCK);
      k_loop<M, RT, D>(acc, blk, per, lane, rw);
    }
    if constexpr (SK > 1) {
      if (sl > 0 && live) {
        float* mine = red + (size_t)(ul * (SK - 1) + sl - 1) * ACC * 32 + lane;
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int j = 0; j < NTW; ++j)
#pragma unroll
              for (int c = 0; c < 4; ++c) mine[(((m * RT + r) * NTW + j) * 4 + c) * 32] = acc[m][r][j][c];
      }
      __syncthreads();
      if (sl == 0 && live) {
        for (int s = 1; s < SK; ++s) {
          const float* theirs = red + (size_t)(ul * (SK - 1) + s - 1) * ACC * 32 + lane;
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
              for (int j = 0; j < NTW; ++j)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][r][j][c] += theirs[(((m * RT + r) * NTW + j) * 4 + c) * 32];
        }
      }
      __syncthreads();
    }
    if (sl == 0 && live) {
#pragma unroll
      for (int r = 0; r < RT; ++r)
        store<M, EPI, RT>(acc, r, S2 + (size_t)e * M, out, N, cb * COLS + 2 * t, pr0[r], pr1[r], rw.v0[r],
                          rw.v1[r]);
    }
  }
}

// The prompt form: a CTA takes an item's pairs (WM warps down, RT row tiles each) against WN column blocks, every
// 64-input group of their rows and weights brought to shared memory once (cp.async, STAGES deep) and read by all its
// warps: the MLX experts' prefill kernel (experts_prefill.cu) on NVIDIA's FP4. Each (pair, column) is one fp32 mma
// chain over K in order, the decode form's fragments in one slice. On a GB10 it takes a 4,096-row chunk's experts in
// 382 ms where one warp a unit (four row tiles, no staging) took 465; on an RTX PRO 6000 Max-Q, 74 ms against 71.
template <int M, int RT, int WM, int WN>
struct Staged {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, XC = GS / 8, WB = M * G::BLOCK;
  static constexpr int XU = BM * XC, SU = XU + WN * WB;          // uint4 a stage: the rows' group, the weight blocks
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGE_BYTES = SU * 16;
  static constexpr int STAGES = STAGE_BYTES * 4 <= 49152 ? 4 : STAGE_BYTES * 3 <= 49152 ? 3 : 2;
  static __device__ __forceinline__ int xslot(int r, int c) { return r * XC + (c ^ (r & 1)); }
};

template <int M, int EPI, int RT, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32)
    fp4_prefill_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                       const float* __restrict__ S2, int KG, int NB, const int* __restrict__ items,
                       const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out,
                       int N) {
  using P = Staged<M, RT, WM, WN>;
  constexpr int STAGES = P::STAGES;
  extern __shared__ uint4 sm[];
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const __nv_bfloat16* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = P::xslot(r, c);
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WN) * (size_t)KG * P::WB;
  auto stage = [&](int s, int g) {
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i]) cp16(xs + xdst[i], xsrc[i] + (size_t)g * GS);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::WB; q += P::THREADS) {
      const int j = q / P::WB, o = q - j * P::WB;
      cp16(ws + q, wsrc + ((size_t)j * KG + g) * P::WB + o);
    }
  };
  const int base = 16 * RT * wm;
  const int nt = max(0, min(RT, (cnt - base + 15) >> 4));
  const bool active = nt > 0 && cb < NB;
  float acc[M][RT][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KG) stage(s, s);
    cp_commit();
  }
  for (int g = 0; g < KG; ++g) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (g + STAGES - 1 < KG) stage((g + STAGES - 1) % STAGES, g + STAGES - 1);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (g % STAGES) * P::SU;
    const uint4* ws = xs + P::XU + wn * P::WB;
    uint4 wv[M][G::WV];
    uint32_t sp[M][NTW];
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int c = 0; c < G::WV; ++c) wv[m][c] = ws[m * G::BLOCK + c * 32 + lane];
      const uint32_t sw = reinterpret_cast<const uint32_t*>(ws + m * G::BLOCK + SCALES)[lane];
#pragma unroll
      for (int j = 0; j < NTW; ++j) sp[m][j] = scale_pair(sw, j);
    }
#pragma unroll
    for (int kk = 0; kk < G::XV; ++kk) {
      uint4 xa[RT], xb[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (r >= nt) break;
        const int r0 = base + 16 * r + gq;
        xa[r] = xs[P::xslot(r0, t * G::XV + kk)];
        xb[r] = xs[P::xslot(r0 + 8, t * G::XV + kk)];
      }
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          const int wi = j * (GS / 32) + kk;
          const uint32_t word = comp(wv[m][wi >> 2], wi & 3);
#pragma unroll
          for (int h = 0; h < 2; ++h) {        // k-step 2 kk + h, the one-warp form's order
            const uint32_t b0 = bmul(fp4x2(word, 8 * h), sp[m][j]), b1 = bmul(fp4x2(word, 8 * h + 4), sp[m][j]);
#pragma unroll
            for (int r = 0; r < RT; ++r) {
              if (r >= nt) break;
              mma(acc[m][r][j], comp(xa[r], 2 * h), comp(xb[r], 2 * h), comp(xa[r], 2 * h + 1),
                  comp(xb[r], 2 * h + 1), b0, b1);
            }
          }
        }
    }
  }
  if (!active) return;
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    if (r >= nt) break;
    const int m0 = base + 16 * r + gq, m1 = m0 + 8;
    const bool v0 = m0 < cnt, v1 = m1 < cnt;
    const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
    store<M, EPI, RT>(acc, r, S2 + (size_t)e * M, out, N, cb * COLS + 2 * t, p0, p1, v0, v1);
  }
}

template <int M, int EPI, int RT, int WM, int WN>
void launch_prefill(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& s2, int kg,
                    int nb, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                    at::Tensor& out, int n, int64_t max_items) {
  using P = Staged<M, RT, WM, WN>;
  constexpr int SMEM = P::STAGES * P::STAGE_BYTES;
  auto* kern = fp4_prefill_kernel<M, EPI, RT, WM, WN>;
  static bool ready = false;
  if (!ready) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    ready = true;
  }
  const int64_t grid = max_items * ((nb + WN - 1) / WN);
  if (grid < 1) return;
  kern<<<static_cast<unsigned>(grid), P::THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), s2.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int M, int EPI, int SK, int UPB, int D, int RT>
void launch(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& s2, int kg, int nb,
            int rg, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out,
            int n, int64_t max_units) {
  auto* kern = fp4_kernel<M, EPI, SK, UPB, D, RT>;
  static int per_sm = 0;
  static int sms = 0;
  if (per_sm == 0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, UPB * SK * 32, 0);
    sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    per_sm = per_sm < 1 ? 1 : per_sm;
  }
  const int64_t need = (max_units + UPB - 1) / UPB;
  const int grid = static_cast<int>(need < (int64_t)per_sm * sms ? need : (int64_t)per_sm * sms);
  if (grid < 1) return;
  kern<<<grid, UPB * SK * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), s2.data_ptr<float>(), kg, nb, rg, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void nvfp4_prefill_experts_cuda(int64_t m, int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots,
                                const at::Tensor& w, const at::Tensor& s2, int64_t kg, int64_t nb,
                                const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                                at::Tensor& out, int64_t n, int64_t max_items) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n);
  // 8 warps a CTA, 64 pairs x 128 columns: two row tiles a warp, two warps down, four across
  if (m == 2 && epi == 2) return launch_prefill<2, 2, 2, 2, 4>(x, xs, sl, w, s2, k, b, items, counts, members, out, nn,
                                                               max_items);
  if (m == 1 && epi == 3) return launch_prefill<1, 3, 2, 2, 4>(x, xs, sl, w, s2, k, b, items, counts, members, out, nn,
                                                               max_items);
  TORCH_CHECK(false, "nvfp4 experts: no staged prompt kernel for ", m, " matrices, epilogue ", epi);
}

void nvfp4_split_experts_cuda(int64_t m, int64_t epi, int64_t sk, int64_t upb, int64_t d, int64_t rt,
                        const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                        const at::Tensor& s2, int64_t kg, int64_t nb, int64_t rg, const at::Tensor& items,
                        const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                        int64_t max_units) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), r = static_cast<int>(rg), nn = static_cast<int>(n);
#define TF_FP4(M_, EPI_, SK_, UPB_, D_, RT_)                                                          \
  if (m == M_ && epi == EPI_ && sk == SK_ && upb == UPB_ && d == D_ && rt == RT_) {                   \
    launch<M_, EPI_, SK_, UPB_, D_, RT_>(x, xs, sl, w, s2, k, b, r, items, counts, members, out, nn, \
                                         max_units);                                               \
    return;                                                                                           \
  }
  // decode form: one row tile a warp, K slices by K (1, 2 or 4), the block shape by the pairs of the call
#define TF_DECODE(M_, EPI_, UPB_, D_) \
  TF_FP4(M_, EPI_, 1, UPB_, D_, 1) TF_FP4(M_, EPI_, 2, UPB_, D_, 1) TF_FP4(M_, EPI_, 4, UPB_, D_, 1)
  TF_DECODE(2, 2, 1, 4) TF_DECODE(2, 2, 1, 2) TF_DECODE(2, 2, 2, 2) TF_DECODE(1, 0, 2, 2) TF_DECODE(1, 0, 4, 2)
#undef TF_DECODE
#undef TF_FP4
  TORCH_CHECK(false, "nvfp4 experts: no kernel for ", m, " matrices, epilogue ", epi, ", ", sk, " slices, ", upb,
              " units a block, ", d, " stages, ", rt, " row tiles");
}
