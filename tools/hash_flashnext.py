"""SHA-256 of Flash Next's CUDA outputs on fixed inputs: python tools/hash_flashnext.py OUT.json, then diff."""

import hashlib
import json
import sys

import torch

sys.path[:0] = ["tests/cuda"]

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io, moe, qmm  # noqa: E402

DEV = "cuda"
OUT = {}


def h(name, *ts):
    m = hashlib.sha256()
    for t in ts:
        if isinstance(t, (list, tuple)):
            t = torch.tensor(t)
        t = t.detach().contiguous().cpu()
        m.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
        m.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes())
    OUT[name] = m.hexdigest()


# inputs are drawn on the CPU and moved: a GPU generator's draws depend on the SM count
def mlx(n, k, seed, lead=(), gs=32):
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=g, dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((*lead, n, k // gs), generator=g) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((*lead, n, k // gs), generator=g) * 0.02).to(torch.bfloat16)
    return words.to(DEV), scales.to(DEV), biases.to(DEV)


def xin(m, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((m, k), generator=g).to(torch.bfloat16).to(DEV)


# -- lane matmul -----------------------------------------------------------------------------------------------
SHAPES = [(324, 10240), (320, 10240), (10240, 320), (16480, 2560), (2560, 6144), (13952, 2560), (2560, 2560),
          (640, 2560), (248320 // 8, 2560)]
ROWS = [1, 2, 5, 16, 17, 33, 64, 130]
for n, k in SHAPES:
    w = mlx(n, k, n * 7 + k)
    for layout in ("frag", "tiled"):
        q = qmm.make_q4(*w, layout)
        h(f"make_q4/{layout}/{n}x{k}", q.weight, q.scales, q.biases)
        h(f"to_mlx/{layout}/{n}x{k}", *qmm.to_mlx(q))
        for m in ROWS:
            x = xin(m, k, m * 31 + n)
            xs = qmm.group_sums(x)
            h(f"group_sums/{n}x{k}/{m}", xs)
            h(f"matmul/{layout}/{n}x{k}/{m}", qmm.matmul(x, q, xs))
            h(f"matmul_f32/{layout}/{n}x{k}/{m}", qmm.matmul(x, q, xs, f32=True))
            if qmm.split_for(n, k) > 1:
                h(f"matmul_parts/{layout}/{n}x{k}/{m}", qmm.matmul(x, q, xs, reduce=False))
            if layout == "frag" and m in (5, 64):
                h(f"prefill/{n}x{k}/{m}", qmm.prefill_matmul(x, q))
        x = xin(9, 2 * k, 99 + n)[:, :k]
        h(f"matmul_strided/{layout}/{n}x{k}", qmm.matmul(x, q))
    if n * k <= 16480 * 2560:
        h(f"dequantize/{n}x{k}", qmm.dequantize(*w))
    h(f"split/{n}x{k}", torch.tensor([qmm.split_for(n, k), qmm.split_k(n, k)]))

S, D, LOW = 4, 2560, 320
down = qmm.stack_q4([mlx(LOW, S * D, 5), mlx(S, S * D, 6)], "tiled")
up = qmm.make_q4(*mlx(S * D, LOW, 7), "tiled")
for rows in (1, 7, 16):
    g = torch.Generator().manual_seed(rows)
    hh = (torch.randn((rows, S * D), generator=g) * 3).to(torch.bfloat16).to(DEV)
    pss = (torch.rand((rows, D // 256, S), generator=g) * 10 + 1).to(DEV)
    scale = (1 + 0.05 * torch.randn((S * D,), generator=g)).to(DEV)
    normed = torch.empty((rows, S * D), dtype=torch.bfloat16, device=DEV)
    sk = qmm.split_for(down.n, down.k)
    out = torch.empty((rows, down.n), dtype=torch.bfloat16, device=DEV)
    part = torch.empty((sk * rows * down.n,), dtype=torch.float32, device=DEV)
    got = qmm.hc_down(hh, pss, scale, normed, down, 1e-6, S, out=out, part=part)
    h(f"hc_down/{rows}", got, normed)
    act = torch.randn((rows, LOW), generator=g).to(torch.bfloat16).to(DEV)
    xs = qmm.group_sums(act)
    mixed = torch.empty((rows, D), dtype=torch.bfloat16, device=DEV)
    xsm = torch.empty((rows, D // 32), dtype=torch.float32, device=DEV)
    qmm.hc_upmix(act, xs, up, normed, mixed, xsm, S)
    h(f"hc_upmix/{rows}", mixed, xsm)


# -- MoE ---------------------------------------------------------------------------------------------------------
E, W, DD = 64, 640, 2560


def table(routed, shared):
    return tuple(torch.cat([a, b[None]]) for a, b in zip(routed, shared))


ex = grouped.make([table(mlx(W, DD, 11, (E,)), mlx(W, DD, 14)), table(mlx(W, DD, 12, (E,)), mlx(W, DD, 15))],
                  table(mlx(DD, W, 13, (E,)), mlx(DD, W, 16)), 32)
h("experts", ex.up, ex.down)


class Cfg:
    num_experts_per_tok = 10
    num_experts = 64
    moe_intermediate_size = 640
    hidden_size = 2560


g = torch.Generator().manual_seed(5)
router_rows = (torch.randn((65, DD), generator=g) * 0.02).to(torch.bfloat16).to(DEV)
for rows in (1, 2, 3, 4, 8, 16, 17, 40):
    x = xin(rows, DD, 1000 + rows)
    for prefill in (False, True):
        buf = moe.MoEBuffers(rows, Cfg, DEV, prefill=prefill)
        moe.moe(x, router_rows, ex, buf, 10, 64)
        tag = f"moe/{'prefill' if prefill else 'decode'}/{rows}"
        h(f"{tag}/logits", buf.logits)
        n_items = int(buf.plan.counts[0])
        h(f"{tag}/select", buf.pick, buf.wts, buf.plan.members[:rows * 11], buf.plan.items[:n_items],
          buf.plan.counts)
        h(f"{tag}/act", buf.act)
        h(f"{tag}/y", buf.y)


# -- GDN chain, replay, gdn_io -------------------------------------------------------------------------------------
def gdn_inputs(nk, nv, rows, seed):
    g = torch.Generator().manual_seed(seed)
    conv, pw = gdn.widths(nk, nv)

    def rnd(shape, scale):
        return torch.randn(shape, generator=g) * scale

    slow = 2.0 * (torch.arange(nv) % 2)
    vals = (rnd((rows, pw), 0.5).to(torch.bfloat16), rnd((3, conv), 0.5).to(torch.bfloat16),
            rnd((conv, 4), 0.3).to(torch.bfloat16), rnd((nv, 128, 128), 0.05), rnd((nv,), 0.5) - slow,
            rnd((nv,), 0.5) - slow, (1 + rnd((128,), 0.1)).to(torch.bfloat16))
    return [v.to(DEV) for v in vals]


for nk, nv in ((16, 48), (8, 24)):
    for rows in (1, 4, 9, 17):
        p, cs, cw, state, a_log, dt, nw = gdn_inputs(nk, nv, rows, 500 + rows + nv)
        sc = gdn.GDNScratch(rows, DEV, nk, nv)
        out = torch.empty((rows, nv * 128), dtype=torch.bfloat16, device=DEV)
        xs = torch.empty((rows, nv * 4), dtype=torch.float32, device=DEV)
        st = torch.empty_like(state)
        gdn.chain(p, cs, cw, state, a_log, dt, nw, 1e-6, rows, sc, st, out, xs)
        h(f"gdn/{nk}x{nv}/{rows}", out, xs, sc.k, sc.v, sc.g, sc.b, st)
        for keep in sorted({1, (rows + 1) // 2, rows}):
            rep = torch.empty_like(state)
            gdn.replay(state, sc, keep, rep)
            h(f"gdn_replay/{nk}x{nv}/{rows}/{keep}", rep)
        if nv == 48:
            win = (torch.arange(rows, dtype=torch.int32, device=DEV)[:, None]
                   + torch.arange(4, dtype=torch.int32, device=DEV)).contiguous()
            sid = torch.zeros((rows,), dtype=torch.int32, device=DEV)
            ptr = torch.tensor([cs.data_ptr()], dtype=torch.int64, device=DEV)
            q, k, v, gt, beta = gdn_io.front(p, ptr, sid, win, cw, a_log, dt, nk)
            h(f"gdn_io_front/{rows}", q, k, v, gt, beta)
            y = torch.randn((rows, nv, 128), generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16).to(DEV)
            bo = torch.empty((rows, nv * 128), dtype=torch.bfloat16, device=DEV)
            bx = torch.empty((rows, nv * 4), dtype=torch.float32, device=DEV)
            gdn_io.back(y, p, nw, 1e-6, bo, bx)
            h(f"gdn_io_back/{rows}", bo, bx)


# -- attention: the Flash Next cases of tests/cuda/test_qwen36moe_attention.py --------------------------------------
import test_qwen36moe_attention as TA  # noqa: E402

for case in sorted(TA.FN_CASES):
    for stage, digest in TA.fn_outputs(case).items():
        OUT[f"attn/{case}/{stage}"] = digest


# -- end to end: the random-weight Flash Next model ------------------------------------------------------------------
import test_flashnext_forward as TF  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402

wts = TF._model()
prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]
for sampling in (None, Sampling(seed=1234, top_k=20, top_p=0.95)):
    tag = "greedy" if sampling is None else "sampled"
    eager = Engine(wts, capacity=1024, max_rows=8, prefill_rows=16)
    first = prefill(eager, prompt, sampling)
    ref = serial_decode(eager, first, 24, sampling).tokens
    h(f"e2e/{tag}/serial", ref)
    h(f"e2e/{tag}/state", eager.st.rec, eager.st.conv, *eager.st.kc, *eager.st.vc)
    graphs = Engine(wts, capacity=1024, max_rows=8, prefill_rows=16, graphs=True)
    prefill(graphs, prompt, sampling)
    h(f"e2e/{tag}/graphs", serial_decode(graphs, first, 24, sampling).tokens)
    for depth in (1, 3):
        prefill(eager, prompt, sampling)
        got = mtp_decode(eager, first, 24, sampling, depth=depth, confidence=0.0)
        h(f"e2e/{tag}/mtp{depth}", got.tokens)
e = Engine(wts, capacity=1024, max_rows=8, prefill_rows=16)
prefill(e, prompt, None)
st = e.st.clone()
lg = forward(wts, st, e.buf, [401, 33, 2048, 5, 77])
h("e2e/window_logits", lg[:5], e.buf.streams[:5])
commit(wts, st, e.buf, 5, 3)
h("e2e/window_commit", st.rec, st.conv, forward(wts, st, e.buf, [5])[:1])

torch.cuda.synchronize()
print(torch.cuda.get_device_name(), torch.__version__, file=sys.stderr)
path = sys.argv[1]
with open(path, "w") as f:
    json.dump(OUT, f, indent=0, sort_keys=True)
print(f"{len(OUT)} hashes -> {path}")
