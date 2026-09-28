"""SHA-256 of Qwen3.6's CUDA outputs on fixed inputs: python tools/hash_qwen36.py OUT.json, then diff."""

import hashlib
import json
import sys

import torch

sys.path[:0] = ["tests/cuda"]

from tensorfold.families.qwen3_5_moe.cuda import moe, qmm  # noqa: E402

DEV = "cuda"
OUT = {}
E, W, D = moe.EXPERTS, moe.WIDTH, moe.HIDDEN


def h(name, *ts):
    m = hashlib.sha256()
    for t in ts:
        if isinstance(t, (list, tuple)):
            t = torch.tensor(t)
        t = t.detach().contiguous().cpu()
        m.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
        m.update(t.view(torch.uint8).numpy().tobytes())
    OUT[name] = m.hexdigest()


def mlx(n, k, seed, lead=()):
    g = torch.Generator(device=DEV).manual_seed(seed)
    words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=g, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((*lead, n, k // 64), generator=g, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((*lead, n, k // 64), generator=g, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


for n, k in qmm.SPLITS:
    q = qmm.make_q4(*mlx(n, k, n + k))
    for m in (1, 2, 5, 16, 17, 64, 200):
        g = torch.Generator(device=DEV).manual_seed(m * 31 + n)
        x = torch.randn((m, k), generator=g, device=DEV).to(torch.bfloat16)
        h(f"qmm/{n}x{k}/{m}", qmm.matmul(x, q), qmm.matmul(x, q, f32=True))
        if qmm.split_for(n, k) > 1:
            h(f"qmm_parts/{n}x{k}/{m}", qmm.matmul(x, q, reduce=False))

ex = qmm.make_experts(mlx(W, D, 11, (E + 1,)), mlx(W, D, 12, (E + 1,)), mlx(D, W, 13, (E + 1,)))
g = torch.Generator(device=DEV).manual_seed(5)
table = (torch.randn((E + 1, D), generator=g, device=DEV) * 0.02).contiguous()
skew = table.clone()
skew[:16] += 0.05 * torch.randn((1, D), generator=g, device=DEV)
big = moe.buffers(512, DEV)
for tname, tab in (("even", table), ("skew", skew)):
    for rows in (1, 2, 3, 5, 16, 17, 33, 64, 128, 129, 200, 511, 512):
        gx = torch.Generator(device=DEV).manual_seed(1000 + rows)
        x = torch.randn((rows, D), generator=gx, device=DEV)
        if tname == "skew":
            x = x + 0.5 * skew[:16].sum(0) / skew[:16].norm()
        x = x.to(torch.bfloat16)
        for bname, buf in (("own", moe.buffers(rows, DEV)), ("view", big)):
            sub = moe.moe(x, None, tab, ex, buf)
            key = f"{tname}/{rows}/{bname}"
            h(f"{key}/logits", sub.logits)
            h(f"{key}/select", sub.pick, sub.wts)
            h(f"{key}/act", sub.act)
            h(f"{key}/y", sub.y)
            h(f"{key}/combine", moe.combine(sub.y, sub.wts))

import test_qwen36moe_forward as TF  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.decode import Decoder, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5_moe.cuda.forward import prepare  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

model = prepare(TF._weights())
e = Decoder(model, capacity=1024, rows=512, states=2)
prompt = TF._tokens(600, 3)
for sampling in (None, Sampling(seed=7, top_k=20, top_p=0.95)):
    tag = "greedy" if sampling is None else "sampled"
    first = prefill(e, prompt, sampling)
    h(f"fwd/{tag}/prefill_state", e.st.rec[e.st.cur], e.st.conv, e.st.kc[:, :600], e.st.vc[:, :600])
    h(f"fwd/{tag}/serial", serial_decode(e, first, 24, sampling).tokens)
prefill(e, prompt, None)
st = e.pool.clone(e.st)
h("fwd/window_logits", e.forward(TF._tokens(9, 4), st)[:9])

torch.cuda.synchronize()
print(torch.cuda.get_device_name(), torch.__version__, file=sys.stderr)
path = sys.argv[1]
with open(path, "w") as f:
    json.dump(OUT, f, indent=0, sort_keys=True)
print(f"{len(OUT)} hashes -> {path}")
