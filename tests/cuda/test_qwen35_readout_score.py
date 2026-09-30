"""The readout scoring contract on a bf16 (dense, unquantized) Qwen3.5 checkpoint: chunk invariance of
``prefill_logits`` and ``Qwen27Engine.score``'s prefix-cache reuse against a cold prefill.

A synthetic two-layer model (one GDN layer, one full-attention layer, the 27B's head shapes at toy width) with
plain bf16 weights, so no checkpoint download is needed: this exercises the new dense-layout path end to end
without depending on the real Qwen3.5-4B download (covered separately, by hand, against the HF reference).
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.streams import PrefixCache
from tensorfold.families.qwen3_5.cuda.decode import prefill_logits
from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
from tensorfold.families.qwen3_5.cuda.prefill import prefill_state
from tensorfold.families.qwen3_5.cuda.weights import GDN, Attention, Config, Layer, QLinear, Weights

V = 512


def _dense(n: int, k: int, gen: torch.Generator) -> QLinear:
    w = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    return QLinear(w.contiguous(), None, None, layout="dense", bits=0, gs=0)


def _norm(d: int, gen: torch.Generator) -> torch.Tensor:
    return (1.0 + 0.1 * torch.randn(d, generator=gen, device="cuda")).bfloat16()


def _model() -> Weights:
    """Plain bf16 (unquantized) weights: one GDN layer then one full-attention layer, the 27B's head shapes."""

    gen = torch.Generator(device="cuda").manual_seed(4)
    # the GDN CUDA kernel's state is a fixed (Hv, Dv, 128) tensor: dk must be 128, as the other synthetic-weight
    # GDN tests (test_qwen27_prompt_end_cache.py) also use. head_dim=256 matches Qwen3.5-4B's own (not just a
    # round number): attention_texts' shared-memory footprint scales with it, and 128 alone missed the overflow
    # a real 256-head_dim request hit (see the TEXT_BM fix in prefill_attention.py).
    c = Config(hidden=128, intermediate=128, layers=2, heads=2, kv_heads=1, head_dim=256, vocab=V, k_heads=1,
              v_heads=2, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32, rope_theta=10000000.0,
              eos=(0,))

    def mlp(hidden):
        return _dense(c.intermediate, hidden, gen), _dense(c.intermediate, hidden, gen), _dense(hidden, c.intermediate, gen)

    layers = []
    for i in range(c.layers):
        gate, up, down = mlp(c.hidden)
        if c.is_linear(i):
            cd = 2 * c.k_heads * c.dk + c.v_heads * c.dv
            gdn = GDN(_dense(cd, c.hidden, gen), _dense(c.v_heads * c.dv, c.hidden, gen),
                      _dense(c.v_heads, c.hidden, gen), _dense(c.v_heads, c.hidden, gen),
                      _dense(c.hidden, c.v_heads * c.dv, gen),
                      (torch.randn(cd, c.conv_kernel, generator=gen, device="cuda") * 0.1).bfloat16(),
                      torch.randn(c.v_heads, generator=gen, device="cuda") * 0.5,
                      torch.randn(c.v_heads, generator=gen, device="cuda") * 0.5, _norm(c.dv, gen))
            layers.append(Layer(True, _norm(c.hidden, gen), _norm(c.hidden, gen), gdn, None, gate, up, down))
        else:
            kv = c.kv_heads * c.head_dim
            attn = Attention(_dense(c.heads * c.head_dim * 2, c.hidden, gen), _dense(kv, c.hidden, gen),
                             _dense(kv, c.hidden, gen), _dense(c.hidden, c.heads * c.head_dim, gen),
                             _norm(c.head_dim, gen), _norm(c.head_dim, gen))
            layers.append(Layer(False, _norm(c.hidden, gen), _norm(c.hidden, gen), None, attn, gate, up, down))
    embed = _dense(V, c.hidden, gen)
    w = Weights(config=c, embed=embed, layers=layers, norm=_norm(c.hidden, gen), head=embed,
               inv_freq=torch.ones(c.rope_dims // 2, device="cuda"))
    assert w.fast_prefill is False           # dense weights always take the bf16 prompt-glue path
    return w


@pytest.fixture(scope="module")
def w():
    return _model()


def _prompt(n: int, seed: int) -> list[int]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (n,), generator=gen).tolist()


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


@pytest.mark.parametrize("n,size", [(4096, 4096), (4096, 512), (300, 7), (67, 16)])
def test_prefill_logits_is_chunk_invariant_on_dense_weights(w, n, size):
    """The readout's one-forward logits do not depend on how the prompt was chunked, on plain bf16 weights."""

    from tensorfold.families.qwen3_5.cuda.forward import State
    from tensorfold.families.qwen3_5.cuda.prefill import head_logits

    prompt = _prompt(n, seed=3)
    whole, _ = prefill_logits(w, prompt)
    st = State(w)
    # precise=True: prefill_logits's own fp32-residual path (SEE-3828 round 3); without it this is a plain bf16
    # prefill, a different (merely close) answer, not a chunk-invariance regression.
    normed = prefill_state(w, prompt, st, size=size, precise=True)
    chunked = head_logits(w, normed)
    assert _same_bits(whole, chunked)


def _fake_engine(w: Weights) -> Qwen27Engine:
    """A ``Qwen27Engine`` with real weights and cache but none of ``__init__``'s file/CUDA-capacity admission."""

    e = Qwen27Engine.__new__(Qwen27Engine)
    e.w, e.tp, e.context_window = w, 1, 100_000
    e.cache, e.points, e.scheduler = PrefixCache(8), None, None
    return e


def test_score_prefix_reuse_matches_a_cold_prefill(w):
    base = _prompt(300, seed=11)
    tail_a = _prompt(20, seed=12)
    tail_b = _prompt(30, seed=13)
    prompt_a = base + tail_a
    prompt_b = base + tail_b            # shares every token with prompt_a except the tails

    warm = _fake_engine(w)
    logits_a = warm.score(prompt_a)      # remembers state at len(prompt_a) - 1 (entry_end)

    # prompt_b's shared prefix is base + tail_a[:-1] (entry_end(prompt_a)), one token short of prompt_a: it still
    # extends that cached entry, so scoring it on ``warm`` should resume instead of a cold prefill.
    warm_logits_b = warm.score(prompt_b)

    cold = _fake_engine(w)
    cold_logits_b = cold.score(prompt_b)

    assert _same_bits(warm_logits_b, cold_logits_b)
    # scoring prompt_b did not retroactively change prompt_a's already-returned logits
    assert _same_bits(logits_a, _fake_engine(w).score(prompt_a))


def test_scoring_one_request_does_not_disturb_an_unrelated_ones_result(w):
    """A request's score must not depend on what else the engine scored around it (batch invariance)."""

    prompt_a = _prompt(50, seed=21)
    prompt_c = _prompt(40, seed=99)      # shares no prefix with prompt_a

    shared = _fake_engine(w)
    first = shared.score(prompt_a)
    second = shared.score(prompt_c)
    third = shared.score(prompt_a)

    solo_a = _fake_engine(w).score(prompt_a)
    solo_c = _fake_engine(w).score(prompt_c)

    assert _same_bits(first, solo_a)
    assert _same_bits(third, solo_a)
    assert _same_bits(second, solo_c)


def test_multi_prefill_logits_is_invariant_to_what_shares_the_batch(w):
    """A text's batched logits do not depend on which other texts share the call, their order or their lengths —
    exactly, at a fixed total row count; to a tight fp32 tolerance across different total row counts.

    The dense tensor-core matmul picks its block shape from the *total* row count (``dense.blocks_for``); its own
    docstring claims every shape gives the same bits, and that holds at bf16 (round 2's test, before this
    checkpoint's residual stream went fp32, passed bit-exact). At full fp32 the two shapes' tl.dot accumulation
    can differ by a couple of ULPs — invisible once rounded to bf16, visible once nothing rounds it away. This is
    two to three orders of magnitude below the bf16-vs-fp32 gaps this checkpoint precision push targets (fp32
    lm_head etc.), so a tight tolerance rather than exact bits is the right check across different totals; the
    same total (just reordered) is still held to exact bits below.
    """

    from tensorfold.families.qwen3_5.cuda.prefill import multi_prefill_logits

    a, b, c, d = (_prompt(n, seed=s) for n, s in ((37, 1), (91, 2), (60, 3), (140, 4)))

    solo_a = multi_prefill_logits(w, [a])
    abc = multi_prefill_logits(w, [a, b, c])
    cba = multi_prefill_logits(w, [c, b, a])
    ad = multi_prefill_logits(w, [a, d])

    def close(x, y, tol=1e-3):
        diff = (x.float() - y.float()).abs().max().item()
        assert diff < tol, f"max abs diff {diff}"

    close(solo_a[0], abc[0])                        # alone (total 37) vs. first in a batch of three (total 188)
    close(solo_a[0], cba[2])                         # alone vs. last, reverse order, same total (188)
    close(solo_a[0], ad[0])                          # alone vs. batched with a different, longer text (total 177)
    # abc and cba share the SAME total row count (188): the tensor-core matmul's block shape depends only on
    # that total, so a text's own row is bit-exact regardless of which other texts (or order) fill out the batch.
    assert _same_bits(abc[1], cba[1])                # b's row: same either way (its own position in both)
    assert _same_bits(abc[2], cba[0])                # c's row: same either way


def test_multi_prefill_logits_matches_the_single_stream_path_closely(w):
    """Batched and single-stream prefill take different (but each internally exact) attention tilings, so their
    bits differ slightly; they should still agree closely and pick the same top token."""

    prompt = _prompt(200, seed=7)
    solo, _ = prefill_logits(w, prompt)
    from tensorfold.families.qwen3_5.cuda.prefill import multi_prefill_logits

    batched = multi_prefill_logits(w, [prompt])
    diff = (solo.float() - batched[0].float()).abs()
    assert diff.max().item() < 1.0, f"max abs diff {diff.max().item()}"
    assert int(solo.argmax()) == int(batched[0].argmax())
