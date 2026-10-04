"""--keep-prefixes: the option, the startup estimate at different counts, and the cache's eviction. Host only."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda.capacity import GIB, Geometry, Weights, make_plan
from tensorfold.cuda.streams import PrefixCache
from tensorfold.families.qwen3_5_moe.cuda.engine import KEEP_MANY, stream_geometry

# Qwen3.6-35B-A3B's text_config (the fields the planner reads)
TEXT = {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_hidden_layers": 40,
        "full_attention_interval": 4, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "vocab_size": 248320, "intermediate_size": 512}
ATTENTION, LINEAR = 10, 30
KV_SLOT = 2 * 2 * 256 * 2 * (ATTENTION + 1)          # keys and values, bf16, the target's layers and the MTP head's
STATE = LINEAR * (32 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 32 * 128) * 2)


def _serve(tmp_path, monkeypatch, *flags, model_type="qwen3_5_moe"):
    from tensorfold import cli
    from tensorfold.cuda import server

    made = []
    family = SimpleNamespace(title="Test family", model_type=model_type,
                             package=SimpleNamespace(cuda_engine=lambda *a, **k: made.append(k) or
                                                     SimpleNamespace(max_len=8192)))
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=8192))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", *flags])
    return cli._serve_cuda(args, family, tmp_path, 8192), made


def test_the_option_reaches_the_engine_and_is_absent_by_default(tmp_path, monkeypatch):
    assert _serve(tmp_path, monkeypatch, "--parallel", "4", "--keep-prefixes", "16")[1][0]["keep_prefixes"] == 16
    assert "keep_prefixes" not in _serve(tmp_path, monkeypatch, "--parallel", "4")[1][0]


@pytest.mark.parametrize("flags,message", [(("--parallel", "4", "--keep-prefixes", "0"), "1 or more"),
                                           (("--parallel", "4", "--keep-prefixes", "-2"), "1 or more"),
                                           (("--keep-prefixes", "8"), "--parallel")])
def test_the_option_is_validated(tmp_path, monkeypatch, flags, message):
    with pytest.raises(ValueError, match=message):
        _serve(tmp_path, monkeypatch, *flags)


def test_other_families_refuse_the_option(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="qwen3_5_moe"):
        _serve(tmp_path, monkeypatch, "--parallel", "4", "--keep-prefixes", "8", model_type="qwen3_5")


def test_the_family_hands_the_count_to_the_engine_and_defaults_to_today_s(tmp_path, monkeypatch):
    from tensorfold.families import qwen3_5_moe
    from tensorfold.families.qwen3_5_moe.cuda import engine

    made = []
    monkeypatch.setattr(engine, "Qwen36Engine", lambda *a, **k: made.append(k))
    qwen3_5_moe.cuda_engine(tmp_path, parallel=4, context=1024)
    qwen3_5_moe.cuda_engine(tmp_path, parallel=4, context=1024, keep_prefixes=12)
    assert [m["keep"] for m in made] == [KEEP_MANY, 12] and KEEP_MANY == 3
    with pytest.raises(ValueError, match="--keep-prefixes"):
        qwen3_5_moe.cuda_engine(tmp_path, parallel=1, keep_prefixes=4)
    with pytest.raises(ValueError, match="--keep-prefixes"):
        qwen3_5_moe.cuda_engine(tmp_path, parallel=4, keep_prefixes=0)


def test_each_kept_state_costs_a_deltanet_copy_and_a_full_window_of_keys_and_values():
    window = 81920 + 5                      # the context and the draft slots past it
    for keep in (3, 8, 16):
        more = stream_geometry(TEXT, 4, keep + 1, 3).bytes_at(window) - stream_geometry(TEXT, 4, keep, 3).bytes_at(window)
        assert more == STATE + window * KV_SLOT
    one = stream_geometry(TEXT, 4, 4, 3).bytes_at(81920 + 4) - stream_geometry(TEXT, 4, 3, 3).bytes_at(81920 + 4)
    assert 1.7 * GIB < one < 1.9 * GIB       # about 1.8 GiB at --context 81920


def test_a_larger_count_shrinks_the_fitting_window_and_refuses_an_explicit_one_that_no_longer_fits():
    weights = Weights(22 * GIB, 0)

    def plan(keep, explicit):
        geometry = Geometry(stream_geometry(TEXT, 4, keep, 3).bytes_at, 4)
        return make_plan(262144, 81920, explicit, 60 * GIB, weights, geometry)

    assert plan(3, True).fitting >= 81920
    sizes = [plan(keep, False).fitting for keep in (3, 8, 16, 32)]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1]
    assert plan(32, True).fitting < 81920


def entry(name, n):
    return (list(range(n)) + [name], f"state-{name}", None)


def test_the_cache_evicts_the_oldest_unused_entry_first_and_keeps_what_a_prompt_resumed_from():
    cache = PrefixCache(3)
    for name in "abc":
        cache.add(*entry(name, 5))
    prompt = entry("a", 5)[0] + [99]
    assert cache.longest(prompt)[1] == "state-a"          # a was resumed from: now the newest and marked used
    cache.add(*entry("d", 5))
    assert [e[1] for e in cache.entries] == ["state-c", "state-a", "state-d"]      # b, the oldest unused, went
    cache.add(*entry("e", 5))
    assert [e[1] for e in cache.entries] == ["state-a", "state-d", "state-e"]      # c, unused, went before a


def test_a_conversation_adds_two_entries_so_a_larger_count_is_what_keeps_its_stop_point():
    def run(keep, others):
        cache = PrefixCache(keep)
        base = list(range(1000))
        cache.add(base[:900], "A-stop", None)            # a message start, then the prompt's end
        cache.add(base, "A-end", None)
        for i in range(others):
            ids = [10_000 * (i + 1) + j for j in range(300)]
            cache.add(ids[:200], f"o{i}-stop", None)
            cache.add(ids, f"o{i}-end", None)
        return cache.longest(base[:950] + [-1])           # an identical repeat resumes from the stop point

    assert run(3, 0)[1] == "A-stop" and run(3, 1) is None
    assert run(8, 3)[1] == "A-stop" and run(16, 6)[1] == "A-stop" and run(8, 4) is None
