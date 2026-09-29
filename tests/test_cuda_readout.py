"""The readout wrapper's scoring contract: request validation and response shape (no CUDA, no torch needed here)."""

from __future__ import annotations

import pytest

from tensorfold.cuda.readout import ScoreError, build_choice, is_score_request, parse_score_request, token_id_key

VALID = {
    "model": "qwen3.5-4b-readout", "prompt": [1, 2, 3], "max_tokens": 1, "temperature": 0,
    "return_tokens_as_token_ids": True, "logprobs": 3, "allowed_token_ids": [5, 6, 7],
}


def test_is_score_request_keys_on_return_tokens_as_token_ids():
    assert is_score_request(VALID)
    assert is_score_request({k: v for k, v in VALID.items() if k != "allowed_token_ids"})  # the no-options fallback
    assert not is_score_request({"model": "x", "prompt": "hi"})
    assert not is_score_request({**VALID, "return_tokens_as_token_ids": False})
    assert not is_score_request("not a dict")


def test_valid_request_parses():
    parsed = parse_score_request(VALID)
    assert parsed.prompt == [1, 2, 3]
    assert parsed.allowed_token_ids == [5, 6, 7]
    assert parsed.num_logprobs == 3


@pytest.mark.parametrize("field,value", [
    ("prompt", "not a list"),
    ("prompt", []),
    ("prompt", [1, "two", 3]),
    ("prompt", [1, 2.5, 3]),
    ("prompt", [1, True, 3]),
])
def test_prompt_must_be_a_nonempty_list_of_ints(field, value):
    body = {**VALID, field: value}
    with pytest.raises(ScoreError, match="prompt"):
        parse_score_request(body)


@pytest.mark.parametrize("max_tokens", [0, 2, None, "1"])
def test_max_tokens_must_be_exactly_one(max_tokens):
    with pytest.raises(ScoreError, match="max_tokens"):
        parse_score_request({**VALID, "max_tokens": max_tokens})


@pytest.mark.parametrize("temperature", [0.1, 1, -0.0001])
def test_temperature_must_be_zero(temperature):
    with pytest.raises(ScoreError, match="temperature"):
        parse_score_request({**VALID, "temperature": temperature})


def test_temperature_zero_int_or_float_both_accepted():
    assert parse_score_request({**VALID, "temperature": 0.0}).prompt == [1, 2, 3]


@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_return_tokens_as_token_ids_must_be_true(value):
    with pytest.raises(ScoreError, match="return_tokens_as_token_ids"):
        parse_score_request({**VALID, "return_tokens_as_token_ids": value})


@pytest.mark.parametrize("value", ["nope", [], [1, "x"], [1, 1]])
def test_allowed_token_ids_must_be_a_nonempty_list_of_distinct_ints(value):
    with pytest.raises(ScoreError, match="allowed_token_ids"):
        parse_score_request({**VALID, "allowed_token_ids": value})


def test_missing_allowed_token_ids_is_the_no_options_fallback():
    """A readout whose options tokenize to nothing sends the request without allowed_token_ids at all."""

    body = {**VALID, "logprobs": 20}
    del body["allowed_token_ids"]
    parsed = parse_score_request(body)
    assert parsed.allowed_token_ids is None
    assert parsed.num_logprobs == 20


@pytest.mark.parametrize("value", [0, 65, -1, 1.5, True, "3"])
def test_logprobs_must_be_an_int_between_1_and_64(value):
    with pytest.raises(ScoreError, match="logprobs"):
        parse_score_request({**VALID, "logprobs": value})


def test_logprobs_must_cover_the_whole_allowed_set():
    body = {**VALID, "allowed_token_ids": [1, 2, 3, 4, 5], "logprobs": 3}
    with pytest.raises(ScoreError, match="logprobs"):
        parse_score_request(body)


def test_logprobs_may_be_capped_at_64_even_with_a_larger_allowed_set():
    body = {**VALID, "allowed_token_ids": list(range(100)), "logprobs": 64}
    parsed = parse_score_request(body)
    assert parsed.num_logprobs == 64


def test_stream_is_refused():
    with pytest.raises(ScoreError, match="stream"):
        parse_score_request({**VALID, "stream": True})


def test_vocab_bounds_are_enforced_when_known():
    with pytest.raises(ScoreError, match="vocabulary"):
        parse_score_request({**VALID, "prompt": [1, 2, 999]}, vocab=100)
    with pytest.raises(ScoreError, match="vocabulary"):
        parse_score_request({**VALID, "allowed_token_ids": [1, -1]}, vocab=100)
    parse_score_request(VALID, vocab=1000)          # no vocab check failure


def test_not_a_dict_is_refused():
    with pytest.raises(ScoreError, match="JSON object"):
        parse_score_request(None)


def test_token_id_key_format():
    assert token_id_key(42) == "token_id:42"


def test_build_choice_shape_and_sorting():
    ranked = [(5, -2.0), (6, -0.1), (7, -5.0)]
    choice = build_choice(6, ranked, num_logprobs=3)
    assert choice["text"] == "token_id:6"
    assert choice["finish_reason"] == "length"
    logprobs = choice["logprobs"]
    assert logprobs["tokens"] == ["token_id:6"]
    assert logprobs["token_logprobs"] == [-0.1]
    assert logprobs["top_logprobs"] == [{"token_id:5": -2.0, "token_id:6": -0.1, "token_id:7": -5.0}]


def test_build_choice_trims_top_logprobs_to_num_logprobs():
    ranked = [(1, -0.1), (2, -1.0), (3, -2.0)]
    choice = build_choice(1, ranked, num_logprobs=2)
    assert choice["logprobs"]["top_logprobs"] == [{"token_id:1": -0.1, "token_id:2": -1.0}]


def test_rank_allowed_renormalises_over_the_allowed_set_only():
    """vLLM's processed_logprobs at temperature 0: log-softmax the raw logits after masking to the allowed set, so
    the allowed set's logprobs alone exponentiate to 1, and the chosen id is the allowed set's argmax raw logit."""

    import math

    torch = pytest.importorskip("torch")
    from tensorfold.cuda.readout import rank_allowed

    gen = torch.Generator().manual_seed(0)
    vocab = 1000
    logits = (torch.randn(1, vocab, generator=gen) * 5)
    allowed = sorted(torch.randperm(vocab, generator=gen)[:17].tolist())

    chosen, ranked = rank_allowed(logits, allowed)

    assert sorted(tid for tid, _ in ranked) == allowed
    total = sum(math.exp(lp) for _, lp in ranked)
    assert total == pytest.approx(1.0, abs=1e-4)
    assert chosen == allowed[int(logits[0, allowed].argmax())]
    assert ranked == sorted(ranked, key=lambda kv: -kv[1])


def test_rank_allowed_matches_manual_masked_log_softmax():
    torch = pytest.importorskip("torch")
    from tensorfold.cuda.readout import rank_allowed

    logits = torch.tensor([[1.0, 5.0, 2.0, -3.0, 0.5]])
    allowed = [0, 2, 4]
    chosen, ranked = rank_allowed(logits, allowed)
    expect = torch.tensor([1.0, 2.0, 0.5]).log_softmax(dim=-1)
    got = {tid: lp for tid, lp in ranked}
    assert got[0] == pytest.approx(expect[0].item(), abs=1e-6)
    assert got[2] == pytest.approx(expect[1].item(), abs=1e-6)
    assert got[4] == pytest.approx(expect[2].item(), abs=1e-6)
    assert chosen == 2


def test_rank_allowed_none_is_unrestricted_top_k_over_the_vocab():
    """The wrapper's no-options fallback: no masking, ranked over the whole vocabulary's top ``num_logprobs``."""

    torch = pytest.importorskip("torch")
    from tensorfold.cuda.readout import rank_allowed

    logits = torch.tensor([[1.0, 5.0, 2.0, -3.0, 0.5]])
    chosen, ranked = rank_allowed(logits, None, num_logprobs=3)
    expect = logits[0].log_softmax(dim=-1)
    assert chosen == 1                                  # argmax of the raw (unmasked) logits
    assert [tid for tid, _ in ranked] == [1, 2, 0]       # top 3 by logprob, descending
    assert ranked[0][1] == pytest.approx(expect[1].item(), abs=1e-6)


class _FakeEngine:
    """Records every call it gets; ``score``/``score_batch`` return the prompt itself so a test can check routing."""

    def __init__(self):
        import threading

        self.calls: list[tuple[str, object]] = []
        self.lock = threading.Lock()

    def score(self, prompt):
        with self.lock:
            self.calls.append(("score", prompt))
        return f"solo:{prompt}"

    def score_batch(self, prompts):
        with self.lock:
            self.calls.append(("score_batch", tuple(prompts)))
        return [f"batch:{p}" for p in prompts]


class _FakeEngineNoBatch:
    """Like ``_FakeEngine`` but with no ``score_batch`` at all, as an engine that predates batching would be."""

    def __init__(self):
        import threading

        self.calls: list[tuple[str, object]] = []
        self.lock = threading.Lock()

    def score(self, prompt):
        with self.lock:
            self.calls.append(("score", prompt))
        return f"solo:{prompt}"


def test_score_batcher_batches_concurrent_submissions():
    import threading

    from tensorfold.cuda.readout import ScoreBatcher

    engine = _FakeEngine()
    batcher = ScoreBatcher(engine, threading.Lock(), window_s=0.05, max_batch=8)
    results: dict[int, str] = {}

    def worker(i):
        results[i] = batcher.score(f"p{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert results == {i: f"batch:p{i}" for i in range(4)}
    assert engine.calls == [("score_batch", ("p0", "p1", "p2", "p3"))]


def test_score_batcher_uses_the_solo_path_when_nothing_else_is_pending():
    import threading

    from tensorfold.cuda.readout import ScoreBatcher

    engine = _FakeEngine()
    batcher = ScoreBatcher(engine, threading.Lock(), window_s=0.01, max_batch=8)
    assert batcher.score("only") == "solo:only"
    assert engine.calls == [("score", "only")]


def test_score_batcher_flushes_early_at_max_batch_without_waiting_the_window():
    import threading
    import time

    from tensorfold.cuda.readout import ScoreBatcher

    engine = _FakeEngine()
    batcher = ScoreBatcher(engine, threading.Lock(), window_s=5.0, max_batch=2)   # a window that would never fire
    results: dict[int, str] = {}

    def worker(i):
        results[i] = batcher.score(f"p{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
    assert time.perf_counter() - start < 1.0             # did not wait for the 5s window
    assert results == {0: "batch:p0", 1: "batch:p1"}


def test_score_batcher_falls_back_to_solo_calls_without_score_batch():
    import threading

    from tensorfold.cuda.readout import ScoreBatcher

    engine = _FakeEngineNoBatch()
    batcher = ScoreBatcher(engine, threading.Lock(), window_s=0.05, max_batch=8)
    results: dict[int, str] = {}

    def worker(i):
        results[i] = batcher.score(f"p{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert results == {i: f"solo:p{i}" for i in range(3)}
    assert sorted(engine.calls) == sorted(("score", f"p{i}") for i in range(3))


def test_score_batcher_delivers_the_exception_to_every_waiter():
    import threading

    from tensorfold.cuda.readout import ScoreBatcher

    class Failing:
        def score_batch(self, prompts):
            raise ValueError("boom")

    batcher = ScoreBatcher(Failing(), threading.Lock(), window_s=0.05, max_batch=8)
    errors: dict[int, Exception] = {}

    def worker(i):
        try:
            batcher.score(f"p{i}")
        except ValueError as exc:
            errors[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert len(errors) == 3 and all(str(e) == "boom" for e in errors.values())
