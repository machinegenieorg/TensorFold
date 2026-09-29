"""The readout wrapper's scoring contract: token-id prompts, an allowed set, one forward, vLLM-shaped logprobs.

The wrapper (``readout/readout_server.py``'s ``_backend_logp``) sends one request shape to ``POST
/v1/completions``: a token-id ``prompt``, ``max_tokens: 1``, ``temperature: 0``, ``return_tokens_as_token_ids:
true``, an integer ``logprobs``, and ``allowed_token_ids`` — except when a readout's options tokenize to no
first-token ids at all, in which case it sends the same request without ``allowed_token_ids`` (unrestricted,
``logprobs: 20``) instead. Anything outside those two shapes is refused (400) rather than guessed at.

vLLM's ``--logprobs-mode processed_logprobs`` at ``temperature: 0`` (v0.25.1, ``vllm/v1/sample/sampler.py``):
``Sampler.apply_logits_processors`` masks every id outside ``allowed_token_ids`` to ``-inf`` on the raw logits
(``sampling_metadata.allowed_token_ids_mask`` then ``logits.masked_fill_(mask, float("-inf"))``), before
``Sampler.sample`` ever runs. At ``temperature: 0`` every request is ``all_greedy``, so ``sample`` takes its early
branch: it greedily argmaxes the already-masked logits (never scaling by temperature, since ``apply_temperature``
sits after that branch's early return) and, for ``logprobs_mode == "processed_logprobs"``, sets
``processed_logprobs = self.compute_logprobs(logits) = logits.log_softmax(dim=-1, dtype=torch.float32)`` on those
same masked logits. So at temperature 0 the reported logprobs are an ordinary log-softmax over the raw (unscaled)
logits, renormalised over exactly the allowed set (every other token has probability 0 from the ``-inf`` mask), and
the reported completion token is whichever allowed id had the largest raw logit. ``vllm/entrypoints/openai/
completion/serving.py`` (``format_token_id_placeholder``) is why ``return_tokens_as_token_ids`` renders every token
as ``"token_id:<id>"`` in both ``tokens`` and the ``top_logprobs`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_LOGPROBS = 64


class ScoreError(ValueError):
    """A scoring request outside the readout contract; the caller turns this into a 400."""


@dataclass(slots=True)
class ScoreRequest:
    prompt: list[int]
    allowed_token_ids: list[int] | None    # None: the wrapper's no-options fallback, unrestricted over the vocab
    num_logprobs: int


def is_score_request(body: Any) -> bool:
    """Only the readout wrapper sends this flag; every other completion is served as before."""

    return isinstance(body, dict) and body.get("return_tokens_as_token_ids") is True


def _int_list(value: Any, name: str, *, vocab: int | None) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ScoreError(f'"{name}" must be a nonempty list of token ids')
    out = []
    for t in value:
        if isinstance(t, bool) or not isinstance(t, int):
            raise ScoreError(f'"{name}" must contain only integer token ids')
        if vocab is not None and not 0 <= t < vocab:
            raise ScoreError(f'"{name}" contains a token id outside the model\'s {vocab}-token vocabulary')
        out.append(int(t))
    return out


def parse_score_request(body: dict[str, Any], *, vocab: int | None = None) -> ScoreRequest:
    """Validate the wrapper's exact contract; raise ``ScoreError`` for anything else instead of guessing."""

    if not isinstance(body, dict):
        raise ScoreError("the request body must be a JSON object")
    if body.get("stream"):
        raise ScoreError("scoring does not stream")
    prompt = _int_list(body.get("prompt"), "prompt", vocab=vocab)
    if body.get("max_tokens") != 1:
        raise ScoreError('scoring requires "max_tokens": 1')
    temperature = body.get("temperature", 0)
    if temperature != 0:
        raise ScoreError('scoring requires "temperature": 0')
    if body.get("return_tokens_as_token_ids") is not True:
        raise ScoreError('scoring requires "return_tokens_as_token_ids": true')
    allowed = None
    if "allowed_token_ids" in body:
        allowed = _int_list(body.get("allowed_token_ids"), "allowed_token_ids", vocab=vocab)
        if len(set(allowed)) != len(allowed):
            raise ScoreError('"allowed_token_ids" must not repeat a token id')
    logprobs = body.get("logprobs")
    if isinstance(logprobs, bool) or not isinstance(logprobs, int):
        raise ScoreError(f'"logprobs" must be an integer between 1 and {MAX_LOGPROBS}')
    if not 1 <= logprobs <= MAX_LOGPROBS:
        raise ScoreError(f'"logprobs" must be between 1 and {MAX_LOGPROBS}')
    if allowed is not None and logprobs < min(len(allowed), MAX_LOGPROBS):
        raise ScoreError('"logprobs" must cover at least min(len(allowed_token_ids), 64) entries')
    return ScoreRequest(prompt=prompt, allowed_token_ids=allowed, num_logprobs=logprobs)


def token_id_key(token_id: int) -> str:
    """``return_tokens_as_token_ids``'s placeholder (``vllm/entrypoints/openai/completion/serving.py``)."""

    return f"token_id:{token_id}"


def rank_allowed(logits, allowed_token_ids: list[int] | None,
                 num_logprobs: int = MAX_LOGPROBS) -> tuple[int, list[tuple[int, float]]]:
    """Mask to the allowed set and log-softmax the raw logits (vLLM's ``processed_logprobs`` at temperature 0).

    ``logits`` is a ``(1, vocab)`` float tensor. Returns the argmax id and every allowed id's logprob, sorted by
    logprob descending (vLLM's ``top_logprobs`` order). ``allowed_token_ids=None`` (the wrapper's no-options
    fallback): no masking, ranked over the whole vocabulary's top ``num_logprobs`` instead.
    """

    import torch

    if logits.dim() != 2 or logits.shape[0] != 1:
        raise ValueError("rank_allowed takes one row of vocab-wide logits")
    row = logits[0].float()
    if allowed_token_ids is None:
        logprobs = row.log_softmax(dim=-1)
        values, idx = logprobs.topk(min(num_logprobs, logprobs.shape[-1]))
        ranked = list(zip((int(t) for t in idx.tolist()), values.tolist()))
        return int(row.argmax()), ranked
    idx = torch.tensor(allowed_token_ids, device=logits.device, dtype=torch.long)
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[idx] = False
    masked = row.masked_fill(mask, float("-inf"))
    logprobs = masked.log_softmax(dim=-1)
    values = logprobs[idx].tolist()
    ranked = sorted(zip(allowed_token_ids, values), key=lambda kv: -kv[1])
    return ranked[0][0], ranked


def build_choice(chosen: int, ranked: list[tuple[int, float]], num_logprobs: int) -> dict[str, Any]:
    """``choices[0]``: vLLM's completion logprobs shape, ``return_tokens_as_token_ids`` keys throughout."""

    top = {token_id_key(tid): lp for tid, lp in ranked[:num_logprobs]}
    chosen_lp = next(lp for tid, lp in ranked if tid == chosen)
    return {
        "text": token_id_key(chosen), "index": 0, "finish_reason": "length",
        "logprobs": {"tokens": [token_id_key(chosen)], "token_logprobs": [chosen_lp],
                    "top_logprobs": [top], "text_offset": [0]},
    }
