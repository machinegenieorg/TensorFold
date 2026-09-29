# Qwen3-Embedding-8B

The `qwen3` family serves Qwen3 dense decoders as last-token embedding models on one NVIDIA GPU, at OpenAI's
`/v1/embeddings`. It runs the prompt forward only: no decoding, no key/value cache kept after a step. Its CUDA
engine reads the checkpoint as shipped (bf16) or converted to MLX affine 4-bit in groups of 64.

## Checkpoint

```bash
tensorfold pull Qwen/Qwen3-Embedding-8B
tensorfold serve Qwen/Qwen3-Embedding-8B --name qwen3-embed-8b --alias nv-embed-v2 --context 2048
```

Tested revision: `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` (15.1 GB). The engine requires the sentence-transformers
pooling file (`1_Pooling/config.json`, last token) and refuses other pooling. Qwen3 checkpoints with attention biases,
scaled rotary positions or sliding windows are refused from `config.json` before any download.

The query instruction is the client's: Qwen3-Embedding is trained with `Instruct: <task>\nQuery:<question>` on
queries and plain documents. The server embeds what it is sent.

## Requests

See [the API page](../api.md#embeddings) for the fields. `truncate_prompt_tokens` keeps a text's start and the
tokenizer's closing `<|endoftext|>`, the token the model pools, as vLLM's default truncation does; an input past
`--context` without it is refused. `dimensions` keeps 32 to 4,096 leading values and normalizes them again.

One worker runs the model. Each step packs the waiting texts of the most urgent priority present, from any requests
in arrival order, up to `--batch-tokens` tokens (default 8,192); a smaller budget shortens a query's wait behind bulk
work (see the measurements). A request's `priority` is an integer, lower first, as with vLLM's
`--scheduling-policy priority`; TensorFold always schedules this way. A live query sent at priority 0 beside bulk
requests at 10 waits for the step in flight, not for the bulk requests.

## Exact vectors

A text's vector has the same bits alone, in any batch, at any position in a step and beside texts of any length,
so batching, priorities and `--batch-tokens` never change a vector. Every kernel is row-local or reads only its own
text:

- projections add each output's inputs in one fixed fp32 chain whatever the row count (`dense.py` for bf16
  weights, the 27B's `qmm_prefill.cu` for 4-bit), and the block shape changes only which block computes an output;
- attention tiles a text's keys from the text's own start (`attention_texts`, beside the 27B's prefill attention),
  exactly as it tiles that text alone;
- norms, rotary and SwiGLU read one row; the residual stream is kept in fp32.

On public text (below), 384 of 384 vectors sent in shuffled batches of 2 to 64 had the bytes of the same text sent
alone, on both bf16 and 4-bit weights. vLLM 0.30.0 on the same GPU returned 12 of 384 identical (least cosine
0.9999049).

## 4-bit conversion

```bash
python -m tensorfold.families.qwen3.convert Qwen/Qwen3-Embedding-8B ./Qwen3-Embedding-8B-4bit
tensorfold serve ./Qwen3-Embedding-8B-4bit --name qwen3-embed-8b
```

The projections become MLX affine 4-bit words in groups of 64 (`--group-size 32` is closer to bf16 and 0.4 GiB
larger), rounded with GPTQ (act-order, static groups) on 128 public WikiText-2 texts of 512 tokens, layer by layer
through the engine's own kernels; the token table keeps 8 bits and the norms bf16. The result is 4.25 GiB (4.66 GiB
at groups of 32); conversion takes under two minutes on one GPU and about 25 GB of its memory.
`--calibration none` rounds to nearest without data and needs no GPU.

## Measurements

One RTX PRO 6000 Blackwell Max-Q (96 GB) in `tensorfold-dev:26.07-xg` (NVIDIA's `pytorch:26.07-py3` base), checkpoint
revision 1d8ad4c, served with `--context 2048`; vLLM 0.30.0 (`vllm/vllm-openai`, `--runner pooling --convert embed
--max-model-len 2048 --max-num-batched-tokens 4096 --scheduling-policy priority --gpu-memory-utilization 0.22`).

Agreement with Hugging Face's fp32 model (transformers, last token, one text at a time) on 53 public texts of 26 to
4,225 tokens (SciFact claims with their instruction, abstracts, and abstracts joined):

| Vectors | Least cosine | Mean cosine | Largest element difference |
| --- | ---: | ---: | ---: |
| TensorFold bf16 | 0.999948 | 0.999979 | 1.7e-3 |
| Hugging Face bf16 | 0.999856 | 0.999903 | 5.3e-3 |
| vLLM bf16 (51 texts under 2,048 tokens) | 0.999901 | 0.999943 | 3.5e-3 |
| TensorFold 4-bit, groups of 64 | 0.981945 | 0.987500 | 2.6e-2 |
| TensorFold 4-bit, groups of 32 | 0.983175 | 0.990046 | 2.4e-2 |

Retrieval on SciFact's 300 test claims over its 5,183 abstracts (`mteb/scifact`, public), ranked by cosine:

| Weights | nDCG@10 | Recall@100 | Top-1 as bf16 | Top-10 overlap with bf16 |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 0.7844 | 0.9733 | | |
| 4-bit, groups of 64 | 0.7958 | 0.9733 | 92.7% | 91.6% |
| 4-bit, groups of 32 | 0.7893 | 0.9800 | 94.3% | 92.2% |
| 4-bit, round to nearest | 0.7851 | 0.9833 | 93.7% | 88.8% |

Throughput with `tools/bench_embeddings.py` (SciFact abstracts cut to the length, one client), tokens a second:

| Tokens a text | Batch | TensorFold bf16 | TensorFold 4-bit | vLLM bf16 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1 / 8 / 32 | 4,995 / 9,297 / 10,648 | 5,238 / 8,991 / 10,537 | 5,269 / 8,796 / 11,890 |
| 500 | 1 / 8 / 32 | 9,539 / 12,099 / 12,254 | 10,281 / 11,599 / 11,665 | 11,870 / 13,312 / 13,493 |
| 1,000 | 1 / 8 / 32 | 11,157 / 12,226 / 12,114 | 10,636 / 11,667 / 11,660 | 12,988 / 13,476 / 13,408 |
| 2,000 | 1 / 8 / 32 | 11,223 / 11,845 / 11,644 | 11,289 / 11,448 / 11,423 | 13,054 / 13,196 / 13,013 |

One instructed query alone: 14.6 ms (bf16), 14.0 ms (4-bit), 14.7 ms (vLLM), median of 50. With two clients sending
25-text bulk requests of about 1,300 tokens at priority 10 and one sending queries at priority 0, a query took
146 ms median with `--batch-tokens 4096` (458 ms at the default 8,192) against vLLM's 401 ms, while bulk ran at
11,181 tok/s against vLLM's 13,603.

GPU memory while serving: 16.2 GB bf16 and 6.1 GB 4-bit for TensorFold (weights 14.1 and 4.3 GiB), 20.8 GB for vLLM
at the settings above.
