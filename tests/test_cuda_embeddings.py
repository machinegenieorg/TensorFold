"""The CUDA server's /v1/embeddings on a stand-in engine: formats, truncation, dimensions, priorities, refusals."""

import base64
import http.client
import json
import threading
import time

import numpy as np
import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, models, pre_tokenizers, processors

from tensorfold.cuda import embeddings, server
from tensorfold.server.cancellation import RequestCancelled
from tests.test_cuda_admission import http_server

EOS = 1
WORDS = ["[UNK]", "<eos>"] + [f"w{i}" for i in range(60)]


def tokenizer():
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(single="$A <eos>", special_tokens=[("<eos>", EOS)])
    return tok


def text(*numbers):
    return " ".join(f"w{i}" for i in numbers)


def ids(*numbers):
    return [WORDS.index(f"w{i}") for i in numbers]


def row(tokens, width=48):
    """A stand-in pooled row that depends on the text's tokens only, never on the batch."""

    rng = np.random.default_rng(abs(hash(tuple(tokens))) % 2**32)
    return rng.standard_normal(width).astype(np.float32) * 3


class Engine:
    dimensions, vocab, min_dimensions = 48, len(WORDS), 8

    def __init__(self, context=16, batch_tokens=64):
        self.context_window, self.batch_tokens = context, batch_tokens
        self.steps = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail = False

    def embed(self, texts):
        self.gate.wait(10)
        self.steps.append([list(t) for t in texts])
        if self.fail:
            self.fail = False
            raise RuntimeError("simulated device failure")
        return np.stack([row(t) for t in texts])


def app_for(tmp_path, engine=None, aliases=()):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    app = server.App.__new__(server.App)
    app.engine = engine or Engine()
    app.served = "embed-test"
    app.aliases = tuple(aliases)
    app.tok = tokenizer()
    app.template = server.ChatTemplate(tmp_path)
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    app.vision = None
    return app


def call(port, body, path="/v1/embeddings", method="POST"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, body=None if body is None else json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def unit(tokens, dims=48):
    return embeddings.normalize(row(tokens), dims)


@pytest.mark.parametrize("given, expected", [
    (text(3, 4), [ids(3, 4) + [EOS]]),
    ([text(3), text(5, 6, 7)], [ids(3) + [EOS], ids(5, 6, 7) + [EOS]]),
    (ids(3, 4), [ids(3, 4)]),                              # a token array is read as given: no end token added
    ([ids(3), ids(8, 9)], [ids(3), ids(8, 9)]),
])
def test_every_input_form_returns_its_rows_in_order(tmp_path, given, expected):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, body = call(port, {"input": given, "model": "embed-test"})
    assert status == 200 and body["object"] == "list" and body["model"] == "embed-test"
    assert [d["index"] for d in body["data"]] == list(range(len(expected)))
    for d, tokens in zip(body["data"], expected):
        assert d["object"] == "embedding"
        assert np.array(d["embedding"], dtype=np.float32).tobytes() == unit(tokens).tobytes()
    count = sum(map(len, expected))
    assert body["usage"] == {"prompt_tokens": count, "total_tokens": count}


def test_base64_carries_the_same_float32_bits(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        _, floats = call(port, {"input": [text(1, 2), text(3)]})
        _, packed = call(port, {"input": [text(1, 2), text(3)], "encoding_format": "base64"})
    for f, p in zip(floats["data"], packed["data"]):
        assert np.frombuffer(base64.b64decode(p["embedding"]), dtype="<f4").tobytes() == \
            np.array(f["embedding"], dtype=np.float32).tobytes()


def test_dimensions_keep_the_leading_values_renormalized(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, body = call(port, {"input": text(5, 6), "dimensions": 16})
    vector = np.array(body["data"][0]["embedding"], dtype=np.float64)
    full = row(ids(5, 6) + [EOS]).astype(np.float64)
    assert status == 200 and vector.shape == (16,)
    assert abs(np.linalg.norm(vector) - 1) < 1e-6
    assert np.allclose(vector, full[:16] / np.linalg.norm(full[:16]), atol=1e-7)


def test_text_truncation_keeps_the_start_and_the_end_token(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, body = call(port, {"input": [text(*range(10)), text(1, 2)], "truncate_prompt_tokens": 4})
    assert status == 200
    assert app.engine.steps[0] == [ids(0, 1, 2) + [EOS], ids(1, 2) + [EOS]]
    assert body["usage"]["prompt_tokens"] == 7


def test_token_arrays_truncate_to_their_start_and_left_keeps_the_end(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        call(port, {"input": [ids(*range(10))], "truncate_prompt_tokens": 3})
        call(port, {"input": [ids(*range(10))], "truncate_prompt_tokens": 3, "truncation_side": "left"})
        call(port, {"input": text(*range(10)), "truncate_prompt_tokens": 3, "truncation_side": "left"})
    assert app.engine.steps == [[ids(0, 1, 2)], [ids(7, 8, 9)], [ids(8, 9) + [EOS]]]


def test_truncation_to_the_server_limit_and_long_inputs_without_it(tmp_path):
    app = app_for(tmp_path, Engine(context=5))
    with http_server(app) as port:
        ok, body = call(port, {"input": text(*range(9)), "truncate_prompt_tokens": -1})
        long_status, long_body = call(port, {"input": text(*range(9))})
        over_status, over_body = call(port, {"input": text(1), "truncate_prompt_tokens": 6})
    assert ok == 200 and app.engine.steps[0] == [ids(0, 1, 2, 3) + [EOS]]
    assert long_status == 400 and "5-token limit" in long_body["error"]["message"]
    assert over_status == 400 and "exceeds" in over_body["error"]["message"]
    assert len(app.engine.steps) == 1


@pytest.mark.parametrize("body", [
    {}, {"input": []}, {"input": ""}, {"input": [[]]}, {"input": [text(1), ids(2)]}, {"input": 5},
    {"input": [ids(1)[0], "w2"]}, {"input": [[1, 999]]}, {"input": [[1, -1]]}, {"input": [[True, 2]]},
    {"input": text(1), "encoding_format": "fp16"}, {"input": text(1), "dimensions": 4},
    {"input": text(1), "dimensions": 64}, {"input": text(1), "dimensions": "16"},
    {"input": text(1), "truncate_prompt_tokens": 0}, {"input": text(1), "truncate_prompt_tokens": -2},
    {"input": text(1), "truncation_side": "middle"}, {"input": text(1), "priority": "background"},
    {"input": text(1), "priority": True}, {"input": text(1), "model": 7}, [text(1)],
])
def test_malformed_requests_are_refused_before_the_engine(tmp_path, body):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, reply = call(port, body)
    if body == {"input": ""}:                     # an empty string still has the tokenizer's end token
        assert status == 200
        return
    assert status == 400 and reply["error"]["type"] == "invalid_request_error"
    assert app.engine.steps == []


def test_aliases_are_listed_and_named_models_echoed(tmp_path):
    app = app_for(tmp_path, aliases=("nv-embed-v2", "embed-test"))
    with http_server(app) as port:
        _, models = call(port, None, "/v1/models", "GET")
        _, alias = call(port, {"input": text(1), "model": "nv-embed-v2"})
        _, other = call(port, {"input": text(1), "model": "something-else"})
    assert [m["id"] for m in models["data"]] == ["embed-test", "nv-embed-v2"]
    assert alias["model"] == "nv-embed-v2" and other["model"] == "embed-test"


def test_text_routes_refuse_an_embedding_engine_and_embeddings_refuse_a_text_engine(tmp_path):
    app = app_for(tmp_path)
    with http_server(app) as port:
        status, body = call(port, {"messages": [{"role": "user", "content": "w1"}]}, "/v1/chat/completions")
    assert status == 400 and "embeddings only" in body["error"]["message"]
    from tests.test_cuda_request_policy import app_for as text_app

    chat = text_app(tmp_path)
    with http_server(chat) as port:
        status, body = call(port, {"input": "Hi"})
    assert status == 400 and "does not serve embeddings" in body["error"]["message"]
    assert chat.engine.calls == []


def test_a_failed_step_answers_its_requests_and_the_queue_goes_on(tmp_path):
    engine = Engine()
    engine.fail = True
    app = app_for(tmp_path, engine)
    with http_server(app) as port:
        failed, body = call(port, {"input": text(1)})
        ok, _ = call(port, {"input": text(2)})
    assert failed == 500 and "simulated" in body["error"]["message"] and ok == 200


def test_waiting_requests_share_steps_up_to_the_budget(tmp_path):
    engine = Engine(batch_tokens=8)
    queue = embeddings.EmbedQueue(engine, engine.batch_tokens)
    engine.gate.clear()
    first = threading.Thread(target=queue.submit, args=([ids(1, 2)],))
    first.start()
    while not queue.jobs or queue.jobs[0].taken == 0:
        time.sleep(0.01)                          # the first step holds the engine
    results = {}
    threads = [threading.Thread(target=lambda n=n: results.setdefault(n, queue.submit([ids(n)] * 3)))
               for n in (3, 4)]
    for t in threads:
        t.start()
    while len(queue.jobs) < 3:
        time.sleep(0.01)
    engine.gate.set()
    for t in [first, *threads]:
        t.join(10)
    assert engine.steps[0] == [ids(1, 2)]
    assert [len(s) for s in engine.steps[1:]] == [6]            # both waiting requests in one step of 6 tokens
    assert [r.tobytes() for r in results[3]] == [row(ids(3)).tobytes()] * 3


def test_a_query_with_a_lower_priority_number_overtakes_a_bulk_request(tmp_path):
    engine = Engine(batch_tokens=4)
    queue = embeddings.EmbedQueue(engine, engine.batch_tokens)
    engine.gate.clear()
    bulk = threading.Thread(target=queue.submit, args=([ids(n, n) for n in range(10, 20)], 10))
    bulk.start()
    while not queue.jobs or queue.jobs[0].taken == 0:
        time.sleep(0.01)                          # bulk's first step (two texts) is on the engine
    query = threading.Thread(target=queue.submit, args=([ids(1, 2, 3)], 0))
    query.start()
    while len(queue.jobs) < 2:
        time.sleep(0.01)
    engine.gate.set()
    for t in (bulk, query):
        t.join(10)
    assert engine.steps[0] == [ids(10, 10), ids(11, 11)]
    assert engine.steps[1] == [ids(1, 2, 3)]                   # the query goes next, ahead of bulk's other steps
    assert sum(len(s) for s in engine.steps) == 11


def test_an_urgent_step_carries_no_later_priority_texts():
    engine = Engine(batch_tokens=64)
    queue = embeddings.EmbedQueue(engine, engine.batch_tokens)
    engine.gate.clear()
    first = threading.Thread(target=queue.submit, args=([ids(1)], 0))
    first.start()
    while not queue.jobs or queue.jobs[0].taken == 0:
        time.sleep(0.01)
    bulk = threading.Thread(target=queue.submit, args=([ids(n, n) for n in range(10, 16)], 10))
    bulk.start()
    while len(queue.jobs) < 2:
        time.sleep(0.01)
    queries = [threading.Thread(target=queue.submit, args=([ids(n)], 0)) for n in (2, 3)]
    for t in queries:
        t.start()
    while len(queue.jobs) < 4:
        time.sleep(0.01)
    engine.gate.set()
    for t in [first, bulk, *queries]:
        t.join(10)
    assert engine.steps == [[ids(1)], [ids(2), ids(3)], [ids(n, n) for n in range(10, 16)]]


def test_a_request_whose_client_left_is_dropped_from_later_steps():
    engine = Engine(batch_tokens=2)
    queue = embeddings.EmbedQueue(engine, engine.batch_tokens)
    engine.gate.clear()
    gone = threading.Event()
    errors = []

    def leaving():
        try:
            queue.submit([ids(n, n) for n in range(10, 20)], 0, cancelled=gone.is_set)
        except RequestCancelled as exc:
            errors.append(exc)

    worker = threading.Thread(target=leaving)
    worker.start()
    while not queue.jobs or queue.jobs[0].taken == 0:
        time.sleep(0.01)                          # its first step is on the engine
    gone.set()
    worker.join(10)
    engine.gate.set()
    assert errors
    assert [r.tobytes() for r in queue.submit([ids(1)])] == [row(ids(1)).tobytes()]
    assert engine.steps == [[ids(10, 10)], [ids(1)]]           # the step in flight, then the next request's


def test_rows_are_normalized_in_float64_one_row_at_a_time():
    values = np.array([3.0, 4.0, 12.0], dtype=np.float32)
    assert embeddings.normalize(values, 2).tolist() == [0.6000000238418579, 0.800000011920929]
    assert embeddings.normalize(np.zeros(4, dtype=np.float32), 4).tolist() == [0.0] * 4
