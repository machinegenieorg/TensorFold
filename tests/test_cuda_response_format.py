"""response_format on the CUDA server: parsing, schemas compiled or refused, and the grammar's rows (no GPU)."""

import json
import threading

import pytest

from tensorfold.cuda import grammar, server
from tensorfold.server.errors import RequestError
from tests.test_cuda_admission import http_server, post
from tests.test_cuda_request_policy import TextTokenizer

SCHEMA = {"type": "object", "properties": {"k": {"type": "integer"}, "tag": {"type": "string", "enum": ["a", "bb"]}},
          "required": ["k"], "additionalProperties": False}


# -- parsing (no xgrammar) -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body, want", [
    ({}, None),
    ({"response_format": None}, None),
    ({"response_format": {"type": "text"}}, None),
    ({"response_format": {"type": "json_object"}}, grammar.Spec("json")),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "v", "schema": SCHEMA, "strict": True}}},
     grammar.Spec("json_schema", json.dumps(SCHEMA))),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": json.dumps(SCHEMA)}}},
     grammar.Spec("json_schema", json.dumps(SCHEMA))),
    ({"guided_json": SCHEMA}, grammar.Spec("json_schema", json.dumps(SCHEMA), "guided_json")),
    ({"structured_outputs": {"json": SCHEMA}}, grammar.Spec("json_schema", json.dumps(SCHEMA), "structured_outputs")),
    ({"structured_outputs": {"json": None, "regex": None}}, None),
])
def test_request_spec_reads_openai_and_vllm_fields(body, want):
    assert grammar.request_spec(body) == want


@pytest.mark.parametrize("body, words", [
    ({"response_format": "json"}, "must be an object"),
    ({"response_format": {"type": "yaml"}}, "text, json_object or json_schema"),
    ({"response_format": {"type": "json_schema"}}, "needs json_schema.schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {"name": "v"}}}, "needs json_schema.schema"),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": "{nope"}}}, "not valid JSON"),
    ({"response_format": {"type": "json_schema", "json_schema": {"schema": [1, 2]}}}, "JSON schema object"),
    ({"guided_regex": "[a-z]+"}, "guided_regex is not supported"),
    ({"guided_choice": ["a", "b"]}, "guided_choice is not supported"),
    ({"structured_outputs": {"regex": "[a-z]+"}}, "structured_outputs regex is not supported"),
    ({"structured_outputs": "json"}, "must be an object"),
])
def test_request_spec_refuses_malformed_requests(body, words):
    with pytest.raises(RequestError, match=words):
        grammar.request_spec(body)


# -- compiling and the grammar's rows (xgrammar, CPU) -----------------------------------------------------------------
THINK_END = 127             # the toy vocabulary's </think>


@pytest.fixture(scope="module")
def grammars():
    """A toy vocabulary: token t is chr(t) (as TextTokenizer encodes), token 0 the stop token."""

    xgr = pytest.importorskip("xgrammar")
    pytest.importorskip("torch")
    vocab = [""] + [chr(t) for t in range(1, 128)]
    info = xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, vocab_size=128, stop_token_ids=[0])
    return grammar.Grammars(info, think_end=THINK_END)


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def _allowed(logits, row: int) -> set[int]:
    import torch

    return set(torch.nonzero(logits[row] > float("-inf")).flatten().tolist())


def _expected(grammars, prefix: list[int]) -> set[int]:
    """The tokens a fresh matcher allows after ``prefix``."""

    m = grammars.xgr.GrammarMatcher(grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))))
    for t in prefix:
        assert m.accept_token(t)
    bits = grammars.xgr.allocate_token_bitmask(1, 128)
    m.fill_next_token_bitmask(bits, 0)
    return {t for t in range(128) if (int(bits[0, t // 32]) >> (t % 32)) & 1}


def test_schemas_compile_or_are_refused_with_the_reason(grammars):
    assert grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))) is not None
    obj = grammars.constraint(grammars.compile(grammar.Spec("json")))            # json_object: any object only
    assert obj.admissible(_ids('{"any": [1, "x"]}') + [0]) == _ids('{"any": [1, "x"]}')
    assert obj.admissible(_ids("[1]")) == []
    for bad, words in (({"type": "nonsense"}, 'Unsupported type "nonsense"'),
                       ({"$ref": "#/definitions/missing"}, "definitions/missing"),
                       ({"type": "string", "pattern": "("}, "parenthesis")):
        with pytest.raises(RequestError, match="the JSON schema cannot be enforced") as err:
            grammars.compile(grammar.Spec("json_schema", json.dumps(bad), "guided_json"))
        message = str(err.value)
        assert message.startswith("guided_json: ") and words in message and ".cc:" not in message, message


def test_rows_are_masked_by_their_paths_and_the_state_stays_committed(grammars):
    import torch

    c = grammars.constraint(grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))))
    committed = _ids('{"k":')
    c.advance(committed)
    drafts = _ids("12}")
    logits = torch.randn(4, 128)
    masked = c.mask(logits, drafts)
    assert masked is not logits
    for r in range(4):
        want = _expected(grammars, committed + drafts[:r])
        assert _allowed(masked, r) == want, r
        assert torch.equal(masked[r][sorted(want)], logits[r][sorted(want)])      # allowed logits keep their bits
    assert 0 in _allowed(masked, 3) and 0 not in _allowed(masked, 2)             # the stop token once complete
    # the matcher is back at the committed tokens: the same masks again, then a partial keep
    assert torch.equal(c.mask(logits, drafts), masked)
    c.advance(_ids("1"))
    again = c.mask(logits[1:], drafts[1:])
    assert torch.equal(again, masked[1:])
    # a rejected draft: its row cannot choose it, the rows after it are left as they are
    bad = c.mask(logits[1:], _ids("x}"))
    assert ord("x") not in _allowed(bad, 0) and torch.equal(bad[1:], logits[2:])


def test_admissible_drops_rejected_drafts_and_the_stop_token(grammars):
    c = grammars.constraint(grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))))
    assert c.admissible(_ids('{"k":1')) == _ids('{"k":1')
    assert c.admissible(_ids('{"x":1')) == _ids('{"')
    assert c.admissible(_ids('{"k":1}') + [0, 5]) == _ids('{"k":1}')             # the reply ends at the stop token
    assert c.admissible([]) == []
    c.advance(_ids('{"k":1}'))
    assert not c.finished and c.admissible([0]) == []
    c.advance([0])
    assert c.finished
    c.advance([5])                                                                 # nothing follows the stop token
    assert c.finished
    with pytest.raises(RuntimeError, match="rejected chosen token"):
        fresh = grammars.constraint(grammars.compile(grammar.Spec("json")))
        fresh.advance(_ids("x"))


def test_with_thinking_the_grammar_starts_after_think_end(grammars):
    import torch

    c = grammars.constraint(grammars.compile(grammar.Spec("json_schema", json.dumps(SCHEMA))), after_think=True)
    logits = torch.randn(3, 128)
    assert c.mask(logits, _ids("ab")) is logits and c.admissible(_ids("x!")) == _ids("x!")
    masked = c.mask(logits, [ord("a"), THINK_END])
    assert torch.equal(masked[:2], logits[:2]) and _allowed(masked, 2) == _expected(grammars, [])
    assert c.admissible([ord("a"), THINK_END, ord("x")]) == [ord("a"), THINK_END]
    c.advance(_ids("hmm") + [THINK_END])
    assert c.active and not c.finished and _allowed(c.mask(logits[:1]), 0) == _expected(grammars, [])


# -- the server --------------------------------------------------------------------------------------------------------
class GrammarEngine:
    """Replies with ``content`` through the constraint's mask (any disallowed token would fail the reply)."""

    eos = (0,)

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
        import torch

        self.calls.append({"draft": draft, "constraint": constraint})
        for t in _ids(self.content) + [0]:
            if constraint is not None:
                logits = torch.zeros(1, 128)
                logits[0, t] = 1.0
                t = int(constraint.mask(logits).argmax())
                constraint.advance([t])
            if on_tokens([t]) or t == 0:
                break
        return {}


class PlainEngine(GrammarEngine):
    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.calls.append({"draft": draft})
        on_tokens(_ids(self.content))
        return {}


def _app(tmp_path, engine, grammars=None):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}assistant:"}))
    app = server.App.__new__(server.App)
    app.engine = engine
    app.served = "fake-cuda"
    app.tok = TextTokenizer()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking = False
    app.sampling = {"temperature": 0.0}
    app.max_tokens = 64
    app.native_context_window = app.context_window = 0
    app.lock = threading.Lock()
    if grammars is not None:
        app.grammars = grammars
    return app


def _body(stream, **extra):
    return {"messages": [{"role": "user", "content": "Hi"}], "stream": stream, **extra}


def _content(stream: bool, text: str) -> str:
    if not stream:
        return json.loads(text)["choices"][0]["message"]["content"]
    parts = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            parts += [c["delta"].get("content") or "" for c in chunk.get("choices", [])]
    return "".join(parts)


@pytest.mark.parametrize("stream", [False, True])
def test_a_schema_reaches_the_engine_as_a_fresh_constraint(tmp_path, grammars, stream):
    engine = GrammarEngine('{"k":42}')
    app = _app(tmp_path, engine, grammars)
    rf = {"type": "json_schema", "json_schema": {"name": "v", "schema": SCHEMA}}
    with http_server(app) as port:
        for _ in range(2):
            status, text = post(port, _body(stream, response_format=rf), True)
            assert status == 200 and json.loads(_content(stream, text)) == {"k": 42}
        status, text = post(port, _body(stream, response_format={"type": "json_object"}, draft=False), True)
        assert status == 200
    first, second, third = (call["constraint"] for call in engine.calls)
    assert isinstance(first, grammar.Constraint) and first is not second and first.finished and second.finished
    assert third.finished and engine.calls[2]["draft"] is False and first.active


@pytest.mark.parametrize("stream", [False, True])
def test_absent_or_text_response_format_calls_the_engine_as_before(tmp_path, stream):
    engine = PlainEngine("Hello")
    app = _app(tmp_path, engine)                   # no grammar compiler: none is built
    with http_server(app) as port:
        for extra in ({}, {"response_format": {"type": "text"}}):
            status, text = post(port, _body(stream, **extra), True)
            assert status == 200 and _content(stream, text) == "Hello"
    assert engine.calls == [{"draft": True}, {"draft": True}] and getattr(app, "grammars", None) is None


@pytest.mark.parametrize("stream", [False, True])
def test_bad_schemas_and_engines_without_grammars_are_refused_before_generating(tmp_path, grammars, stream):
    engine = GrammarEngine("{}")
    app = _app(tmp_path, engine, grammars)
    plain = PlainEngine("{}")
    plain_app = _app(tmp_path, plain, grammars)
    bad = {"type": "json_schema", "json_schema": {"name": "v", "schema": {"type": "nonsense"}}}
    with http_server(app) as port:
        for body, words in ((_body(stream, response_format=bad), 'Unsupported type "nonsense"'),
                            (_body(stream, response_format={"type": "xml"}), "text, json_object or json_schema"),
                            (_body(stream, guided_regex="a+"), "guided_regex is not supported")):
            status, text = post(port, body, True)
            assert status == 400 and words in json.loads(text)["error"]["message"], text
    with http_server(plain_app) as port:
        status, text = post(port, _body(stream, response_format={"type": "json_object"}), True)
        assert status == 400 and "does not enforce structured output" in json.loads(text)["error"]["message"]
    assert engine.calls == [] and plain.calls == []


def test_the_first_structured_request_builds_the_compiler_from_the_checkpoint(tmp_path, monkeypatch):
    built = []

    def fake(model_dir, vocab, stop_ids):
        built.append((str(model_dir), vocab, stop_ids))
        return "compiler"

    monkeypatch.setattr(grammar, "for_model", fake)
    app = _app(tmp_path, GrammarEngine(""))
    app.model_dir = tmp_path
    with pytest.raises(RequestError, match="config.json vocab_size"):
        app._grammars()
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 128}, "vocab_size": 7}))
    assert app._grammars() == "compiler" and app._grammars() == "compiler"
    assert built == [(str(tmp_path), 128, (0,))]
