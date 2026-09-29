"""usage.prompt_tokens_details.cached_tokens reports the prompt tokens an engine resumed from a kept prefix."""

import json

import pytest

from tests.test_cuda_admission import http_server, post
from tests.test_cuda_request_policy import app_for


class Resuming:
    eos = (0,)

    def __init__(self, cached):
        self.cached = cached

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        on_tokens([ord(c) for c in "Hi"])
        return {"generated": 2} if self.cached is None else {"generated": 2, "cached": self.cached}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cached", [None, 0, 7])
def test_usage_reports_the_prompt_tokens_resumed(tmp_path, stream, cached):
    app = app_for(tmp_path)
    app.engine = Resuming(cached)
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Hello"}], "stream": stream,
                                   "stream_options": {"include_usage": True}}, True)
    assert status == 200
    if stream:
        usage = [json.loads(line[5:])["usage"] for line in body.splitlines()
                 if line.startswith("data:") and line != "data: [DONE]" and '"usage"' in line][0]
    else:
        usage = json.loads(body)["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    if cached is None:
        assert "prompt_tokens_details" not in usage
    else:
        assert usage["prompt_tokens_details"] == {"cached_tokens": cached}
