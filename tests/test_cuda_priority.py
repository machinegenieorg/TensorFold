"""CUDA scheduler priorities: waiting requests start lower-priority-first, and reserved streams stay free for foreground
requests (any machine: a stand-in decoder, no GPU)."""

import threading
import time

import pytest

from tensorfold.cuda.scheduler import Scheduler

WAIT = 10.0


class Decoder:
    """``MultiDecoder``'s scheduling surface: each round gives every live stream one token."""

    def __init__(self):
        self.streams, self.started, self.peak_background = [], [], 0
        self.gate = threading.Event()             # cleared: rounds produce nothing, so a test can fill the queue
        self.gate.set()

    def live(self):
        return len(self.streams)

    def admit(self, s):
        self.started.append(s.prompt[0])
        s.started = time.perf_counter()
        self.streams.append(s)

    def round(self):
        if not self.gate.is_set():                 # held: a round that produces nothing, so admission keeps running
            time.sleep(0.005)
            return []
        done = []
        for s in list(self.streams):
            s.take([7])
            if s.done:
                done.append(s)
        return done

    def finish(self, done):
        for s in done:
            self.streams.remove(s)

    def drop(self):
        out, self.streams = self.streams, []
        return out


def submit(sched, tag, count, priority=0, results=None):
    def go():
        stats = sched.submit([tag], count, None, True, lambda new: False, priority=priority)
        if results is not None:
            results.append((tag, stats.get("rounds")))
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def settle(decoder, n):
    for _ in range(int(WAIT / 0.01)):
        if len(decoder.started) >= n:
            return
        time.sleep(0.01)
    raise AssertionError(f"only {len(decoder.started)} of {n} requests started")


def test_waiting_requests_start_by_priority_then_arrival():
    d = Decoder()
    d.gate.clear()                                 # the first request holds the only stream until the queue is full
    sched = Scheduler(d, max_streams=1)
    threads = [submit(sched, 0, 3)]
    settle(d, 1)
    for tag, prio in ((1, 10), (2, 5), (3, 10), (4, 0), (5, -1)):
        threads.append(submit(sched, tag, 1, prio))
    time.sleep(0.2)                                # all five queued behind the running one
    d.gate.set()
    for t in threads:
        t.join(WAIT)
    assert d.started == [0, 5, 4, 2, 1, 3]         # lower first; within priority 10, the earlier arrival first


def test_reserved_streams_stay_free_for_foreground_requests():
    d = Decoder()
    d.gate.clear()
    sched = Scheduler(d, max_streams=4, reserve=1)
    background = [submit(sched, tag, 5, priority=10) for tag in range(10, 16)]   # six background requests
    time.sleep(0.2)
    assert sorted(d.started) == [10, 11, 12]       # only max_streams - reserve of them start
    fg = submit(sched, 99, 1, priority=0)
    settle(d, 4)
    assert d.started[-1] == 99                     # the foreground request takes the reserved stream at once
    d.gate.set()
    for t in background + [fg]:
        t.join(WAIT)
    assert sorted(d.started) == [10, 11, 12, 13, 14, 15, 99]


def test_without_a_reserve_background_work_can_fill_every_stream():
    d = Decoder()
    d.gate.clear()
    sched = Scheduler(d, max_streams=3)
    threads = [submit(sched, tag, 3, priority=10) for tag in range(3)]
    settle(d, 3)
    assert sorted(d.started) == [0, 1, 2]
    d.gate.set()
    for t in threads:
        t.join(WAIT)


def test_a_failed_background_admission_frees_its_share():
    d = Decoder()
    calls = {"n": 0}
    real = d.admit

    def admit(s):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("stand-in admission failure")
        real(s)
    d.admit = admit
    sched = Scheduler(d, max_streams=2, reserve=1)
    got = []

    def go(tag):
        try:
            got.append(("done", tag, sched.submit([tag], 1, None, True, lambda new: False, priority=10)))
        except RuntimeError:
            got.append(("error", tag, None))
    a = threading.Thread(target=go, args=(1,), daemon=True)
    a.start()
    a.join(WAIT)
    b = threading.Thread(target=go, args=(2,), daemon=True)
    b.start()
    b.join(WAIT)
    assert [g[0] for g in got] == ["error", "done"]   # the failure left no phantom background stream behind
    assert sched.background == set()


@pytest.mark.parametrize("reserve", [-1, 4, 5])
def test_the_reserve_must_leave_background_a_stream(reserve):
    with pytest.raises(ValueError, match="reserve"):
        Scheduler(Decoder(), max_streams=4, reserve=reserve)


def test_the_server_refuses_a_non_integer_priority_and_passes_an_integer_on(tmp_path):
    pytest.importorskip("jinja2")
    import json

    from tests.test_cuda_admission import http_server
    from tests.test_cuda_server_errors import HI, app_for, request

    app = app_for(tmp_path)
    seen = []
    real = app.engine.generate

    def generate(prompt, max_tokens, sampling, on_tokens, draft=True, priority=0):
        seen.append(priority)
        return real(prompt, max_tokens, sampling, on_tokens, draft=draft)
    app.engine.generate = generate
    with http_server(app) as port:
        for bad in ("high", 1.5, True):
            status, _, text = request(port, {"messages": HI, "max_tokens": 4, "priority": bad})
            assert status == 400 and "priority" in json.loads(text)["error"]["message"]
        status, _, _ = request(port, {"messages": HI, "max_tokens": 4, "priority": 10})
        assert status == 200
        status, _, _ = request(port, {"messages": HI, "max_tokens": 4})
        assert status == 200
    assert seen == [10, 0]                             # priority 0 is the default and is not sent


def test_serve_hands_reserve_streams_to_the_cuda_engine(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tensorfold import cli
    from tensorfold.cuda import server

    made = []
    family = SimpleNamespace(title="Test family", model_type="test",
                             package=SimpleNamespace(cuda_engine=lambda *a, **k: made.append(k) or
                                                     SimpleNamespace(max_len=8192)))
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=8192))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts",
                                          "--parallel", "8", "--reserve-streams", "2"])
    assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
    assert made[0]["reserve_streams"] == 2 and made[0]["parallel"] == 8
    alone = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--reserve-streams", "2"])
    with pytest.raises(ValueError, match="--parallel"):
        cli._serve_cuda(alone, family, tmp_path, 8192)
