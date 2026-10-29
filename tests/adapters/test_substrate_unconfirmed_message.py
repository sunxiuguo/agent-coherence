# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Issue #468: report substrate state when the bump has no usable outcome.

Drive the public commit path with synthetic bytes and a loopback proxy that
replaces the bump's answer. The coordinator either processes that request
first or never receives it: an unusable answer cannot distinguish the two.
"""

from __future__ import annotations

import http.client
import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.substrate import CoordinatedSubstrate, SubstrateCoordinatorSession
from ccs.core.exceptions import COMMIT_UNCONFIRMED_REASON, CommitUnconfirmed, ViewWedged
from ccs.testing.substrate_conformance import ConformanceSubstrate, InMemoryStore
from tests.coordinator_helpers import recorded_version, serve_in_process

REF = "data/shared.txt"
BUMP = "/hooks/post-edit-cas"


@contextmanager
def _replace_bump_answer(
    coordinator_port: int, status: int, answer: bytes, *, forwarded: bool,
) -> Iterator[tuple[int, list[str]]]:
    """Forward real HTTP requests, substituting only the bump's response.

    All sockets are loopback, upstream waits are bounded, and the server is
    closed and joined even when an assertion fails. No client method is mocked.
    """
    routes: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            routes.append(self.path)
            code, response = status, answer
            if self.path != BUMP or forwarded:
                conn = http.client.HTTPConnection("127.0.0.1", coordinator_port, timeout=2)
                try:
                    conn.request("POST", self.path, body, dict(self.headers))
                    upstream = conn.getresponse()
                    received = upstream.read()
                    if self.path != BUMP:
                        code, response = upstream.status, received
                    else:
                        assert upstream.status == 200, "control: the bump reached the coordinator"
                finally:
                    conn.close()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(response)

    proxy = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=proxy.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield proxy.server_port, routes
    finally:
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive(), "proxy did not stop"


@pytest.fixture
def coordinator(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    (tmp_path / "data").mkdir()
    (tmp_path / REF).write_bytes(b"s1")
    server = serve_in_process(tmp_path, "unconfirmed-substrate-message")
    try:
        yield server
    finally:
        server.shutdown()


@pytest.mark.parametrize("converged", [False, True], ids=["landed", "converged"])
@pytest.mark.parametrize("forwarded", [False, True], ids=["not-recorded", "recorded"])
@pytest.mark.parametrize(
    ("status", "answer"),
    [
        (200, b"not JSON"),
        (200, b""),
        (200, b"[]"),
        (200, b"null"),
        (200, b'{"ok": false, "degraded": true}'),
        (200, b'{"ok": false, "reason": "commit_unconfirmed"}'),
        (200, b'{"ok": false}'),
        (200, b'{"ok": false, "reason": ["version_mismatch"]}'),
        (200, b'{"ok": false, "reason": "internal: RuntimeError"}'),
        (503, b"unavailable"),
    ],
    ids=["malformed", "empty", "array", "null", "degraded", "unconfirmed", "absent-reason", "list-reason", "failure-envelope", "http-503"],
)
def test_an_unconfirmed_bump_reports_the_substrate_state_and_keeps_recovery(
    tmp_path: Path, coordinator: CoordinatorHTTPServer, converged: bool, forwarded: bool,
    status: int, answer: bytes,
) -> None:
    store = InMemoryStore()
    store.set(REF, b"s1")
    binding = ConformanceSubstrate(store)
    session = SubstrateCoordinatorSession(tmp_path, managed=("data/**",))
    sub = CoordinatedSubstrate(binding, session)
    _, token = sub.read(REF)
    observed = dict(sub._observed_hash)
    version = recorded_version(coordinator, REF)
    if converged:
        binding.script_unknown(landed=True)

    with _replace_bump_answer(session._endpoint.port, status, answer, forwarded=forwarded) as (port, routes):
        session._endpoint = replace(session._endpoint, port=port)
        with pytest.raises(CommitUnconfirmed) as raised:
            sub.commit(REF, expected_token=token, new_bytes=b"s2")

        exc = raised.value
        assert type(exc) is CommitUnconfirmed
        assert exc.reason == COMMIT_UNCONFIRMED_REASON
        cause = exc.__cause__
        assert type(cause) is CommitUnconfirmed, "the original unknown must remain the cause"
        assert cause.reason == exc.reason
        landed = (
            f"the substrate holds the bytes this write intended for {REF!r} "
            "(converged: which write put them there is not claimed)"
            if converged else f"the substrate write to {REF!r} landed"
        )
        assert str(exc) == f"{landed}, but the coordinator bump is unconfirmed: {cause}"
        assert sub._observed_hash == observed, "an unconfirmed bump must not record success"
        assert routes == ["/hooks/pre-read", BUMP], "neither leg may retry"
        assert recorded_version(coordinator, REF) == version + int(forwarded)
        assert len(binding.cas_calls) == 1
        fresh_bytes, fresh_token = binding.read(REF)
        assert fresh_bytes == b"s2" and fresh_token != token

        # Re-read before deciding whether to retry. A recorded bump permits a
        # no-op; an unrecorded one leaves the existing coordinator-behind wedge.
        if forwarded:
            assert sub.read(REF) == (fresh_bytes, fresh_token)
            assert sub.commit(REF, expected_token=fresh_token, new_bytes=fresh_bytes).noop
        else:
            with pytest.raises(ViewWedged):
                sub.read(REF)
            assert sub.reacquire(REF) == (fresh_bytes, fresh_token)
            with pytest.raises(ViewWedged):
                sub.commit(REF, expected_token=fresh_token, new_bytes=fresh_bytes)
        assert len(binding.cas_calls) == 1, "recovery must not write the bytes again"
        assert routes.count(BUMP) == 1, "recovery must not drive a duplicate bump"
        assert recorded_version(coordinator, REF) == version + int(forwarded)
