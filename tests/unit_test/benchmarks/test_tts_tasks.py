from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.tasks.tts import TTS_USAGE_HEADERS, make_tts_send_fn


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 240)
    return buffer.getvalue()


class _FakeResponse:
    status = 200
    headers = {
        "X-Prompt-Tokens": "4",
        "X-Completion-Tokens": "12",
        "X-Engine-Time": "0.5",
    }

    async def read(self) -> bytes:
        return _wav_bytes()


class _FakePostContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict, headers: dict[str, str]) -> _FakePostContext:
        self.posts.append({"url": url, "json": json, "headers": headers})
        return _FakePostContext(_FakeResponse())


def test_tts_send_fn_requests_usage_headers_and_computes_token_throughput() -> None:
    session = _FakeSession()
    send_fn = make_tts_send_fn("s2-pro", "http://localhost:8000/v1/audio/speech")
    sample = SampleInput(
        sample_id="sample-1",
        ref_text="reference",
        ref_audio="/tmp/ref.wav",
        target_text="hello world",
    )

    result = asyncio.run(send_fn(session, sample))

    assert session.posts[0]["headers"] == TTS_USAGE_HEADERS
    assert result.is_success
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 12
    assert result.engine_time_s == 0.5
    assert result.tok_per_s == 24.0
