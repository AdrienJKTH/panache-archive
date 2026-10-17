import io
import urllib.error
from email.message import Message

import pytest

from panache_archive import web


def failing(*errors: Exception) -> object:
    """A `urlopen` that raises `errors` in turn, then answers."""
    left = list(errors)

    def urlopen(_request: object, timeout: float) -> io.BytesIO:
        assert timeout == web.TIMEOUT_SECONDS
        if left:
            raise left.pop(0)
        return io.BytesIO(b"answer")

    return urlopen


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.com", code, "error", Message(), None)


def test_server_errors_and_lost_connections_are_tried_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = (http_error(502), urllib.error.URLError("reset"), TimeoutError("slow"))
    monkeypatch.setattr("urllib.request.urlopen", failing(*errors))
    waits: list[float] = []

    assert web.read("https://example.com", sleep=waits.append) == b"answer"
    assert waits == [2.0, 4.0, 8.0]


def test_a_client_error_is_raised_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", failing(http_error(429)))
    waits: list[float] = []

    with pytest.raises(urllib.error.HTTPError, match="429"):
        web.read("https://example.com", sleep=waits.append)
    assert waits == []


@pytest.mark.parametrize(
    "error", [http_error(503), urllib.error.URLError("reset")], ids=["5xx", "connection"]
)
def test_the_last_failure_is_raised(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr("urllib.request.urlopen", failing(*[error] * web.ATTEMPTS))
    waits: list[float] = []

    with pytest.raises(type(error)):
        web.read("https://example.com", sleep=waits.append)
    assert len(waits) == web.ATTEMPTS - 1
