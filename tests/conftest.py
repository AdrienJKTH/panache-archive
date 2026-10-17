from pathlib import Path

import pytest

from helpers import FakeAds


@pytest.fixture
def fake_ads(tmp_path: Path) -> FakeAds:
    source = tmp_path / "ads"
    source.mkdir()
    return FakeAds(workdir=source)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never reach the ADS or Open-Meteo: a forgotten fake fails loudly."""

    def refuse(*args: object, **_: object) -> None:
        raise RuntimeError(f"a test tried to reach the network: {args}")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    monkeypatch.setattr("cdsapi.Client", refuse)
