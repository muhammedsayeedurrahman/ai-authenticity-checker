"""Tests for core/metadata.py's check_c2pa() manifest parsing.

The real c2pa-python library needs an actual signed asset on disk to
exercise end-to-end, so these tests fake the library's Reader class to
verify the parsing/classification logic in check_c2pa() itself: AI
generation declared, a tampered (invalid) provenance chain, a
valid/trusted chain with no AI declaration, and the library being
unavailable altogether.
"""

import sys
from unittest.mock import MagicMock

import pytest


def _fake_c2pa_module(*, active_manifest, validation_state, raise_on_open=False):
    """Build a fake `c2pa` module exposing just what check_c2pa() uses."""
    fake = MagicMock()

    class FakeReader:
        def __init__(self, file_path):
            if raise_on_open:
                raise RuntimeError("no embedded manifest")

        def get_active_manifest(self):
            return active_manifest

        def get_validation_state(self):
            return validation_state

    fake.Reader = FakeReader
    return fake


@pytest.fixture()
def _isolate_c2pa_import(monkeypatch):
    """Ensure a real c2pa import (if installed) doesn't leak into tests
    that want a fake, and that the fake doesn't leak into later tests."""
    monkeypatch.delitem(sys.modules, "c2pa", raising=False)
    yield
    monkeypatch.delitem(sys.modules, "c2pa", raising=False)


def test_c2pa_ai_generated_signal(monkeypatch, tmp_path, _isolate_c2pa_import):
    from core.metadata import check_c2pa

    manifest = {
        "claim_generator": "Adobe Firefly/2.0",
        "assertions": [{
            "label": "c2pa.actions",
            "data": {"actions": [
                {"action": "c2pa.created", "digitalSourceType":
                    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"},
            ]},
        }],
    }
    fake = _fake_c2pa_module(active_manifest=manifest, validation_state="Trusted")
    monkeypatch.setitem(sys.modules, "c2pa", fake)

    result = check_c2pa(str(tmp_path / "fake.jpg"))

    assert result["has_c2pa"] is True
    assert result["ai_generated_signal"] is True
    assert result["generator"] == "Adobe Firefly/2.0"
    assert result["trust_boost"] > 0  # increases risk


def test_c2pa_tampered_chain(monkeypatch, tmp_path, _isolate_c2pa_import):
    from core.metadata import check_c2pa

    manifest = {"claim_generator": "some_camera_app/1.0", "assertions": []}
    fake = _fake_c2pa_module(active_manifest=manifest, validation_state="Invalid")
    monkeypatch.setitem(sys.modules, "c2pa", fake)

    result = check_c2pa(str(tmp_path / "fake.jpg"))

    assert result["has_c2pa"] is True
    assert result["ai_generated_signal"] is False
    assert result["valid"] is False
    assert result["trust_boost"] > 0  # tampered chain is suspicious, increases risk


def test_c2pa_valid_authentic_chain(monkeypatch, tmp_path, _isolate_c2pa_import):
    from core.metadata import check_c2pa

    manifest = {
        "claim_generator": "some_camera_app/1.0",
        "assertions": [{
            "label": "c2pa.actions",
            "data": {"actions": [{"action": "c2pa.opened", "digitalSourceType":
                "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"}]},
        }],
    }
    fake = _fake_c2pa_module(active_manifest=manifest, validation_state="Valid")
    monkeypatch.setitem(sys.modules, "c2pa", fake)

    result = check_c2pa(str(tmp_path / "fake.jpg"))

    assert result["has_c2pa"] is True
    assert result["ai_generated_signal"] is False
    assert result["valid"] is True
    assert result["trust_boost"] < 0  # authenticity signal, reduces risk


def test_c2pa_no_manifest_present(monkeypatch, tmp_path, _isolate_c2pa_import):
    from core.metadata import check_c2pa

    fake = _fake_c2pa_module(active_manifest=None, validation_state=None, raise_on_open=True)
    monkeypatch.setitem(sys.modules, "c2pa", fake)

    result = check_c2pa(str(tmp_path / "plain.jpg"))

    assert result["has_c2pa"] is False
    assert result["available"] is True
    assert result["trust_boost"] == 0.0


def test_c2pa_library_not_installed(monkeypatch, tmp_path, _isolate_c2pa_import):
    from core.metadata import check_c2pa

    monkeypatch.setitem(sys.modules, "c2pa", None)  # forces ImportError on `import c2pa`

    result = check_c2pa(str(tmp_path / "plain.jpg"))

    assert result["has_c2pa"] is False
    assert result["available"] is False
    assert result["trust_boost"] == 0.0
