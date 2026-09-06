"""Tests for core_models/reverse_search.py.

No live Bing API key is available in this environment, so these tests
mock `requests.post` to verify the response-parsing logic against the
shape documented for the Bing Visual Search API v7, and verify the
module degrades gracefully (never raises) when unconfigured or when the
request fails.
"""

from unittest.mock import MagicMock, patch

from core_models.reverse_search import is_configured, reverse_image_search


def test_not_configured_without_api_key(monkeypatch):
    monkeypatch.delenv("BING_VISUAL_SEARCH_API_KEY", raising=False)

    assert is_configured() is False

    result = reverse_image_search(b"fake-bytes", "photo.jpg")
    assert result["available"] is False
    assert result["matches"] == []
    assert result["error"]


def test_configured_when_api_key_set(monkeypatch):
    monkeypatch.setenv("BING_VISUAL_SEARCH_API_KEY", "test-key")
    assert is_configured() is True


def test_parses_matching_pages(monkeypatch):
    monkeypatch.setenv("BING_VISUAL_SEARCH_API_KEY", "test-key")

    fake_payload = {
        "tags": [{
            "actions": [{
                "actionType": "PagesIncluding",
                "data": {"value": [
                    {"hostPageUrl": "https://example.com/a", "name": "Page A",
                     "hostPageDisplayUrl": "example.com/a"},
                    {"hostPageUrl": "https://example.com/b", "name": "Page B",
                     "hostPageDisplayUrl": "example.com/b"},
                    # duplicate URL should be deduped
                    {"hostPageUrl": "https://example.com/a", "name": "Page A again"},
                ]},
            }],
        }],
    }

    fake_response = MagicMock()
    fake_response.json.return_value = fake_payload
    fake_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=fake_response) as mock_post:
        result = reverse_image_search(b"fake-bytes", "photo.jpg")

    assert mock_post.called
    assert result["available"] is True
    assert result["error"] is None
    assert result["match_count"] == 2
    urls = {m["url"] for m in result["matches"]}
    assert urls == {"https://example.com/a", "https://example.com/b"}


def test_request_failure_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("BING_VISUAL_SEARCH_API_KEY", "test-key")

    with patch("requests.post", side_effect=ConnectionError("network down")):
        result = reverse_image_search(b"fake-bytes", "photo.jpg")

    assert result["available"] is True
    assert result["matches"] == []
    assert "network down" in result["error"]
