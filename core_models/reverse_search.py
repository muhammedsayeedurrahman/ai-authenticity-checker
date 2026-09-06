"""
Reverse-image-search corroboration (optional, opt-in).

Not a detection signal on its own — cross-references an uploaded image
against the public web via the Bing Visual Search API (Azure Cognitive
Services / Azure AI Vision) to surface "this exact image also appears
at X", useful supporting evidence for a complaint (e.g. proving a photo
is a stolen/reused profile picture, or that a supposedly-original image
is actually a stock photo or screenshot from elsewhere).

Disabled by default in two independent ways, since this sends the
user's uploaded image to a third-party service:
  1. Server-side: no-ops unless BING_VISUAL_SEARCH_API_KEY is set. There
     is no free tier that ships with this project — the caller must
     provision their own Azure resource and key.
  2. Request-side: the API route only calls this when the client
     explicitly opts in per-request (a checkbox in the UI, off by
     default) — it never runs silently as part of a normal analysis.

Not validated against a live Bing endpoint in this codebase (no API key
was available while writing it) — the request/response shape below
follows Microsoft's published Visual Search API v7 documentation, but a
real key is needed to confirm it end-to-end before relying on it for a
demo. Any request failure (missing key, network error, quota, schema
drift) degrades to an empty/unavailable result rather than raising, so
it can never break the analysis it's attached to.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BING_VISUAL_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
_MAX_MATCHES = 10
_REQUEST_TIMEOUT_SEC = 15


def is_configured() -> bool:
    """Whether a provider API key is present — gates the opt-in checkbox
    in the UI so it isn't offered when it can't actually do anything."""
    return bool(os.environ.get("BING_VISUAL_SEARCH_API_KEY"))


def reverse_image_search(
    image_bytes: bytes, filename: str = "image.jpg",
) -> dict[str, Any]:
    """
    Look up an image against the public web via Bing Visual Search.

    Returns a dict that's always safe to embed in an API response, even
    when the provider isn't configured or the request fails:
      {available, provider, matches: [{title, url, host}], match_count,
       error}
    """
    empty: dict[str, Any] = {
        "available": False,
        "provider": "bing_visual_search",
        "matches": [],
        "match_count": 0,
        "error": None,
    }

    api_key = os.environ.get("BING_VISUAL_SEARCH_API_KEY")
    if not api_key:
        return {**empty, "error": "BING_VISUAL_SEARCH_API_KEY not configured"}

    try:
        import requests
    except ImportError:
        return {**empty, "available": True, "error": "requests library not installed"}

    # The API docs require the image part to carry a Content-Type of a
    # recognized image MIME type — a bare (filename, bytes) 2-tuple leaves
    # it unset, which Bing may reject or silently mishandle.
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"

    try:
        response = requests.post(
            _BING_VISUAL_SEARCH_URL,
            headers={"Ocp-Apim-Subscription-Key": api_key},
            files={"image": (filename, image_bytes, content_type)},
            timeout=_REQUEST_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:  # Broad catch: this must never break the caller's analysis
        logger.warning("Reverse image search failed: %s", e)
        return {**empty, "available": True, "error": str(e)}

    matches = _extract_matches(payload)
    return {
        "available": True,
        "provider": "bing_visual_search",
        "matches": matches[:_MAX_MATCHES],
        "match_count": len(matches),
        "error": None,
    }


def _extract_matches(payload: dict[str, Any]) -> list[dict[str, Optional[str]]]:
    """
    Pull page matches out of a Bing Visual Search response.

    Response shape (v7 API): tags[].actions[] where actionType is
    "PagesIncluding" or "VisualSearch" carry a `data.value[]` list of
    matching pages, each with hostPageUrl / name / hostPageDisplayUrl.
    """
    matches: list[dict[str, Optional[str]]] = []
    seen_urls: set[str] = set()

    for tag in payload.get("tags", []) or []:
        for action in tag.get("actions", []) or []:
            if action.get("actionType") not in ("PagesIncluding", "VisualSearch"):
                continue
            for item in (action.get("data") or {}).get("value", []) or []:
                url = item.get("hostPageUrl")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                matches.append({
                    "url": url,
                    "title": item.get("name"),
                    "host": item.get("hostPageDisplayUrl") or url,
                })

    return matches
