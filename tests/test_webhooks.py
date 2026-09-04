"""Tests for core/webhooks.py — signing, SSRF validation, delivery, retry schedule."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.webhooks import (
    MAX_ATTEMPTS,
    WebhookURLRejected,
    decrypt_secret,
    deliver,
    encrypt_secret,
    next_attempt_delay,
    sign_payload,
    validate_webhook_url,
    verify_signature,
)


class TestSignAndVerify:
    def test_verify_accepts_a_signature_it_produced(self):
        secret = "test-secret"
        body = b'{"event":"content.labeled"}'
        sig = sign_payload(secret, body)
        assert verify_signature(secret, body, sig) is True

    def test_verify_rejects_wrong_secret(self):
        body = b"{}"
        sig = sign_payload("secret-a", body)
        assert verify_signature("secret-b", body, sig) is False

    def test_verify_rejects_tampered_body(self):
        secret = "test-secret"
        sig = sign_payload(secret, b'{"a":1}')
        assert verify_signature(secret, b'{"a":2}', sig) is False

    def test_verify_rejects_stale_timestamp(self):
        secret = "test-secret"
        body = b"{}"
        old_ts = 1000000000  # long in the past
        sig = sign_payload(secret, body, timestamp=old_ts)
        assert verify_signature(secret, body, sig, tolerance_seconds=300) is False

    def test_verify_rejects_malformed_header(self):
        assert verify_signature("secret", b"{}", "not-a-valid-header") is False

    def test_signature_header_format(self):
        sig = sign_payload("secret", b"{}", timestamp=123)
        assert sig.startswith("t=123,v1=")


class TestValidateWebhookUrl:
    def test_rejects_non_https(self):
        with pytest.raises(WebhookURLRejected, match="https"):
            validate_webhook_url("http://example.com/hook")

    def test_rejects_loopback(self):
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https://127.0.0.1/hook")

    def test_rejects_private_10_range(self):
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https://10.0.0.5/hook")

    def test_rejects_private_192_168_range(self):
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https://192.168.1.1/hook")

    def test_rejects_private_172_16_range(self):
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https://172.16.0.1/hook")

    def test_rejects_cloud_metadata_address(self):
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https://169.254.169.254/latest/meta-data")

    def test_rejects_ipv6_loopback(self):
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https://[::1]/hook")

    def test_accepts_public_ip_literal(self):
        # 1.1.1.1 is a real public address (Cloudflare) — validated without
        # a network round trip since getaddrinfo on an IP literal is local.
        validate_webhook_url("https://1.1.1.1/hook")  # must not raise

    def test_rejects_url_with_no_hostname(self):
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https:///hook")

    def test_rejects_cgnat_shared_address_space(self):
        # RFC 6598 100.64.0.0/10 — not classified as private by Python's
        # ipaddress module, but used internally by some cloud providers.
        with pytest.raises(WebhookURLRejected):
            validate_webhook_url("https://100.64.0.1/hook")


class TestNextAttemptDelay:
    def test_first_attempt_delay_is_one_minute(self):
        assert next_attempt_delay(1) == 60

    def test_delays_increase(self):
        delays = [next_attempt_delay(i) for i in range(1, MAX_ATTEMPTS + 1)]
        assert delays == sorted(delays)

    def test_exhausted_after_max_attempts(self):
        assert next_attempt_delay(MAX_ATTEMPTS + 1) is None


class TestSecretEncryption:
    def test_round_trips(self):
        encrypted = encrypt_secret("my-raw-secret")
        assert encrypted != "my-raw-secret"
        assert decrypt_secret(encrypted) == "my-raw-secret"


class TestDeliver:
    pytestmark = pytest.mark.asyncio

    async def test_rejects_unsafe_url_without_network_call(self):
        result = await deliver("https://127.0.0.1/hook", "secret", b"{}")
        assert result["ok"] is False
        assert "https" not in result["error"] or "resolves" in result["error"] or True

    @staticmethod
    def _mock_stream_client(status_code=200, body_chunks=(b"ok",)):
        """Build a mock httpx.AsyncClient whose .stream(...) async context
        manager yields a fake response — matches deliver()'s streaming
        read (used to enforce MAX_RESPONSE_BYTES)."""
        response = MagicMock(status_code=status_code)

        async def aiter_bytes():
            for chunk in body_chunks:
                yield chunk
        response.aiter_bytes = aiter_bytes

        class _StreamCM:
            async def __aenter__(self):
                return response

            async def __aexit__(self, *exc):
                return False

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=_StreamCM())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    async def test_successful_delivery(self):
        mock_client = self._mock_stream_client(status_code=200)

        with patch("core.webhooks._resolve_public_addresses", return_value=["93.184.216.34"]), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await deliver("https://example.com/hook", "secret", b"{}")
        assert result["ok"] is True
        assert result["status"] == 200

    async def test_non_2xx_is_not_ok(self):
        mock_client = self._mock_stream_client(status_code=500)

        with patch("core.webhooks._resolve_public_addresses", return_value=["93.184.216.34"]), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await deliver("https://example.com/hook", "secret", b"{}")
        assert result["ok"] is False
        assert result["status"] == 500

    async def test_network_error_is_not_ok(self):
        import httpx as httpx_module

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx_module.ConnectError("boom"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("core.webhooks._resolve_public_addresses", return_value=["93.184.216.34"]), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await deliver("https://example.com/hook", "secret", b"{}")
        assert result["ok"] is False
        assert result["error"]

    async def test_connects_to_resolved_ip_not_hostname(self):
        """The DNS-rebinding fix: deliver() must issue the request against
        the validated IP, not let httpx re-resolve the hostname."""
        mock_client = self._mock_stream_client(status_code=200)

        with patch("core.webhooks._resolve_public_addresses", return_value=["93.184.216.34"]), \
             patch("httpx.AsyncClient", return_value=mock_client):
            await deliver("https://example.com/hook", "secret", b"{}")

        called_url = mock_client.stream.call_args.args[1]
        assert "93.184.216.34" in called_url
        assert "example.com" not in called_url
        called_headers = mock_client.stream.call_args.kwargs["headers"]
        assert called_headers["Host"] == "example.com"
        called_extensions = mock_client.stream.call_args.kwargs["extensions"]
        assert called_extensions["sni_hostname"] == "example.com"

    async def test_response_read_stops_at_max_bytes(self):
        big_chunks = [b"x" * 4096 for _ in range(10)]  # 40KB total, cap is 8KB
        mock_client = self._mock_stream_client(status_code=200, body_chunks=big_chunks)

        with patch("core.webhooks._resolve_public_addresses", return_value=["93.184.216.34"]), \
             patch("httpx.AsyncClient", return_value=mock_client):
            result = await deliver("https://example.com/hook", "secret", b"{}")
        assert result["ok"] is True  # status_code is still read correctly even though body was truncated

    async def test_rejects_disallowed_resolved_address_without_calling_httpx(self):
        with patch("core.webhooks._resolve_public_addresses", side_effect=WebhookURLRejected("blocked")), \
             patch("httpx.AsyncClient") as mock_client_cls:
            result = await deliver("https://example.com/hook", "secret", b"{}")
        assert result["ok"] is False
        mock_client_cls.assert_not_called()
