"""
Multi-key fallback system for ProofyX.

Provides round-robin API key rotation with automatic cooldown on failure.
Supports multiple keys per service loaded from environment variables.

Key naming convention:
    SERVICE_KEY_1=...
    SERVICE_KEY_2=...
    SERVICE_KEY_3=...
"""

from __future__ import annotations

import logging
import os
import secrets as _secrets
import threading
import time

logger = logging.getLogger("proofyx.secrets")


class AllKeysCooledDown(Exception):
    """Raised when every key in the pool is on cooldown."""

    def __init__(self, service: str, retry_after: float):
        self.service = service
        self.retry_after = retry_after
        super().__init__(
            f"All keys for '{service}' are on cooldown. "
            f"Earliest retry in {retry_after:.0f}s."
        )


class KeyPool:
    """Round-robin API key pool with automatic fallback on failure.

    Thread-safe. Keys are rotated in order; failed keys are placed on
    a timed cooldown so subsequent calls skip them automatically.
    """

    def __init__(self, service: str, keys: list[str]) -> None:
        if not keys:
            raise ValueError(f"KeyPool for '{service}' requires at least one key")
        self.service = service
        self._keys: tuple[str, ...] = tuple(keys)
        self._index: int = 0
        self._cooldowns: dict[int, float] = {}  # key_index → cooldown_until
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Number of keys in the pool."""
        return len(self._keys)

    def _mask(self, key: str) -> str:
        """Return a masked version of a key for logging."""
        if len(key) <= 8:
            return key[:2] + "***"
        return key[:4] + "***" + key[-4:]

    def get_key(self) -> str:
        """Get the next available key, skipping cooled-down ones.

        Raises:
            AllKeysCooledDown: If every key is currently on cooldown.
        """
        with self._lock:
            now = time.monotonic()
            # Clean expired cooldowns
            self._cooldowns = {
                idx: until
                for idx, until in self._cooldowns.items()
                if until > now
            }

            # Try each key starting from current index
            for _ in range(len(self._keys)):
                idx = self._index % len(self._keys)
                self._index = idx + 1

                if idx not in self._cooldowns:
                    return self._keys[idx]

            # All keys on cooldown — find earliest retry
            earliest = min(self._cooldowns.values()) - now
            raise AllKeysCooledDown(self.service, max(0.0, earliest))

    def report_failure(self, key: str, cooldown_sec: float = 300.0) -> None:
        """Mark a key as failed — skip it for *cooldown_sec* seconds."""
        with self._lock:
            try:
                idx = self._keys.index(key)
            except ValueError:
                return
            until = time.monotonic() + cooldown_sec
            self._cooldowns[idx] = until
            logger.warning(
                "Key %s for '%s' cooled down for %.0fs",
                self._mask(key), self.service, cooldown_sec,
            )

    def report_success(self, key: str) -> None:
        """Clear cooldown for a key on success."""
        with self._lock:
            try:
                idx = self._keys.index(key)
            except ValueError:
                return
            removed = self._cooldowns.pop(idx, None)
            if removed is not None:
                logger.info(
                    "Key %s for '%s' recovered (cooldown cleared)",
                    self._mask(key), self.service,
                )

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the pool (constant-time comparison)."""
        return any(_secrets.compare_digest(key, k) for k in self._keys)

    def available_count(self) -> int:
        """Number of keys not currently on cooldown."""
        with self._lock:
            now = time.monotonic()
            cooled = sum(1 for until in self._cooldowns.values() if until > now)
            return len(self._keys) - cooled

    def __repr__(self) -> str:
        return (
            f"KeyPool(service={self.service!r}, "
            f"total={self.size}, available={self.available_count()})"
        )


def load_key_pool(service: str) -> KeyPool | None:
    """Load keys from env vars following the SERVICE_KEY_1, _2, ... convention.

    Returns None if no keys are found for the given service.

    Examples:
        load_key_pool("PROOFYX_API_KEY")
        → reads PROOFYX_API_KEY_1, PROOFYX_API_KEY_2, ...

        load_key_pool("GOOGLE_VISION_KEY")
        → reads GOOGLE_VISION_KEY_1, GOOGLE_VISION_KEY_2, ...
    """
    keys: list[str] = []
    for i in range(1, 100):  # support up to 99 keys
        value = os.environ.get(f"{service}_{i}")
        if value is None:
            break
        value = value.strip()
        if value:
            keys.append(value)

    if not keys:
        return None

    pool = KeyPool(service=service, keys=keys)
    logger.info(
        "Loaded %d key(s) for '%s'",
        pool.size, service,
    )
    return pool


# ──────────────────────────────────────────────
# Pre-configured service pools (lazy singletons)
# ──────────────────────────────────────────────

_pools: dict[str, KeyPool | None] = {}
_pools_lock = threading.Lock()

# Services that ProofyX supports
KNOWN_SERVICES = (
    "PROOFYX_API_KEY",
    "GOOGLE_VISION_KEY",
    "HF_TOKEN",
    "TWILIO_SID",
    "TWILIO_TOKEN",
)


def get_pool(service: str) -> KeyPool | None:
    """Get or create a key pool for the named service (cached)."""
    with _pools_lock:
        if service not in _pools:
            _pools[service] = load_key_pool(service)
        return _pools[service]


def get_active_pools() -> dict[str, KeyPool]:
    """Return all configured (non-empty) pools. Used for startup logging."""
    result: dict[str, KeyPool] = {}
    for service in KNOWN_SERVICES:
        pool = get_pool(service)
        if pool is not None:
            result[service] = pool
    return result
