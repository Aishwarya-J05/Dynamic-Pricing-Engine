from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import redis


class RedisCache:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = redis.from_url(redis_url, decode_responses=True) if redis_url else None

    def is_enabled(self) -> bool:
        return self.client is not None

    def set_json(self, key: str, payload: dict[str, Any], ttl_seconds: int = 300) -> None:
        if self.client is None:
            return
        serialized = json.dumps(
            {
                "payload": payload,
                "cached_at": datetime.now(UTC).isoformat(),
            }
        )
        self.client.setex(key, ttl_seconds, serialized)

    def get_json(self, key: str) -> dict[str, Any] | None:
        if self.client is None:
            return None
        value = self.client.get(key)
        return json.loads(value) if value else None
