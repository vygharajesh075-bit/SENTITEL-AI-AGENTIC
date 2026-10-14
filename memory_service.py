
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any


class MemoryService:
    """Small thread-safe memory store for recent agent runs."""

    def __init__(self, max_items: int = 500):
        self._items = deque(maxlen=max_items)
        self._lock = Lock()

    def remember(
        self,
        telemetry: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "telemetry": dict(telemetry),
            "result": dict(result),
        }

        with self._lock:
            self._items.append(item)

        return item

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))

        with self._lock:
            return list(self._items)[-limit:][::-1]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._items)


memory_service = MemoryService()
