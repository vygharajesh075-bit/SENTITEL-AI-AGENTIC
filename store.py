import os
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import PyMongoError

class AuditStore:
    """MongoDB persistence with a graceful in-memory fallback."""
    def __init__(self):
        uri = os.getenv("MONGODB_URI", "").strip()
        self.db = None
        self.client = None
        self.memory = {"telemetry": [], "incidents": [], "agent_traces": []}
        if uri:
            try:
                self.client = MongoClient(uri, serverSelectionTimeoutMS=3500)
                self.client.admin.command("ping")
                self.db = self.client[os.getenv("MONGODB_DB", "sentinel")]
                print(f"[SENTINEL] MongoDB connected: {self.db.name}")
            except Exception as exc:
                print(f"[SENTINEL] MongoDB unavailable; using in-memory fallback: {exc}")
                self.db = None

    @property
    def persistent(self):
        return self.db is not None

    def insert(self, collection: str, document: dict):
        item = dict(document)
        item.setdefault("stored_at", datetime.now(timezone.utc).isoformat())
        if self.db is not None:
            try:
                self.db[collection].insert_one(item)
                return
            except PyMongoError as exc:
                print(f"[SENTINEL] Mongo insert failed, retaining in memory: {exc}")
        self.memory.setdefault(collection, []).insert(0, item)
        self.memory[collection] = self.memory[collection][:250]

    def recent(self, collection: str, limit: int = 100):
        if self.db is not None:
            try:
                docs = list(self.db[collection].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
                return docs
            except PyMongoError as exc:
                print(f"[SENTINEL] Mongo read failed: {exc}")
        return self.memory.get(collection, [])[:limit]

    def close(self):
        if self.client:
            self.client.close()
