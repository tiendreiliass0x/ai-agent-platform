import hashlib
import json
from typing import Any


def hash_dict(data: dict[str, Any]) -> str:
    normalized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()
