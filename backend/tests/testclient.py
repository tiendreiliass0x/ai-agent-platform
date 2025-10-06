"""Local TestClient wrapper suppressing httpx app shortcut deprecation warnings."""

from __future__ import annotations

import warnings

from fastapi.testclient import TestClient as _FastAPITestClient

warnings.filterwarnings(
    "ignore",
    r"The 'app' shortcut is now deprecated.*",
    DeprecationWarning,
)


class TestClient(_FastAPITestClient):
    """Re-export FastAPI's TestClient with the deprecation warning suppressed."""

    __test__ = False

    def __init__(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                r"The 'app' shortcut is now deprecated.*",
                DeprecationWarning,
            )
            super().__init__(*args, **kwargs)
