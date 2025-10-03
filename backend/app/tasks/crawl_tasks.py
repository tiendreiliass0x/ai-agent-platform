from typing import List, Dict
from urllib.parse import urljoin, urlparse
import re
import requests
from bs4 import BeautifulSoup

from app.celery_app import celery_app
from app.core.config import settings

try:
    from firecrawl import FirecrawlApp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FirecrawlApp = None  # type: ignore


def _same_origin(base: str, href: str) -> bool:
    try:
        b = urlparse(base)
        u = urlparse(urljoin(base, href))
        return u.scheme in ("http", "https") and (u.netloc == b.netloc)
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = re.sub(r"/+$", "", parsed.path)
    normalized = parsed._replace(fragment="", query=parsed.query, path=path).geturl()
    return normalized


def _basic_discover(root_url: str, max_pages: int, update=None) -> Dict:
    seen = set()
    queue: List[str] = []
    urls: List[str] = []
    visited = 0

    # Try sitemap.xml first
    try:
        sitemap_url = urljoin(root_url, "/sitemap.xml")
        res = requests.get(sitemap_url, timeout=10)
        if res.ok and "<urlset" in res.text:
            locs = re.findall(r"<loc>(.*?)</loc>", res.text)
            for loc in locs:
                if _same_origin(root_url, loc):
                    u = _normalize_url(loc)
                    if u not in seen:
                        seen.add(u)
                        urls.append(u)
                        if update:
                            update(discovered=len(urls), visited=visited, urls=urls)
                        if len(urls) >= max_pages:
                            return {"urls": urls, "visited": visited, "discovered": len(urls)}
    except Exception:
        pass

    if len(urls) == 0:
        queue.append(root_url)

    while queue and len(urls) < max_pages:
        current = _normalize_url(queue.pop(0))
        if current in seen:
            continue
        seen.add(current)
        visited += 1
        try:
            res = requests.get(current, timeout=10)
            if not res.ok or "text/html" not in res.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if not href:
                    continue
                if not _same_origin(root_url, href):
                    continue
                u = _normalize_url(urljoin(current, href))
                if u not in seen and u not in queue:
                    queue.append(u)
                    if u not in urls:
                        urls.append(u)
                        if update:
                            update(discovered=len(urls), visited=visited, urls=urls)
                        if len(urls) >= max_pages:
                            break
        except Exception:
            continue

    return {"urls": urls, "visited": visited, "discovered": len(urls)}


def _firecrawl_discover(root_url: str, max_pages: int, update=None) -> Dict:
    # Prefer firecrawl-py if available
    if not (settings.FIRECRAWL_API_KEY and FirecrawlApp is not None):
        raise RuntimeError("Firecrawl not configured")

    app = FirecrawlApp(api_key=settings.FIRECRAWL_API_KEY)  # type: ignore
    urls: List[str] = []

    # The firecrawl client API may change; we try to request same-origin links.
    # Fallback exceptions will be caught by the caller.
    try:
        # Hypothetical API: crawl with link discovery
        # Replace with the correct firecrawl method/signature in your environment.
        result = app.crawl_url(  # type: ignore[attr-defined]
            root_url,
            params={
                "limit": max_pages,
                "pageOptions": {"onlySameDomain": True},
                "formats": ["links"],
            },
        )
        # Attempt to extract URLs from result
        if isinstance(result, dict):
            candidates = result.get("links") or result.get("urls") or []
            for u in candidates:
                if isinstance(u, str) and _same_origin(root_url, u):
                    urls.append(_normalize_url(u))
        elif isinstance(result, list):
            for u in result:
                if isinstance(u, str) and _same_origin(root_url, u):
                    urls.append(_normalize_url(u))
        urls = list(dict.fromkeys(urls))[:max_pages]
        if update:
            update(discovered=len(urls), visited=0, urls=urls)
        return {"urls": urls, "visited": 0, "discovered": len(urls)}
    except Exception as e:
        raise RuntimeError(f"Firecrawl error: {e}")


@celery_app.task(name="app.tasks.discover_urls", bind=True)
def discover_urls(self, root_url: str, max_pages: int = 50, provider: str = "firecrawl") -> Dict:
    """Discover same-origin URLs for a given root using Firecrawl (default) with basic fallback.

    Emits progress via task.update_state with PROGRESS meta.
    """

    def progress(**meta):
        self.update_state(state="PROGRESS", meta=meta)

    try:
        if provider == "firecrawl":
            try:
                return _firecrawl_discover(root_url, max_pages, update=progress)
            except Exception:
                # fallback to basic
                return _basic_discover(root_url, max_pages, update=progress)
        else:
            return _basic_discover(root_url, max_pages, update=progress)
    except Exception as e:
        # Return structured error; Celery will set state to FAILURE
        raise e

