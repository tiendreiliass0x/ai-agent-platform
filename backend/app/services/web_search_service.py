"""
Web Search Service - Controlled web search with site restrictions and budget caps
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
from urllib.parse import quote_plus


@dataclass
class SearchResult:
    """Web search result"""
    title: str
    url: str
    snippet: str
    domain: str
    date: Optional[datetime] = None
    rank: int = 0


class WebSearchService:
    """Controlled web search with budget limits and site whitelisting"""

    def __init__(self):
        # Search provider configuration
        self.search_api_key = os.getenv("SERP_API_KEY")
        self.search_engine = os.getenv("SEARCH_ENGINE", "google")

        # Budget and rate limiting
        self.max_calls_per_hour = 100
        self.max_calls_per_agent = 10
        self.default_timeout = 8

        # Call tracking (in production, use Redis or database)
        self.call_tracker = {}

    async def search(
        self,
        query: str,
        site_whitelist: List[str] = None,
        max_results: int = 5,
        timeout_seconds: int = None,
        agent_id: Optional[int] = None
    ) -> List[SearchResult]:
        """Perform controlled web search with site restrictions"""

        timeout = timeout_seconds or self.default_timeout

        # Check budget limits
        if agent_id and not self._check_budget(agent_id):
            raise Exception("Agent has exceeded web search budget")

        # Build search query with site restrictions
        search_query = self._build_search_query(query, site_whitelist)

        try:
            # Perform search based on provider
            if self.search_api_key:
                results = await self._search_with_api(search_query, max_results, timeout)
            else:
                # Fallback to mock results for development
                results = self._mock_search_results(search_query, max_results)

            # Track usage
            if agent_id:
                self._track_usage(agent_id)

            # Filter and validate results
            filtered_results = self._filter_results(results, site_whitelist)

            return filtered_results[:max_results]

        except asyncio.TimeoutError:
            print(f"Web search timeout after {timeout}s")
            return []
        except Exception as e:
            print(f"Web search error: {e}")
            return []

    def _build_search_query(self, query: str, site_whitelist: List[str] = None) -> str:
        """Build search query with site restrictions"""

        if not site_whitelist:
            return query

        # Add site: filters for whitelisted domains
        site_filters = " OR ".join([f"site:{domain}" for domain in site_whitelist])

        return f"{query} ({site_filters})"

    async def _search_with_api(
        self,
        query: str,
        max_results: int,
        timeout: int
    ) -> List[SearchResult]:
        """Search using SerpAPI or similar service"""

        if not self.search_api_key:
            return self._mock_search_results(query, max_results)

        # SerpAPI parameters
        params = {
            "api_key": self.search_api_key,
            "engine": "google",
            "q": query,
            "num": max_results,
            "hl": "en",
            "gl": "us"
        }

        url = "https://serpapi.com/search"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_serp_results(data)
                    else:
                        print(f"Search API error: {response.status}")
                        return []
            except Exception as e:
                print(f"Search API request failed: {e}")
                return []

    def _parse_serp_results(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse SerpAPI response into SearchResult objects"""

        results = []
        organic_results = data.get("organic_results", [])

        for i, result in enumerate(organic_results):
            try:
                # Extract domain from URL
                domain = result.get("link", "").split("/")[2] if result.get("link") else ""

                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("link", ""),
                    snippet=result.get("snippet", ""),
                    domain=domain,
                    rank=i + 1
                )

                # Try to parse date if available
                if result.get("date"):
                    try:
                        # This would need proper date parsing based on SerpAPI format
                        pass
                    except:
                        pass

                results.append(search_result)

            except Exception as e:
                print(f"Error parsing search result: {e}")
                continue

        return results

    def _mock_search_results(self, query: str, max_results: int) -> List[SearchResult]:
        """Mock search results for development/testing"""

        mock_results = [
            SearchResult(
                title=f"Mock Result 1 for: {query}",
                url="https://example.com/result1",
                snippet=f"This is a mock search result snippet for the query '{query}'. It contains relevant information about the topic.",
                domain="example.com",
                rank=1
            ),
            SearchResult(
                title=f"Mock Result 2 for: {query}",
                url="https://docs.example.com/result2",
                snippet=f"Another mock result providing additional context and information related to '{query}'.",
                domain="docs.example.com",
                rank=2
            ),
            SearchResult(
                title=f"Latest News: {query}",
                url="https://news.example.com/latest",
                snippet=f"Recent developments and updates regarding '{query}' with current market insights.",
                domain="news.example.com",
                rank=3
            )
        ]

        return mock_results[:max_results]

    def _filter_results(
        self,
        results: List[SearchResult],
        site_whitelist: List[str] = None
    ) -> List[SearchResult]:
        """Filter results based on site whitelist and quality"""

        if not site_whitelist:
            return results

        # Filter by whitelisted domains
        filtered = []
        for result in results:
            if any(domain in result.domain for domain in site_whitelist):
                filtered.append(result)

        return filtered

    def _check_budget(self, agent_id: int) -> bool:
        """Check if agent is within budget limits"""

        now = datetime.now()
        hour_key = f"{agent_id}_{now.hour}_{now.date()}"

        # Check hourly limit
        hourly_calls = self.call_tracker.get(hour_key, 0)
        if hourly_calls >= self.max_calls_per_agent:
            return False

        return True

    def _track_usage(self, agent_id: int):
        """Track search usage for budget limits"""

        now = datetime.now()
        hour_key = f"{agent_id}_{now.hour}_{now.date()}"  # Include date to avoid confusion

        self.call_tracker[hour_key] = self.call_tracker.get(hour_key, 0) + 1

        # Clean up old tracking data more aggressively
        # Keep only current hour and previous hour
        current_hour_key = f"{agent_id}_{now.hour}_{now.date()}"
        previous_hour = (now.hour - 1) % 24
        previous_date = now.date() if now.hour > 0 else (now.date() - timedelta(days=1))
        previous_hour_key = f"{agent_id}_{previous_hour}_{previous_date}"

        # Remove all keys except current and previous hour
        valid_keys = {current_hour_key, previous_hour_key}
        keys_to_remove = [key for key in self.call_tracker.keys() if key not in valid_keys]

        for key in keys_to_remove:
            del self.call_tracker[key]

        # Additional safety: if tracker grows too large, clear it
        if len(self.call_tracker) > 1000:
            self.call_tracker.clear()
            self.call_tracker[current_hour_key] = 1

    async def search_site_specific(
        self,
        query: str,
        site_domain: str,
        max_results: int = 5
    ) -> List[SearchResult]:
        """Search within a specific site domain"""

        return await self.search(
            query=query,
            site_whitelist=[site_domain],
            max_results=max_results
        )

    def get_usage_stats(self, agent_id: int) -> Dict[str, Any]:
        """Get usage statistics for an agent"""

        now = datetime.now()
        hour_key = f"{agent_id}_{now.hour}_{now.date()}"

        calls_this_hour = self.call_tracker.get(hour_key, 0)
        return {
            "calls_this_hour": calls_this_hour,
            "limit_per_hour": self.max_calls_per_agent,
            "remaining_calls": max(0, self.max_calls_per_agent - calls_this_hour),
            "tracker_size": len(self.call_tracker)  # For monitoring memory usage
        }


# Global instance
web_search_service = WebSearchService()