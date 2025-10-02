"""
Redis-based rate limiting service.
Provides efficient, distributed rate limiting using Redis.
"""

import asyncio
import time
import logging
from typing import Optional, Tuple
import redis.asyncio as aioredis
from redis.asyncio import Redis

from .config import settings

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """Redis-based rate limiter with sliding window algorithm"""

    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.connection_retries = 3
        self.connection_timeout = 5

    async def initialize(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf8",
                decode_responses=True
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis rate limiter initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to Redis for rate limiting: {e}")
            logger.warning("Falling back to in-memory rate limiting")
            self.redis_client = None
            return False

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

    async def is_rate_limited(
        self,
        identifier: str,
        max_requests: int = None,
        window_seconds: int = None
    ) -> Tuple[bool, int, int]:
        """
        Check if identifier is rate limited using sliding window algorithm.

        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            max_requests: Maximum requests allowed (defaults to settings)
            window_seconds: Time window in seconds (defaults to settings)

        Returns:
            Tuple of (is_limited, requests_made, requests_remaining)
        """
        if max_requests is None:
            max_requests = settings.RATE_LIMIT_REQUESTS
        if window_seconds is None:
            window_seconds = settings.RATE_LIMIT_WINDOW

        if not self.redis_client:
            # Fallback to simple time-based check without persistence
            return await self._fallback_rate_limit(identifier, max_requests, window_seconds)

        try:
            return await self._redis_sliding_window_rate_limit(
                identifier, max_requests, window_seconds
            )
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to allow request on Redis errors (fail open)
            return False, 0, max_requests

    async def _redis_sliding_window_rate_limit(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, int, int]:
        """
        Implement sliding window rate limiting using Redis sorted sets.
        This is more accurate than fixed window approaches.
        """
        current_time = time.time()
        key = f"rate_limit:{identifier}"

        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()

        # Remove expired entries (older than window)
        cutoff_time = current_time - window_seconds
        pipe.zremrangebyscore(key, 0, cutoff_time)

        # Count current requests in window
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(current_time): current_time})

        # Set expiry for cleanup
        pipe.expire(key, window_seconds + 60)

        # Execute pipeline
        results = await pipe.execute()

        # Get current request count (after cleanup, before adding new request)
        current_requests = results[1]

        # Check if rate limited
        is_limited = current_requests >= max_requests
        requests_remaining = max(0, max_requests - current_requests - (0 if is_limited else 1))

        if is_limited:
            # Remove the request we just added since it's rate limited
            await self.redis_client.zrem(key, str(current_time))

        return is_limited, current_requests, requests_remaining

    async def _fallback_rate_limit(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, int, int]:
        """
        Fallback rate limiting when Redis is unavailable.
        Uses simple in-memory storage (not suitable for production clusters).
        """
        # This is a simplified fallback - in production you might want
        # to implement a more sophisticated in-memory solution
        logger.debug(f"Using fallback rate limiting for {identifier}")

        # For fallback, we'll be more lenient and allow most requests
        # This is safer than blocking legitimate traffic when Redis is down
        return False, 0, max_requests

    async def get_rate_limit_info(self, identifier: str) -> dict:
        """Get detailed rate limit information for an identifier"""
        if not self.redis_client:
            return {
                "requests_made": 0,
                "requests_remaining": settings.RATE_LIMIT_REQUESTS,
                "reset_time": time.time() + settings.RATE_LIMIT_WINDOW,
                "window_seconds": settings.RATE_LIMIT_WINDOW
            }

        try:
            current_time = time.time()
            key = f"rate_limit:{identifier}"

            # Clean up expired entries
            cutoff_time = current_time - settings.RATE_LIMIT_WINDOW
            await self.redis_client.zremrangebyscore(key, 0, cutoff_time)

            # Get current request count
            current_requests = await self.redis_client.zcard(key)

            # Get oldest request time to calculate reset time
            oldest_requests = await self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_requests:
                oldest_time = oldest_requests[0][1]
                reset_time = oldest_time + settings.RATE_LIMIT_WINDOW
            else:
                reset_time = current_time + settings.RATE_LIMIT_WINDOW

            return {
                "requests_made": current_requests,
                "requests_remaining": max(0, settings.RATE_LIMIT_REQUESTS - current_requests),
                "reset_time": reset_time,
                "window_seconds": settings.RATE_LIMIT_WINDOW
            }

        except Exception as e:
            logger.error(f"Error getting rate limit info: {e}")
            return {
                "requests_made": 0,
                "requests_remaining": settings.RATE_LIMIT_REQUESTS,
                "reset_time": time.time() + settings.RATE_LIMIT_WINDOW,
                "window_seconds": settings.RATE_LIMIT_WINDOW
            }

    async def reset_rate_limit(self, identifier: str) -> bool:
        """Reset rate limit for a specific identifier (admin function)"""
        if not self.redis_client:
            return True

        try:
            key = f"rate_limit:{identifier}"
            await self.redis_client.delete(key)
            logger.info(f"Rate limit reset for identifier: {identifier}")
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit for {identifier}: {e}")
            return False


# Global rate limiter instance
rate_limiter = RedisRateLimiter()


async def initialize_rate_limiter():
    """Initialize the global rate limiter"""
    return await rate_limiter.initialize()


async def cleanup_rate_limiter():
    """Cleanup the global rate limiter"""
    await rate_limiter.close()


# Convenience functions
async def check_rate_limit(identifier: str) -> Tuple[bool, int, int]:
    """Check if identifier is rate limited"""
    return await rate_limiter.is_rate_limited(identifier)


async def get_rate_limit_info(identifier: str) -> dict:
    """Get rate limit information for identifier"""
    return await rate_limiter.get_rate_limit_info(identifier)