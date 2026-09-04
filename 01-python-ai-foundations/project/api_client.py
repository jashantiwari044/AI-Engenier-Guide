"""
Asynchronous, Resilient API Client for AI Engineering Pipelines.

Key Features:
- Non-blocking HTTP I/O using httpx.AsyncClient
- Concurrency limiting with asyncio.Semaphore (preventing HTTP 429 rate limits)
- Production retry logic with exponential backoff using tenacity
- End-to-end performance tracking (latency in milliseconds)
- Integration with Open-Meteo public APIs (no API key required)
- Token streaming generator simulating LLM inference streams
"""

import asyncio
import logging
import os
import time
from typing import AsyncGenerator, List, Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from models import (
    APIResponseEnvelope,
    CurrentWeather,
    GeoLocation,
    WeatherReport,
    TokenMetrics
)

# Setup logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("AIClient")


class ResilientAIClient:
    """
    Production-ready asynchronous client designed for high-concurrency AI agent workflows.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        timeout_seconds: float = 10.0,
        max_retries: int = 3
    ):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        self.limits = httpx.Limits(max_keepalive_connections=5, max_connections=15)
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Context manager entry: initializes connection pool."""
        self._client = httpx.AsyncClient(timeout=self.timeout, limits=self.limits)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: gracefully cleans up connections."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            raise RuntimeError("ResilientAIClient must be used within an 'async with' block.")
        return self._client

    # =========================================================================
    # Resilient Network Calls with Tenacity
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _fetch_url_with_retry(self, url: str, params: Optional[dict] = None) -> dict:
        """Internal HTTP GET with automatic retries and exponential backoff."""
        async with self.semaphore:  # Enforce concurrency throttle
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    # =========================================================================
    # Real-World API Operations (Geocoding & Weather as AI Agent Tools)
    # =========================================================================

    async def get_city_coordinates(self, city_name: str) -> GeoLocation:
        """
        Calls Open-Meteo Geocoding API to resolve city coordinates.
        Converts raw JSON directly into a validated GeoLocation Pydantic model.
        """
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
        
        data = await self._fetch_url_with_retry(url, params=params)
        results = data.get("results")
        if not results:
            raise ValueError(f"City '{city_name}' could not be resolved by Geocoding API.")

        first_match = results[0]
        return GeoLocation(
            city_name=first_match.get("name", city_name),
            country=first_match.get("country", "Unknown"),
            latitude=first_match["latitude"],
            longitude=first_match["longitude"],
            timezone=first_match.get("timezone", "UTC")
        )

    async def get_weather_report(self, city_name: str) -> APIResponseEnvelope[WeatherReport]:
        """
        Full orchestration: Resolve coordinates -> Fetch weather -> Return validated envelope.
        Calculates wall-clock latency for observability.
        """
        start_time = time.perf_counter()
        try:
            # 1. Geocode
            geo = await self.get_city_coordinates(city_name)

            # 2. Fetch current weather for coordinates
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": geo.latitude,
                "longitude": geo.longitude,
                "current_weather": True
            }
            weather_data = await self._fetch_url_with_retry(url, params=params)
            current_raw = weather_data.get("current_weather")
            if not current_raw:
                raise ValueError("Weather data missing current_weather field.")

            weather = CurrentWeather(
                temperature_celsius=current_raw["temperature"],
                windspeed_kmh=current_raw["windspeed"],
                weather_code=current_raw["weathercode"],
                is_day=bool(current_raw.get("is_day", 1))
            )

            latency = round((time.perf_counter() - start_time) * 1000, 2)
            report = WeatherReport(
                location=geo,
                weather=weather,
                cached=False,
                retrieval_latency_ms=latency
            )

            return APIResponseEnvelope(
                success=True,
                data=report,
                latency_ms=latency
            )

        except Exception as exc:
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"Error fetching weather for '{city_name}': {str(exc)}")
            return APIResponseEnvelope(
                success=False,
                error_message=str(exc),
                latency_ms=latency
            )

    async def get_weather_batch(self, cities: List[str]) -> List[APIResponseEnvelope[WeatherReport]]:
        """
        Executes parallel requests across multiple cities concurrently using asyncio.gather.
        Protected by the internal concurrency semaphore.
        """
        tasks = [self.get_weather_report(city) for city in cities]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    # =========================================================================
    # Simulated LLM Token Streaming
    # =========================================================================

    async def stream_simulated_ai_response(
        self,
        prompt: str,
        system_role: str = "Assistant"
    ) -> AsyncGenerator[tuple[str, TokenMetrics], None]:
        """
        Simulates token-by-token streaming from an LLM provider (OpenAI/Anthropic).
        Yields (chunk_text, running_metrics).
        """
        response_template = (
            f"Based on your query '{prompt}', I have processed the request. "
            "In modern AI architectures, streaming tokens provides immediate feedback "
            "to users while backend agents continue their reasoning loops."
        )
        words = response_template.split(" ")
        prompt_tokens = max(1, len(prompt.split()) + 5)
        accumulated_output_tokens = 0

        for i, word in enumerate(words):
            await asyncio.sleep(0.04)  # Simulate network chunk interval (40ms)
            chunk = word + (" " if i < len(words) - 1 else "")
            accumulated_output_tokens += 1
            metrics = TokenMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=accumulated_output_tokens,
                model_name="gpt-4o-mini"
            )
            yield chunk, metrics
