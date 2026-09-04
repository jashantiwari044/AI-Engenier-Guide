"""
Automated Test Suite for Phase 1 Python Foundations.

Tests coverage:
- Pydantic validation rules and edge cases
- Temperature conversions and computed properties
- LLM Tool Calling Schema generation
- ChatMessage role validation
- Token metrics and cost calculations
- Asynchronous API client calls and streaming
"""

import pytest
from pydantic import ValidationError
from models import (
    GeoLocation,
    CurrentWeather,
    WeatherReport,
    ChatMessage,
    TokenMetrics,
    ToolDefinition
)
from api_client import ResilientAIClient


# ============================================================================
# 1. Pydantic Model Tests
# ============================================================================

def test_geo_location_valid():
    """Test valid coordinate instantiations."""
    geo = GeoLocation(
        city_name="  Tokyo  ",
        country="Japan ",
        latitude=35.6762,
        longitude=139.6503,
        timezone="Asia/Tokyo"
    )
    assert geo.city_name == "Tokyo"  # Whitespace stripped
    assert geo.country == "Japan"
    assert geo.latitude == 35.6762
    assert geo.longitude == 139.6503


def test_geo_location_invalid_latitude():
    """Latitude must strictly be between -90 and 90."""
    with pytest.raises(ValidationError) as exc:
        GeoLocation(
            city_name="Invalid City",
            country="Nowhere",
            latitude=95.0,  # Invalid!
            longitude=0.0
        )
    assert "latitude" in str(exc.value)


def test_current_weather_computed_properties():
    """Test dynamic Fahrenheit calculation and WMO condition summary."""
    weather = CurrentWeather(
        temperature_celsius=20.0,
        windspeed_kmh=15.0,
        weather_code=0,
        is_day=True
    )
    assert weather.temperature_fahrenheit == 68.0  # (20 * 9/5) + 32
    assert weather.condition_summary == "Clear sky"

    # Test unknown code fallback
    weather_unknown = CurrentWeather(
        temperature_celsius=0.0,
        windspeed_kmh=0.0,
        weather_code=999
    )
    assert "Weather condition code 999" in weather_unknown.condition_summary


def test_chat_message_tool_validator():
    """A message with role 'tool' MUST include a name field."""
    # Valid user message
    msg = ChatMessage(role="user", content="What is the weather?")
    assert msg.role == "user"

    # Valid tool message
    tool_msg = ChatMessage(role="tool", content="22°C", name="get_weather")
    assert tool_msg.name == "get_weather"

    # Invalid tool message without name
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="22°C")


def test_token_cost_calculation():
    """Test token usage and USD cost estimation."""
    metrics = TokenMetrics(
        prompt_tokens=1000,
        completion_tokens=500,
        model_name="gpt-4o-mini"
    )
    assert metrics.total_tokens == 1500
    # gpt-4o-mini: 1000 * 0.15/1M + 500 * 0.60/1M = 0.00015 + 0.00030 = 0.000450
    cost = metrics.calculate_estimated_cost_usd()
    assert cost == 0.00045


def test_tool_definition_schema_generation():
    """Test automated JSON schema generation from Pydantic model."""
    tool = ToolDefinition.from_pydantic_model(
        name="get_current_weather",
        description="Fetch real-time weather metrics for a geographic coordinate",
        model_cls=GeoLocation
    )
    assert tool.name == "get_current_weather"
    assert "properties" in tool.parameters
    assert "latitude" in tool.parameters["properties"]
    assert "longitude" in tool.parameters["properties"]


# ============================================================================
# 2. Async Client Tests
# ============================================================================

@pytest.mark.asyncio
async def test_async_client_context_manager():
    """Verify that the client safely manages its HTTP session lifecycle."""
    async with ResilientAIClient() as client:
        assert client.client is not None
        assert not client.client.is_closed

    # Outside the context block, accessing the client should raise error
    with pytest.raises(RuntimeError):
        _ = client.client


@pytest.mark.asyncio
async def test_real_geocoding_and_weather():
    """Integration test with Open-Meteo public endpoints."""
    async with ResilientAIClient(timeout_seconds=15.0) as client:
        response = await client.get_weather_report("London")
        assert response.success is True
        assert response.data is not None
        report: WeatherReport = response.data
        assert "London" in report.location.city_name
        assert isinstance(report.weather.temperature_celsius, float)
        assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_simulated_token_stream():
    """Test async generator token streaming."""
    client = ResilientAIClient()
    tokens = []
    async for chunk, metrics in client.stream_simulated_ai_response("Test prompt"):
        tokens.append(chunk)

    full_text = "".join(tokens)
    assert len(tokens) > 5
    assert "Test prompt" in full_text
    assert metrics.total_tokens > 0
