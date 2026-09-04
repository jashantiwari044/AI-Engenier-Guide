"""
Core Pydantic v2 Data Models for AI Engineering.

Demonstrates:
- Strict field validation and constraints
- Nested models and type annotations
- Custom validators (@field_validator and @model_validator)
- Dynamic JSON Schema export for LLM Function Calling
- Token usage & cost estimation calculations
"""

from datetime import datetime, timezone
from typing import Generic, List, Literal, Optional, TypeVar
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# 1. Geographic and Weather Models (Simulating External API / Tool Execution)
# ============================================================================

class GeoLocation(BaseModel):
    """Geographical coordinate pair with strict boundary validations."""
    city_name: str = Field(..., min_length=1, max_length=100, description="Name of the city")
    country: str = Field(..., min_length=2, max_length=100, description="Country name")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude from -90 to 90 degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude from -180 to 180 degrees")
    timezone: str = Field(default="UTC", description="IANA Timezone string")

    @field_validator("city_name", "country")
    @classmethod
    def strip_whitespace(cls, value: str) -> str:
        return value.strip()


class CurrentWeather(BaseModel):
    """Validated current weather data."""
    temperature_celsius: float = Field(..., description="Temperature in degrees Celsius")
    windspeed_kmh: float = Field(..., ge=0.0, description="Wind speed in km/h")
    weather_code: int = Field(..., description="WMO Weather interpretation code")
    is_day: bool = Field(default=True, description="True if daytime, False if nighttime")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Observation timestamp in UTC"
    )

    @property
    def temperature_fahrenheit(self) -> float:
        """Helper property to compute Fahrenheit on the fly."""
        return round((self.temperature_celsius * 9 / 5) + 32, 2)

    @property
    def condition_summary(self) -> str:
        """Translates WMO weather codes into human-readable descriptions."""
        code_map = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snowfall",
            73: "Moderate snowfall",
            80: "Rain showers",
            95: "Thunderstorm",
        }
        return code_map.get(self.weather_code, f"Weather condition code {self.weather_code}")


class WeatherReport(BaseModel):
    """Complete aggregated weather report for a location."""
    location: GeoLocation
    weather: CurrentWeather
    cached: bool = Field(default=False, description="Whether this report was served from cache")
    retrieval_latency_ms: float = Field(default=0.0, ge=0.0, description="API retrieval latency in milliseconds")


# ============================================================================
# 2. LLM Chat & Tool-Calling Specifications
# ============================================================================

RoleType = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """Standardized chat message representation compatible with OpenAI/Anthropic."""
    role: RoleType = Field(..., description="The role of the author of this message")
    content: str = Field(..., description="The textual contents of the message")
    name: Optional[str] = Field(default=None, description="Optional name of the participant or tool")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_tool_message(self):
        """Tool responses must provide the name of the tool called."""
        if self.role == "tool" and not self.name:
            raise ValueError("Messages with role 'tool' must specify the 'name' field")
        return self


class TokenMetrics(BaseModel):
    """Token counting and cost estimation model."""
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    model_name: str = Field(default="gpt-4o-mini")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def calculate_estimated_cost_usd(self) -> float:
        """
        Estimates USD cost based on representative pricing:
        gpt-4o-mini: $0.15 / 1M input tokens, $0.60 / 1M output tokens
        """
        rates = {
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-4o": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000},
            "claude-3-5-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        }
        pricing = rates.get(self.model_name, rates["gpt-4o-mini"])
        cost = (self.prompt_tokens * pricing["input"]) + (self.completion_tokens * pricing["output"])
        return round(cost, 6)


class ToolDefinition(BaseModel):
    """
    Schema representation for an LLM tool / function call.
    Matches OpenAI Function Calling JSON Schema specification.
    """
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = Field(..., min_length=5, description="Clear instructions for when the LLM should call this tool")
    parameters: dict = Field(..., description="Valid JSON Schema representing tool input arguments")

    @classmethod
    def from_pydantic_model(cls, name: str, description: str, model_cls: type[BaseModel]) -> "ToolDefinition":
        """Factory helper: Generates an LLM tool definition directly from any Pydantic model class."""
        return cls(
            name=name,
            description=description,
            parameters=model_cls.model_json_schema()
        )


# ============================================================================
# 3. Generic Envelope for API Responses
# ============================================================================

T = TypeVar("T")

class APIResponseEnvelope(BaseModel, Generic[T]):
    """Standardized envelope for robust API responses across AI pipelines."""
    success: bool
    data: Optional[T] = None
    error_message: Optional[str] = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
