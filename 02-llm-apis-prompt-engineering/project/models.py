"""
Phase 2: Core Data Models & Schemas for LLM APIs & Prompt Engineering.

Defines:
- Structured Output Pydantic schemas for LLM generation
- Article generation data structures (sections, SEO, tags)
- Prompt engineering comparison results
- Token usage & multi-provider cost estimation models
"""

import re
from datetime import datetime, timezone
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# 1. Structured Output Schemas (Target for LLM Generation)
# ============================================================================

class ArticleSection(BaseModel):
    """A distinct section of a generated technical article."""
    heading: str = Field(..., min_length=3, max_length=120, description="Section heading")
    content: str = Field(..., min_length=20, description="In-depth section body text")
    key_takeaways: List[str] = Field(
        default_factory=list,
        description="2-4 bullet points summarizing the core insight of this section"
    )

    @field_validator("heading")
    @classmethod
    def clean_heading(cls, v: str) -> str:
        return v.strip().title()


class GeneratedArticle(BaseModel):
    """
    Comprehensive structured technical article produced by LLMs.
    Demonstrates nested models, array fields, and custom field validators.
    """
    title: str = Field(..., min_length=5, max_length=150, description="Catchy, professional title")
    slug: str = Field(..., description="URL-friendly kebab-case slug")
    summary: str = Field(..., min_length=20, max_length=300, description="Executive 2-sentence summary")
    target_audience: Literal["beginner", "intermediate", "advanced", "executive"] = Field(
        default="intermediate",
        description="Target readership depth"
    )
    reading_time_minutes: int = Field(default=5, ge=1, le=60, description="Estimated reading time in minutes")
    sections: List[ArticleSection] = Field(
        ...,
        min_length=2,
        description="At least 2 detailed sections exploring the topic"
    )
    tags: List[str] = Field(
        default_factory=list,
        min_length=2,
        max_length=8,
        description="Relevant technology tags, e.g. ['python', 'rag', 'llm']"
    )
    seo_meta_description: str = Field(
        ...,
        min_length=50,
        max_length=160,
        description="SEO-optimized meta description (50-160 chars)"
    )

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", clean):
            # Auto-sanitize if not strictly formatted
            clean = re.sub(r"[^a-z0-9]+", "-", clean).strip("-")
        return clean

    @field_validator("tags")
    @classmethod
    def lowercase_tags(cls, tags: List[str]) -> List[str]:
        return [t.strip().lower() for t in tags if t.strip()]


# ============================================================================
# 2. Prompt Engineering Comparison Models
# ============================================================================

PromptTechnique = Literal[
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "persona_conditioned"
]


class PromptComparisonResult(BaseModel):
    """Captures performance and output characteristics across prompt techniques."""
    technique: PromptTechnique
    technique_name: str
    prompt_used: str
    raw_output: str
    reasoning_steps: Optional[List[str]] = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    provider_name: str = Field(default="mock")


# ============================================================================
# 3. Token Accounting & Multi-Provider Pricing Models
# ============================================================================

class TokenCostBreakdown(BaseModel):
    """Calculates token costs across major AI providers based on standard pricing."""
    model_name: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def calculate_cost_usd(self) -> float:
        """
        Pricing per 1M tokens (Standard 2025/2026 rates):
        - gpt-4o: $2.50 input / $10.00 output
        - gpt-4o-mini: $0.15 input / $0.60 output
        - claude-3-5-sonnet: $3.00 input / $15.00 output
        - claude-3-haiku: $0.25 input / $1.25 output
        - gemini-1.5-flash: $0.075 input / $0.30 output
        - gemini-1.5-pro: $1.25 input / $5.00 output
        """
        rates: Dict[str, Dict[str, float]] = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "claude-3-5-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
            "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
            "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
            "gemini-1.5-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
        }

        # Match model family (longest key first so 'gpt-4o-mini' matches before 'gpt-4o')
        matched_rate = None
        for key in sorted(rates.keys(), key=len, reverse=True):
            if key in self.model_name.lower():
                matched_rate = rates[key]
                break

        if not matched_rate:
            matched_rate = rates["gpt-4o-mini"]  # safe fallback

        cost = (self.input_tokens * matched_rate["input"]) + (self.output_tokens * matched_rate["output"])
        return round(cost, 6)
