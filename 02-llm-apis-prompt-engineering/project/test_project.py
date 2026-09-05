"""
Automated Test Suite for Phase 2: LLM APIs & Prompt Engineering.

Verifies:
- Pydantic models & validation constraints for GeneratedArticle
- Prompt template builder functions & formatting
- Tiktoken BPE token counting accuracy
- Multi-model pricing & cost calculator calculations
- BaseLLMProvider & MockLLMProvider generation, structured output, and streaming
"""

import pytest
from pydantic import ValidationError

from models import GeneratedArticle, ArticleSection, TokenCostBreakdown
from prompt_templates import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_chain_of_thought_prompt,
    build_persona_prompt,
    build_structured_article_prompt
)
from providers import (
    MockLLMProvider,
    get_provider,
    count_tokens_with_tiktoken
)


# ============================================================================
# 1. Pydantic Model Validation Tests
# ============================================================================

def test_article_section_valid():
    """Verify clean title casing and structure for ArticleSection."""
    sec = ArticleSection(
        heading="introduction to vector embeddings",
        content="Embeddings represent semantic meaning in continuous vector spaces.",
        key_takeaways=["Meaning is mapped to vectors", "Cosine similarity measures distance"]
    )
    assert sec.heading == "Introduction To Vector Embeddings"
    assert len(sec.key_takeaways) == 2


def test_generated_article_validation():
    """Verify slug generation, tag sanitization, and section requirements."""
    article = GeneratedArticle(
        title="Comprehensive Guide to LLM Observability",
        slug="Comprehensive Guide to LLM Observability!",
        summary="A detailed study on tracing, token metrics, and latency analysis in AI pipelines.",
        target_audience="intermediate",
        reading_time_minutes=8,
        sections=[
            ArticleSection(
                heading="Tracing Token Latency",
                content="Distributed tracing allows engineers to identify bottleneck calls.",
                key_takeaways=["Track TTFT", "Measure queue delays"]
            ),
            ArticleSection(
                heading="Cost Optimization Tactics",
                content="Semantic caching and model routing drastically reduce monthly API bills.",
                key_takeaways=["Use smaller models for classification", "Cache frequent prompts"]
            )
        ],
        tags=["LLM", "Observability", " Python "],
        seo_meta_description="Learn how to monitor, trace, and optimize token costs in production LLM applications."
    )

    # Verify auto-sanitization
    assert article.slug == "comprehensive-guide-to-llm-observability"
    assert "llm" in article.tags
    assert "python" in article.tags
    assert len(article.sections) == 2


def test_article_invalid_short_summary():
    """Summary under 20 chars should trigger ValidationError."""
    with pytest.raises(ValidationError):
        GeneratedArticle(
            title="Valid Long Title Here",
            slug="valid-slug",
            summary="Too short",  # < 20 chars
            target_audience="advanced",
            sections=[
                ArticleSection(heading="Sec 1", content="Content that is long enough to pass validation.", key_takeaways=["T1"]),
                ArticleSection(heading="Sec 2", content="Content that is long enough to pass validation.", key_takeaways=["T2"])
            ],
            tags=["rag", "ai"],
            seo_meta_description="A valid meta description that is easily over fifty characters long for SEO."
        )


# ============================================================================
# 2. Prompt Template Tests
# ============================================================================

def test_prompt_builders():
    """Verify that prompt builder functions output proper delimiters and instructions."""
    topic = "Prompt Caching"

    zero_shot = build_zero_shot_prompt(topic)
    assert "Topic to explain: \"Prompt Caching\"" in zero_shot
    assert "Core Definition" in zero_shot

    few_shot = build_few_shot_prompt(topic)
    assert "### Example 1" in few_shot
    assert "Input: Explain Prompt Caching" in few_shot

    cot = build_chain_of_thought_prompt(topic)
    assert "<thinking>" in cot
    assert "<final_answer>" in cot

    sys_p, user_p = build_persona_prompt(topic, "technical_lead")
    assert "Staff AI Systems Architect" in sys_p
    assert topic in user_p


# ============================================================================
# 3. Token Counting & Pricing Tests
# ============================================================================

def test_tiktoken_counting():
    """Verify BPE token counting returns consistent, reasonable numbers."""
    text = "Hello, world! AI Engineering is transformative."
    tokens = count_tokens_with_tiktoken(text)
    assert tokens > 0
    assert tokens < len(text)  # tokens are fewer than raw characters


def test_token_cost_calculation():
    """Verify USD cost math for different model families."""
    # 1,000,000 input tokens on gpt-4o-mini ($0.15) + 1,000,000 output ($0.60) = $0.75
    breakdown = TokenCostBreakdown(
        model_name="gpt-4o-mini",
        input_tokens=1_000_000,
        output_tokens=1_000_000
    )
    assert breakdown.calculate_cost_usd() == 0.75

    # 10,000 input tokens on claude-3-5-sonnet ($3.00/1M) = $0.03
    sonnet_breakdown = TokenCostBreakdown(
        model_name="claude-3-5-sonnet",
        input_tokens=10_000,
        output_tokens=0
    )
    assert sonnet_breakdown.calculate_cost_usd() == 0.03


# ============================================================================
# 4. Provider Tests (Mock & Factory)
# ============================================================================

@pytest.mark.asyncio
async def test_mock_provider_generation():
    """Verify that MockLLMProvider generates valid text and token metrics."""
    provider = MockLLMProvider()
    prompt = "Explain RAG pipelines."
    output, metrics = await provider.generate_completion(prompt)

    assert len(output) > 20
    assert metrics.total_tokens > 0
    assert metrics.calculate_cost_usd() >= 0


@pytest.mark.asyncio
async def test_mock_provider_chain_of_thought():
    """Verify CoT prompt triggers thinking and final answer tags in mock output."""
    provider = MockLLMProvider()
    cot_prompt = build_chain_of_thought_prompt("RAG Architecture")
    output, _ = await provider.generate_completion(cot_prompt)

    assert "<thinking>" in output
    assert "<final_answer>" in output


@pytest.mark.asyncio
async def test_mock_provider_structured_output():
    """Verify that MockLLMProvider returns a fully valid GeneratedArticle."""
    provider = MockLLMProvider()
    article, metrics = await provider.generate_structured_article("Vector Search", audience="advanced")

    assert isinstance(article, GeneratedArticle)
    assert len(article.sections) >= 2
    assert len(article.tags) >= 2
    assert metrics.total_tokens > 0


@pytest.mark.asyncio
async def test_mock_provider_streaming():
    """Verify async generator yields chunks and running metrics."""
    provider = MockLLMProvider()
    chunks = []
    async for chunk, metrics in provider.stream_completion("Test stream"):
        chunks.append(chunk)

    assert len(chunks) > 5
    assert "".join(chunks).startswith("Streaming response")
    assert metrics.output_tokens > 0


def test_get_provider_fallback():
    """get_provider should default to a usable provider even if env vars are empty."""
    provider = get_provider()
    assert provider is not None
    assert hasattr(provider, "generate_completion")
