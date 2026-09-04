# ⚡ AI Engineer Python Cheat Sheet (Phase 1)

Essential, battle-tested Python patterns for building LLM applications, RAG pipelines, and Agent workflows.

---

## 1. Pydantic v2 Mastery

### A. Defining Structured Output / Tool Call Schema
```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal, List
from datetime import datetime

class ToolParameterSchema(BaseModel):
    query: str = Field(..., min_length=2, description="Search query string")
    max_results: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    category: Optional[Literal["news", "academic", "general"]] = Field(
        default="general",
        description="Filter domain category"
    )
    include_raw_html: bool = Field(default=False)

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        return v.strip().lower()

    @model_validator(mode="after")
    def validate_rules(self):
        if self.category == "academic" and self.include_raw_html:
            raise ValueError("Raw HTML cannot be returned for academic papers")
        return self
```

### B. Serialization & Export for LLMs
```python
# 1. Generate JSON Schema (passed to OpenAI/Anthropic tool definitions)
json_schema = ToolParameterSchema.model_json_schema()

# 2. Parse from raw LLM string output
raw_json = '{"query": "LangGraph tutorial", "max_results": 10}'
parsed_obj = ToolParameterSchema.model_validate_json(raw_json)

# 3. Convert back to Python dict or JSON string
data_dict = parsed_obj.model_dump()
json_str = parsed_obj.model_dump_json(indent=2)
```

---

## 2. Asyncio & High-Throughput I/O

### A. Parallel Requests with Concurrency Throttle (Semaphore)
```python
import asyncio
import httpx

# Throttle to max 5 concurrent requests (prevents HTTP 429 Rate Limits)
concurrency_limit = asyncio.Semaphore(5)

async def fetch_item(client: httpx.AsyncClient, item_id: int):
    async with concurrency_limit:
        response = await client.get(f"https://api.example.com/items/{item_id}")
        response.raise_for_status()
        return response.json()

async def batch_fetch(item_ids: list[int]):
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(limits=limits, timeout=10.0) as client:
        tasks = [fetch_item(client, i) for i in item_ids]
        # Run all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

### B. Streaming Tokens (Async Generator)
```python
import asyncio
from typing import AsyncGenerator

async def mock_llm_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Simulates token-by-token streaming from an LLM."""
    tokens = ["Hello", " world,", " this", " is", " streaming", " AI!"]
    for token in tokens:
        await asyncio.sleep(0.05) # simulate generation latency
        yield token

# Consuming the stream:
async def consume():
    async for token in mock_llm_stream("hi"):
        print(token, end="", flush=True)
    print()
```

---

## 3. Resilience: Retries with Exponential Backoff (`tenacity`)

```python
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

# Retry up to 4 times, wait 1s, 2s, 4s, 8s (exponential), retry only on network errors or 5xx/429
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def call_llm_with_resilience(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
```

---

## 4. Environment & Secret Management

```python
import os
from dotenv import load_dotenv

# Load .env file at application startup
load_dotenv(override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables. Check your .env file.")
```

---

## 5. Type Hinting Patterns in AI Workflows

| Type | Purpose | Example in AI Engineering |
|---|---|---|
| `Literal["a", "b"]` | Restricts value to specific strings | Routing between agents (`Literal["billing", "tech_support"]`) |
| `Optional[T]` or `T \| None` | Nullable fields | Optional metadata or citations in RAG chunk |
| `TypedDict` | Dictionary with typed keys | State dictionary in LangGraph |
| `Annotated[T, Reducer]` | Carries runtime metadata | Reducer function for state aggregation in LangGraph |
