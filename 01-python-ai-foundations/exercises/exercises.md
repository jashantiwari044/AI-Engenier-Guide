# 🏋️ Phase 1 Practice Exercises & Challenges

These exercises reinforce the core Python skills every AI Engineer needs daily. Try solving each exercise on your own before looking at the solution!

---

## Exercise 1: Concurrency-Throttled Prompt Batching

### Background
When evaluating a prompt across 50 test cases, sending all 50 requests simultaneously triggers an HTTP 429 (Rate Limit Exceeded) error.

### Task
Write an asynchronous function `process_eval_batch(prompts: list[str], max_concurrency: int = 3)` that:
1. Uses `asyncio.Semaphore` to allow no more than `max_concurrency` concurrent executions.
2. Simulates an LLM call taking between 0.1 and 0.3 seconds.
3. Returns a list of dictionaries with `{"prompt": prompt, "result": result, "duration": elapsed}`.
4. Preserves the original input order.

### Solution

```python
import asyncio
import random
import time

async def simulate_llm_call(prompt: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        start = time.perf_counter()
        latency = random.uniform(0.1, 0.3)
        await asyncio.sleep(latency)
        elapsed = time.perf_counter() - start
        return {
            "prompt": prompt,
            "result": f"Answer to '{prompt}'",
            "duration": round(elapsed, 3)
        }

async def process_eval_batch(prompts: list[str], max_concurrency: int = 3) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [simulate_llm_call(p, sem) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# Test run
async def main():
    test_prompts = [f"Eval Question #{i}" for i in range(10)]
    t0 = time.perf_counter()
    results = await process_eval_batch(test_prompts, max_concurrency=3)
    print(f"Processed {len(results)} prompts in {time.perf_counter() - t0:.2f}s")
    for r in results[:3]:
        print(" ", r)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Exercise 2: Strict Pydantic Schema for an SQL Agent Tool

### Background
When giving an LLM an SQL execution tool, you must guarantee that it only executes read-only queries and cannot run destructive statements (`DROP`, `DELETE`, `UPDATE`, `ALTER`).

### Task
Create a Pydantic model `SQLQueryToolInput` that:
1. Requires a `query: str` with at least 6 characters.
2. Validates that the query starts with `SELECT` (case-insensitive).
3. Rejects any query containing forbidden keywords: `DROP`, `DELETE`, `UPDATE`, `ALTER`, `INSERT`, `TRUNCATE`.
4. Has an optional `limit: int` with a default of 100, maximum of 1000.
5. Strips extra whitespace from the query.

### Solution

```python
from pydantic import BaseModel, Field, field_validator
import re

class SQLQueryToolInput(BaseModel):
    query: str = Field(
        ...,
        min_length=6,
        description="The read-only SQL query to execute against the analytics database."
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum rows to return."
    )

    @field_validator("query")
    @classmethod
    def enforce_read_only_sql(cls, v: str) -> str:
        clean = v.strip()
        # Must start with SELECT
        if not re.match(r"^SELECT\s+", clean, re.IGNORECASE):
            raise ValueError("Only 'SELECT' queries are permitted for this tool.")

        # Forbidden destructive keywords
        forbidden = ["DROP", "DELETE", "UPDATE", "ALTER", "INSERT", "TRUNCATE"]
        for word in forbidden:
            if re.search(rf"\b{word}\b", clean, re.IGNORECASE):
                raise ValueError(f"Destructive keyword '{word}' is strictly forbidden.")

        return clean

# Test cases
if __name__ == "__main__":
    # Valid
    valid = SQLQueryToolInput(query="SELECT * FROM users WHERE active = 1")
    print("Valid query:", valid.query)

    # Invalid - will raise ValidationError
    try:
        SQLQueryToolInput(query="DROP TABLE users;")
    except Exception as e:
        print("Caught malicious query:", e)
```

---

## Exercise 3: Resilient HTTP Caller with Exponential Backoff

### Background
Third-party LLM providers frequently return HTTP 429 or 503 errors during traffic spikes.

### Task
Use `tenacity` and `httpx` to create an async function `resilient_post(url: str, json_data: dict)` that:
1. Retries up to 4 times on HTTP status errors (especially 429, 500, 502, 503, 504) and connection timeouts.
2. Uses exponential backoff (e.g. 0.5s, 1s, 2s, 4s).
3. Reraises the original exception if all 4 attempts fail.

### Solution

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class TransientAPIError(Exception):
    pass

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=5.0),
    retry=retry_if_exception_type((httpx.RequestError, TransientAPIError)),
    reraise=True
)
async def resilient_post(url: str, json_data: dict) -> dict:
    async with httpx.AsyncClient(timeout=8.0) as client:
        response = await client.post(url, json=json_data)
        
        # Classify transient 5xx / 429 as retryable
        if response.status_code in (429, 500, 502, 503, 504):
            raise TransientAPIError(f"HTTP {response.status_code}: {response.text}")
            
        response.raise_for_status()
        return response.json()
```

---

## Exercise 4: Async Generator Token Buffering

### Background
When receiving raw token chunks from an LLM stream, you often want to buffer them until a complete sentence (ending in `.`, `?`, or `!`) is ready to send to a text-to-speech (TTS) engine.

### Task
Write an async generator `sentence_buffer(token_stream)` that:
1. Consumes an async stream of token strings.
2. Accumulates tokens until punctuation (`.`, `!`, `?`) is detected.
3. Yields full sentences as soon as they are completed.
4. Yields any leftover text at the end of the stream.

### Solution

```python
import asyncio
import re
from typing import AsyncGenerator

async def mock_token_stream() -> AsyncGenerator[str, None]:
    chunks = [
        "AI ", "agents ", "are ", "revolutionary. ",
        "They ", "use ", "tools! ",
        "Can ", "they ", "reason? ",
        "Yes, ", "step ", "by ", "step."
    ]
    for c in chunks:
        await asyncio.sleep(0.05)
        yield c

async def sentence_buffer(token_stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    buffer = ""
    sentence_end_pattern = re.compile(r"([.!?])(\s+|$)")

    async for token in token_stream:
        buffer += token
        match = sentence_end_pattern.search(buffer)
        if match:
            end_idx = match.end()
            sentence = buffer[:end_idx].strip()
            buffer = buffer[end_idx:]
            if sentence:
                yield sentence

    if buffer.strip():
        yield buffer.strip()

# Test run
async def main():
    async for sentence in sentence_buffer(mock_token_stream()):
        print(f"🔊 Ready for TTS: \"{sentence}\"")

if __name__ == "__main__":
    asyncio.run(main())
```
