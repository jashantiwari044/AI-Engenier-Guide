# ⚡ Phase 2 Cheat Sheet: LLM APIs & Prompt Engineering

A quick reference guide for working with OpenAI, Anthropic, Gemini, Tiktoken, and Prompt Engineering patterns.

---

## 1. OpenAI SDK Quick Reference (Python)

### Direct Chat Completion
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-...")

response = await client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    messages=[
        {"role": "system", "content": "You are a senior database architect."},
        {"role": "user", "content": "Explain Postgres B-Tree indexing."}
    ]
)
content = response.choices[0].message.content
print(f"Tokens: {response.usage.total_tokens}")
```

### Native Pydantic Structured Output (Parse API)
```python
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

class ExtractionSchema(BaseModel):
    sentiment: str = Field(..., description="positive, negative, or neutral")
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_phrases: list[str]

client = AsyncOpenAI()
completion = await client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Review: Antigravity IDE is lightning fast!"}],
    response_format=ExtractionSchema
)
# Returns already validated Pydantic object!
result: ExtractionSchema = completion.choices[0].message.parsed
```

### Real-Time Token Streaming
```python
stream = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Stream a haiku"}],
    stream=True
)

async for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
```

---

## 2. Anthropic Claude SDK Quick Reference

### Messages API & Streaming
```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key="sk-ant-...")

# Regular Message
response = await client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    system="You are an expert compiler engineer.",
    messages=[{"role": "user", "content": "Explain LLVM IR."}]
)
print(response.content[0].text)

# Streaming
async with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Explain async IO."}]
) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## 3. BPE Token Counting with `tiktoken`

```python
import tiktoken

# OpenAI's cl100k_base tokenizer (used by GPT-4o, GPT-4, GPT-3.5)
encoding = tiktoken.get_encoding("cl100k_base")

text = "RAG architecture requires vector indexing."
tokens = encoding.encode(text)

print(f"Token count: {len(tokens)}")  # e.g., 7
print(f"Token IDs: {tokens}")          # [50, 1204, ...]
print(f"Decoded: {encoding.decode(tokens)}")
```

---

## 4. Prompt Engineering Patterns Cheatsheet

| Technique | When to Use | Prompt Formula |
|---|---|---|
| **Zero-Shot** | General Q&A, simple summarization | `"Task: {instruction}\nInput: {data}"` |
| **Few-Shot** | Enforcing specific format, syntax, or tone | Provide 2-3 `Input: ... \nOutput: ...` pairs before user prompt |
| **Chain-of-Thought (CoT)** | Multi-step reasoning, math, logic, architecture | `"Think step-by-step inside <thinking> tags before giving the <final_answer>"` |
| **Persona Steering** | Expert domain output, adjusting complexity | `"You are a {Role}. Your responses prioritize {values}..."` |
| **Directional Stimulus** | Guiding summary focus without fine-tuning | `"Summarize the article with emphasis on {specific metric/angle}"` |

---

## 5. Hyperparameter Tuning Guide

| Parameter | Range | Recommended Value | Impact |
|---|---|---|---|
| **`temperature`** | 0.0 – 2.0 | **0.0 - 0.2**: Code, Math, JSON<br>**0.7**: Default Q&A<br>**1.2+**: Creative brainstorming | Controls randomness. Lower = deterministic; Higher = diverse/creative. |
| **`top_p`** (Nucleus) | 0.0 – 1.0 | **0.9 - 1.0** (Change either `temp` OR `top_p`, not both) | Samples from top cumulative probability mass. |
| **`max_tokens`** | 1 – context limit | Set based on expected output length | Hard cap on output length. Protects against infinite generation loops. |
| **`presence_penalty`** | -2.0 – 2.0 | **0.0 - 0.5** | Penalizes tokens based on whether they appeared at all. Increases topic diversity. |
| **`frequency_penalty`** | -2.0 – 2.0 | **0.0 - 0.5** | Penalizes tokens based on frequency count. Reduces word repetition. |
