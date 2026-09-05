# 🏋️ Phase 2 Practice Exercises & Challenges

These exercises test your mastery of LLM APIs, BPE tokenization, prompt engineering strategies, and schema enforcement. Solve each challenge independently before consulting the provided solutions!

---

## Exercise 1: Few-Shot Sentiment & Aspect Extractor with Strict JSON

### Background
Real customer reviews contain mixed sentiments across multiple product aspects (e.g., "The food was delicious, but the service was terrible and overpriced"). A simple binary sentiment classifier fails on these nuances.

### Task
Write a function `extract_aspect_sentiments(review: str) -> dict` that:
1. Uses **Few-Shot Prompting** with at least two detailed exemplars.
2. Identifies all distinct aspects (e.g. `service`, `food`, `pricing`, `ambiance`).
3. Assigns each aspect a sentiment: `"positive"`, `"negative"`, or `"neutral"`, alongside a 1-sentence quote explanation.
4. Returns a validated dictionary structure.

### Solution

```python
import json
from pydantic import BaseModel, Field
from typing import List, Literal

class AspectDetail(BaseModel):
    aspect: str = Field(..., description="The specific feature mentioned")
    sentiment: Literal["positive", "negative", "neutral"]
    evidence_quote: str = Field(..., description="Direct quote supporting this sentiment")

class ReviewAnalysis(BaseModel):
    overall_sentiment: Literal["positive", "negative", "mixed", "neutral"]
    aspects: List[AspectDetail]

FEW_SHOT_PROMPT = """Analyze customer reviews by extracting multi-aspect sentiments. Follow the exact format of the examples.

### Example 1
Input: "The camera on this phone is mind-blowing, but battery life barely lasts 4 hours."
Output:
{
  "overall_sentiment": "mixed",
  "aspects": [
    {"aspect": "camera", "sentiment": "positive", "evidence_quote": "camera on this phone is mind-blowing"},
    {"aspect": "battery life", "sentiment": "negative", "evidence_quote": "barely lasts 4 hours"}
  ]
}

### Example 2
Input: "Great packaging and delivered right on time. Product works exactly as described."
Output:
{
  "overall_sentiment": "positive",
  "aspects": [
    {"aspect": "packaging", "sentiment": "positive", "evidence_quote": "Great packaging"},
    {"aspect": "delivery", "sentiment": "positive", "evidence_quote": "delivered right on time"},
    {"aspect": "functionality", "sentiment": "positive", "evidence_quote": "works exactly as described"}
  ]
}

### New Review
Input: "{user_review}"
Output:
"""

def parse_review(raw_json: str) -> ReviewAnalysis:
    return ReviewAnalysis.model_validate_json(raw_json)
```

---

## Exercise 2: Self-Reflecting Chain-of-Thought (CoT) with Verification

### Background
LLMs frequently jump to incorrect conclusions on subtle Python concurrency bugs. Adding a secondary **"Verification Step"** inside the CoT prompt cuts hallucination and logic errors by over 50%.

### Task
Write a prompt template that:
1. Forces the model to generate an initial draft solution inside `<draft>...</draft>`.
2. Forces the model to critique and stress-test its own draft inside `<critique>...</critique>` (checking for race conditions, deadlocks, and edge cases).
3. Produces the verified, definitive response inside `<final_answer>...</final_answer>`.

### Solution

```python
def build_self_reflecting_cot_prompt(code_snippet: str) -> str:
    return f"""You are a Principal Software Reliability Engineer.
Analyze the following Python snippet for subtle concurrency bugs, memory leaks, or race conditions:

```python
{code_snippet}
```

Follow this mandatory 3-stage Chain-of-Thought procedure:

1. <draft>
   Analyze the code line-by-line. Identify potential failure modes and propose an initial fix.
</draft>

2. <critique>
   Act as an adversarial reviewer. Challenge your draft:
   - What happens under high load or network failure?
   - Are there unhandled exceptions or deadlocks?
   - Does this fix introduce any new regression?
</critique>

3. <final_answer>
   State the root cause clearly and provide the battle-tested, production-ready corrected code.
</final_answer>
"""
```

---

## Exercise 3: Dynamic Model & Temperature Router

### Background
In production, you should not send every query to expensive frontier models ($15-$30/1M tokens) or use a static temperature of 0.7. Simple classifications should route to low-cost models (`gpt-4o-mini` at $0.15/1M tokens) with `temperature=0.0`. Complex coding or architectural design should route to frontier reasoning models.

### Task
Implement `route_request(task_type: str) -> dict` that returns the optimal `(model_name, temperature, max_tokens)` configuration.

### Solution

```python
from typing import Dict, Any

ROUTING_TABLE: Dict[str, Dict[str, Any]] = {
    "classification": {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 150,
        "description": "Deterministic, lowest cost, sub-second latency"
    },
    "json_extraction": {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 800,
        "description": "Deterministic schema conformance"
    },
    "general_qa": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1000,
        "description": "Balanced factual conversation"
    },
    "system_architecture": {
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.3,
        "max_tokens": 3000,
        "description": "High reasoning fidelity, low hallucination"
    },
    "creative_writing": {
        "model": "gpt-4o",
        "temperature": 1.2,
        "max_tokens": 2000,
        "description": "High entropy, expressive vocabulary"
    }
}

def route_request(task_type: str) -> Dict[str, Any]:
    """Returns optimal model hyperparameters based on task characteristics."""
    return ROUTING_TABLE.get(task_type.lower(), ROUTING_TABLE["general_qa"])

# Example verification:
if __name__ == "__main__":
    config = route_request("classification")
    print("Routing for classification:", config)
    assert config["temperature"] == 0.0
```

---

## Exercise 4: Context Window Budgeter & Message Truncator

### Background
Chat applications accumulate messages over time. If you blindly append messages to the conversation history, you will exceed the model's context window or incur massive token costs.

### Task
Write a function `fit_messages_within_budget(messages: list[dict], max_token_budget: int = 1000) -> list[dict]` that:
1. Always preserves the `system` prompt (first message).
2. Calculates exact token counts using `tiktoken`.
3. Retains the most recent user and assistant messages, dropping older messages from the middle if the total exceeds `max_token_budget`.

### Solution

```python
import tiktoken
from typing import List, Dict

def count_msg_tokens(msg: Dict[str, str], enc) -> int:
    # Overhead: ~4 tokens per message for role/delimiters in ChatML format
    return len(enc.encode(msg["content"])) + 4

def fit_messages_within_budget(
    messages: List[Dict[str, str]],
    max_token_budget: int = 1000,
    model_encoding: str = "cl100k_base"
) -> List[Dict[str, str]]:
    """
    Guarantees that the returned message list fits strictly within max_token_budget
    while preserving the system prompt and the most recent messages.
    """
    enc = tiktoken.get_encoding(model_encoding)

    if not messages:
        return []

    # 1. Isolate system message
    system_msg = None
    conversation = []
    for m in messages:
        if m.get("role") == "system" and system_msg is None:
            system_msg = m
        else:
            conversation.append(m)

    budget_remaining = max_token_budget
    result_messages = []

    # System message has top priority
    if system_msg:
        sys_tokens = count_msg_tokens(system_msg, enc)
        if sys_tokens > max_token_budget:
            raise ValueError("System prompt alone exceeds total token budget!")
        budget_remaining -= sys_tokens

    # 2. Iterate backwards from newest messages to oldest
    selected_recent = []
    for msg in reversed(conversation):
        msg_tokens = count_msg_tokens(msg, enc)
        if budget_remaining - msg_tokens >= 0:
            selected_recent.append(msg)
            budget_remaining -= msg_tokens
        else:
            # Cannot fit older message; stop
            break

    # Reconstruct in chronological order
    final_list = []
    if system_msg:
        final_list.append(system_msg)
    final_list.extend(reversed(selected_recent))

    return final_list

# Test run
if __name__ == "__main__":
    test_msgs = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Old message 1" * 100},
        {"role": "assistant", "content": "Old response 1" * 100},
        {"role": "user", "content": "Recent question: How do I sort in Python?"},
        {"role": "assistant", "content": "Use the sorted() function or list.sort()."}
    ]
    trimmed = fit_messages_within_budget(test_msgs, max_token_budget=200)
    print(f"Trimmed from {len(test_msgs)} down to {len(trimmed)} messages.")
    for m in trimmed:
        print(f"[{m['role']}]: {m['content'][:40]}...")
```
