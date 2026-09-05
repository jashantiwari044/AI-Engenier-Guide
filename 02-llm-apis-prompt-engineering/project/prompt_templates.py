"""
Production-Grade Prompt Engineering Templates & Prompt Builders.

Implements:
1. Zero-Shot Prompting
2. Few-Shot In-Context Learning
3. Chain-of-Thought (CoT) Reasoning (Scratchpad / Step-by-Step)
4. Persona / Role Conditioning
5. Structured Output Prompt Formatting
"""

from typing import Dict, List, Optional
from models import PromptTechnique


# ============================================================================
# 1. System Prompts & Personas
# ============================================================================

PERSONA_SYSTEM_PROMPTS: Dict[str, str] = {
    "technical_lead": (
        "You are a Staff AI Systems Architect at a top-tier tech company. "
        "Your explanations are deep, precise, technically accurate, and focused on production reliability, "
        "scalability, and architectural trade-offs. Avoid marketing buzzwords."
    ),
    "friendly_teacher": (
        "You are an inspiring, beginner-friendly AI educator. "
        "You use clear real-world analogies, encouraging language, and intuitive explanations. "
        "You break down complex jargon so that anyone can understand."
    ),
    "executive_advisor": (
        "You are a Chief AI Strategist advising Fortune 500 executives. "
        "Your responses are concise, direct, high-impact, and focused on business value, "
        "cost-benefit analysis, security, and measurable ROI."
    ),
}


# ============================================================================
# 2. Few-Shot Exemplars
# ============================================================================

FEW_SHOT_EXEMPLARS = [
    {
        "input": "Explain why vector databases are needed for LLMs.",
        "output": (
            "1. Problem: LLMs have fixed context windows and cannot remember private company data.\n"
            "2. Solution: Convert documents into high-dimensional numerical vectors (embeddings).\n"
            "3. Mechanism: Vector DBs perform approximate nearest neighbor (ANN) search using cosine similarity.\n"
            "4. Benefit: Sub-millisecond retrieval of the top-k most semantically relevant chunks for RAG."
        )
    },
    {
        "input": "What is the difference between Fine-Tuning and RAG?",
        "output": (
            "1. RAG (Retrieval-Augmented Generation): Injects external factual context at query time. "
            "Best for frequently updating dynamic data and source attribution.\n"
            "2. Fine-Tuning: Modifies the underlying model weights on domain datasets. "
            "Best for specialized style, syntax, formatting, or niche jargon.\n"
            "3. Rule of Thumb: Use RAG for knowledge; use Fine-Tuning for behavior and form."
        )
    }
]


# ============================================================================
# 3. Prompt Builder Functions
# ============================================================================

def build_zero_shot_prompt(topic: str) -> str:
    """Builds a direct, unambiguous zero-shot prompt with clear delimiters."""
    return (
        f"Topic to explain: \"{topic}\"\n\n"
        "Provide a structured, authoritative explanation covering:\n"
        "- Core Definition\n"
        "- Why it matters in modern AI systems\n"
        "- Practical Production Use-Case\n"
        "- 1 Critical Pitfall or Gotcha to avoid"
    )


def build_few_shot_prompt(topic: str) -> str:
    """Constructs an in-context few-shot prompt with demonstrated exemplars."""
    prompt_lines = [
        "Analyze the given AI engineering concept following the exact format of the examples below.\n"
    ]

    for i, ex in enumerate(FEW_SHOT_EXEMPLARS, 1):
        prompt_lines.append(f"### Example {i}")
        prompt_lines.append(f"Input: {ex['input']}")
        prompt_lines.append(f"Output:\n{ex['output']}\n")

    prompt_lines.append("### New Task")
    prompt_lines.append(f"Input: Explain {topic}")
    prompt_lines.append("Output:")
    return "\n".join(prompt_lines)


def build_chain_of_thought_prompt(topic_or_problem: str) -> str:
    """
    Constructs a Chain-of-Thought (CoT) prompt enforcing step-by-step reasoning.
    Uses explicit scratchpad tags (<thinking> and <final_answer>).
    """
    return (
        f"Task: Analyze and solve the following engineering architectural challenge:\n"
        f"\"{topic_or_problem}\"\n\n"
        "Instructions:\n"
        "1. First, reason through the problem step-by-step inside <thinking>...</thinking> tags.\n"
        "   - Break down the requirements\n"
        "   - Analyze bottlenecks and constraints\n"
        "   - Evaluate 2 alternative approaches\n"
        "2. Then, provide your definitive recommendation inside <final_answer>...</final_answer> tags."
    )


def build_persona_prompt(topic: str, persona_key: str = "technical_lead") -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) configured for a specific persona."""
    system_prompt = PERSONA_SYSTEM_PROMPTS.get(
        persona_key,
        PERSONA_SYSTEM_PROMPTS["technical_lead"]
    )
    user_prompt = (
        f"Provide your expert breakdown of: \"{topic}\".\n"
        "Ensure your tone, depth, and vocabulary reflect your designated role."
    )
    return system_prompt, user_prompt


def build_structured_article_prompt(topic: str, audience: str = "intermediate") -> tuple[str, str]:
    """
    Constructs instructions for generating a complete technical article
    that strictly satisfies the GeneratedArticle Pydantic schema.
    """
    system_prompt = (
        "You are an elite technical author and AI engineer. "
        "When generating technical articles, you provide actionable code patterns, "
        "production insights, and clear section breakdowns. "
        "Your output must adhere strictly to the requested schema."
    )

    user_prompt = (
        f"Generate a comprehensive, production-ready technical article about: \"{topic}\".\n"
        f"Target Audience Level: {audience}\n\n"
        "Requirements:\n"
        "- Compelling title and clean URL slug\n"
        "- Executive summary (2-3 sentences)\n"
        "- At least 2 thorough sections, each with a heading, body content, and 2-4 key takeaway bullet points\n"
        "- SEO meta description between 50 and 160 characters\n"
        "- 3 to 6 lowercase technology tags"
    )
    return system_prompt, user_prompt
