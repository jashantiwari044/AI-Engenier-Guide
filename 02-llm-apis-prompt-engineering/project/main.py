"""
Phase 2: LLM APIs & Prompt Engineering Masterclass CLI.

Demonstrating:
1. Prompt Engineering Arena (Zero-Shot vs Few-Shot vs Chain-of-Thought vs Persona)
2. Temperature & Sampling Lab (0.0 vs 0.7 vs 1.4)
3. Production Structured Output Article Generator (Pydantic Schema Validation)
4. Real-time Token Streaming & Live Cost Ticker
5. Multi-Model Token Cost & BPE Tokenizer Lab
"""

import asyncio
import os
import sys
import time
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from rich.prompt import Prompt

from models import GeneratedArticle, TokenCostBreakdown
from prompt_templates import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_chain_of_thought_prompt,
    build_persona_prompt
)
from providers import get_provider, count_tokens_with_tiktoken

load_dotenv()
console = Console()


def print_banner(provider_name: str):
    """Renders the welcoming banner."""
    title = Text("🧠 Phase 2: LLM APIs & Prompt Engineering Masterclass", style="bold magenta")
    subtitle = Text(
        f"Active Provider: {provider_name}\n"
        "BPE Tokenization • Sampling Parameters • CoT & Few-Shot • Structured Outputs",
        style="dim white"
    )
    banner = Text.assemble(title, "\n\n", subtitle)
    console.print(Panel(banner, border_style="magenta", padding=(1, 2)))


async def demo_prompt_engineering_arena():
    """Compares Zero-Shot, Few-Shot, Chain-of-Thought, and Persona prompting side-by-side."""
    console.print("\n[bold yellow]══════ 1. The Prompt Engineering Arena ══════[/bold yellow]")
    console.print("[dim]Comparing 4 prompt engineering strategies on the exact same challenge...[/dim]\n")

    topic = "Why do LLMs hallucinate, and how do we prevent it in production?"
    provider = get_provider()

    # 1. Zero-Shot
    console.print("[bold cyan]▶ 1. Zero-Shot Prompting[/bold cyan] (Direct task instruction)")
    t0 = time.perf_counter()
    zero_shot_prompt = build_zero_shot_prompt(topic)
    resp_zero, m_zero = await provider.generate_completion(zero_shot_prompt, temperature=0.5)
    lat_zero = (time.perf_counter() - t0) * 1000
    console.print(Panel(
        resp_zero,
        title=f"Zero-Shot Result ({m_zero.total_tokens} tokens | {lat_zero:.1f}ms)",
        border_style="cyan"
    ))

    # 2. Few-Shot
    console.print("\n[bold green]▶ 2. Few-Shot In-Context Learning[/bold green] (Steered via exemplars)")
    t0 = time.perf_counter()
    few_shot_prompt = build_few_shot_prompt(topic)
    resp_few, m_few = await provider.generate_completion(few_shot_prompt, temperature=0.5)
    lat_few = (time.perf_counter() - t0) * 1000
    console.print(Panel(
        resp_few,
        title=f"Few-Shot Result ({m_few.total_tokens} tokens | {lat_few:.1f}ms)",
        border_style="green"
    ))

    # 3. Chain-of-Thought
    console.print("\n[bold magenta]▶ 3. Chain-of-Thought (CoT)[/bold magenta] (Explicit reasoning scratchpad)")
    t0 = time.perf_counter()
    cot_prompt = build_chain_of_thought_prompt(topic)
    resp_cot, m_cot = await provider.generate_completion(cot_prompt, temperature=0.5)
    lat_cot = (time.perf_counter() - t0) * 1000
    console.print(Panel(
        resp_cot,
        title=f"Chain-of-Thought Result ({m_cot.total_tokens} tokens | {lat_cot:.1f}ms)",
        border_style="magenta"
    ))

    # Summary Table
    table = Table(title="Prompt Engineering Strategy Comparison", border_style="yellow")
    table.add_column("Strategy", style="bold")
    table.add_column("Prompt Approach", style="dim")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Cost (USD)", justify="right", style="green")

    table.add_row("Zero-Shot", "Direct instruction, no examples", str(m_zero.input_tokens), str(m_zero.output_tokens), f"${m_zero.calculate_cost_usd():.6f}")
    table.add_row("Few-Shot", "Exemplar pattern steering", str(m_few.input_tokens), str(m_few.output_tokens), f"${m_few.calculate_cost_usd():.6f}")
    table.add_row("Chain-of-Thought", "Explicit reasoning scratchpad", str(m_cot.input_tokens), str(m_cot.output_tokens), f"${m_cot.calculate_cost_usd():.6f}")
    console.print(table)


async def demo_temperature_sampling_lab():
    """Demonstrates how temperature impacts entropy, creativity, and determinism."""
    console.print("\n[bold yellow]══════ 2. Temperature & Sampling Hyperparameter Lab ══════[/bold yellow]")
    console.print("[dim]Evaluating temperature=0.0 (greedy) vs 0.7 (balanced) vs 1.4 (high entropy)...[/dim]\n")

    prompt = "Create a 1-sentence tagline for an AI Engineering platform."
    provider = get_provider()

    temps = [0.0, 0.7, 1.4]
    descriptions = {
        0.0: "Deterministic / Greedy (Code, Math, JSON Extraction)",
        0.7: "Balanced / Production Default (Q&A, Summaries, General)",
        1.4: "High Entropy / Creative (Brainstorming, Fiction, Divergence)"
    }

    table = Table(title="Temperature Sampling Impact", border_style="cyan")
    table.add_column("Temperature", justify="center", style="bold yellow")
    table.add_column("Sampling Profile", style="cyan")
    table.add_column("Generated Tagline", style="white")

    for temp in temps:
        out, _ = await provider.generate_completion(prompt, temperature=temp, max_tokens=100)
        table.add_row(f"{temp:.1f}", descriptions[temp], out.strip().replace("\n", " "))

    console.print(table)


async def demo_structured_article_generator():
    """Generates a complete technical article strictly validated against Pydantic schema."""
    console.print("\n[bold yellow]══════ 3. Production Structured Output Article Generator ══════[/bold yellow]")
    console.print("[dim]Forcing the LLM to output strictly validated JSON conforming to GeneratedArticle Pydantic schema...[/dim]\n")

    topic = "Hybrid Search and Re-Ranking in Advanced RAG"
    provider = get_provider()

    t0 = time.perf_counter()
    article, metrics = await provider.generate_structured_article(topic, audience="intermediate")
    duration = time.perf_counter() - t0

    console.print(f"[bold green]✓ Article Generated & Validated via Pydantic in {duration:.2f}s![/bold green]\n")

    # Overview Table
    meta_table = Table(title="Generated Article Metadata", border_style="green")
    meta_table.add_column("Property", style="bold magenta")
    meta_table.add_column("Value", style="white")

    meta_table.add_row("Title", article.title)
    meta_table.add_row("Slug", article.slug)
    meta_table.add_row("Audience Level", article.target_audience.upper())
    meta_table.add_row("Est. Reading Time", f"{article.reading_time_minutes} minutes")
    meta_table.add_row("SEO Meta Description", article.seo_meta_description)
    meta_table.add_row("Tags", ", ".join(article.tags))
    meta_table.add_row("Total Tokens / Cost", f"{metrics.total_tokens} tokens (${metrics.calculate_cost_usd():.6f} USD)")
    console.print(meta_table)

    # Sections Preview
    for i, section in enumerate(article.sections, 1):
        content_text = Text(section.content, style="white")
        takeaways_text = "\n".join(f"• {t}" for t in section.key_takeaways)
        full_section = Text.assemble(content_text, "\n\n", Text("Key Takeaways:\n", style="bold green"), Text(takeaways_text, style="green"))
        console.print(Panel(full_section, title=f"Section {i}: {section.heading}", border_style="blue"))


async def demo_token_streaming():
    """Streams tokens in real-time with an active cost and token meter."""
    console.print("\n[bold yellow]══════ 4. Real-Time Token Streaming & Cost Ticker ══════[/bold yellow]")
    console.print("[dim]Streaming response token-by-token with live token accumulation meter...[/dim]\n")

    prompt = "Explain how Byte Pair Encoding (BPE) turns raw characters into tokens."
    provider = get_provider()

    console.print(f"[bold cyan]User Prompt:[/bold cyan] {prompt}\n")
    console.print("[bold green]Assistant:[/bold green] ", end="")

    last_metrics = None
    async for chunk, metrics in provider.stream_completion(prompt):
        console.print(chunk, end="", style="bold white")
        sys.stdout.flush()
        last_metrics = metrics

    console.print("\n")
    if last_metrics:
        cost = last_metrics.calculate_cost_usd()
        stats = (
            f"[dim]Model:[/dim] [bold]{last_metrics.model_name}[/bold] | "
            f"[dim]Input Tokens:[/dim] [bold]{last_metrics.input_tokens}[/bold] | "
            f"[dim]Output Tokens:[/dim] [bold]{last_metrics.output_tokens}[/bold] | "
            f"[dim]Total:[/dim] [bold]{last_metrics.total_tokens}[/bold] | "
            f"[dim]Cost:[/dim] [bold yellow]${cost:.6f} USD[/bold yellow]"
        )
        console.print(Panel(stats, border_style="cyan"))


def demo_token_calculator_lab():
    """Demonstrates tiktoken BPE token counting and multi-model cost analysis."""
    console.print("\n[bold yellow]══════ 5. BPE Token Counting & Multi-Model Cost Lab ══════[/bold yellow]")
    console.print("[dim]Analyzing token length and cost projections across 6 industry foundation models...[/dim]\n")

    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an architectural pattern that combines "
        "pre-trained foundation language models with an external, dynamic retrieval mechanism "
        "to ground generations in authoritative facts."
    )

    tokens_count = count_tokens_with_tiktoken(sample_text)
    chars_count = len(sample_text)
    words_count = len(sample_text.split())

    console.print(f"[white]Sample String:[/white] \"[dim]{sample_text}[/dim]\"")
    console.print(
        f"[bold cyan]Characters:[/bold cyan] {chars_count} | "
        f"[bold cyan]Words:[/bold cyan] {words_count} | "
        f"[bold green]Tokens (BPE cl100k):[/bold green] {tokens_count} "
        f"[dim](~{chars_count/tokens_count:.2f} chars/token)[/dim]\n"
    )

    # Cost Matrix for a 10,000 prompt / 2,000 completion workload
    in_tokens = 10_000
    out_tokens = 2_000

    table = Table(title=f"Workload Cost Comparison ({in_tokens:,} Input + {out_tokens:,} Output Tokens)", border_style="magenta")
    table.add_column("Model Provider", style="bold")
    table.add_column("Model Name", style="cyan")
    table.add_column("Input Rate / 1M", justify="right")
    table.add_column("Output Rate / 1M", justify="right")
    table.add_column("Estimated Total Cost", justify="right", style="bold green")

    models = [
        ("OpenAI", "gpt-4o-mini", "$0.15", "$0.60"),
        ("OpenAI", "gpt-4o", "$2.50", "$10.00"),
        ("Anthropic", "claude-3-haiku", "$0.25", "$1.25"),
        ("Anthropic", "claude-3-5-sonnet", "$3.00", "$15.00"),
        ("Google", "gemini-1.5-flash", "$0.075", "$0.30"),
        ("Google", "gemini-1.5-pro", "$1.25", "$5.00"),
    ]

    for provider_name, model_name, in_rate, out_rate in models:
        breakdown = TokenCostBreakdown(
            model_name=model_name,
            input_tokens=in_tokens,
            output_tokens=out_tokens
        )
        cost = breakdown.calculate_cost_usd()
        table.add_row(provider_name, model_name, in_rate, out_rate, f"${cost:.4f}")

    console.print(table)


async def main():
    provider = get_provider()
    print_banner(provider.provider_name)

    await demo_prompt_engineering_arena()
    await demo_temperature_sampling_lab()
    await demo_structured_article_generator()
    await demo_token_streaming()
    demo_token_calculator_lab()

    console.print("\n" + "═" * 72)
    console.print(
        "[bold green]✨ Phase 2: LLM APIs & Prompt Engineering Mastered![/bold green]\n"
        "You now understand:\n"
        " • BPE Tokenization mechanics & cost optimization\n"
        " • The trade-offs of Zero-Shot vs Few-Shot vs Chain-of-Thought\n"
        " • Sampling entropy (Temperature, Top_p)\n"
        " • Multi-provider abstraction (OpenAI, Anthropic, Gemini, Mock)\n"
        " • Production structured output enforcement with Pydantic\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
