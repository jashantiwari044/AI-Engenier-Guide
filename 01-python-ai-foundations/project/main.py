"""
Phase 1 Showcase: Python + AI Foundations
Interactive CLI demonstrating:
1. High-concurrency async network calls with Semaphore throttling
2. Pydantic v2 data validation and error handling
3. Dynamic JSON schema export for LLM Tool-Calling
4. Simulated real-time LLM token streaming with token & cost tracking
"""

import asyncio
import json
import os
import sys
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax
from pydantic import ValidationError

from models import (
    GeoLocation,
    CurrentWeather,
    WeatherReport,
    ToolDefinition,
    ChatMessage
)
from api_client import ResilientAIClient

# Load environment variables
load_dotenv()
console = Console()


def print_banner():
    """Renders the welcoming banner."""
    title = Text("🚀 Phase 1: Python + AI Foundations Toolkit", style="bold cyan")
    subtitle = Text(
        "Mastering AsyncIO • Pydantic v2 • Resilient HTTP • Tool Calling Schemas\n"
        "Environment: Python 3.12 (macOS / Homebrew)",
        style="dim white"
    )
    banner_content = Text.assemble(title, "\n\n", subtitle)
    console.print(Panel(banner_content, border_style="cyan", padding=(1, 2)))


async def demo_concurrent_api():
    """Demonstrates parallel async fetching of real API data with concurrency control."""
    console.print("\n[bold yellow]═══ 1. High-Concurrency Async API Execution ═══[/bold yellow]")
    console.print("[dim]Fetching live global weather data in parallel using httpx + asyncio.gather + Semaphore...[/dim]\n")

    cities = ["London", "Tokyo", "New York", "San Francisco", "Sydney"]
    start_time = time.perf_counter()

    async with ResilientAIClient(max_concurrent=3, timeout_seconds=10.0) as client:
        reports = await client.get_weather_batch(cities)

    total_duration = time.perf_counter() - start_time

    table = Table(title="Live Weather Data (Validated with Pydantic)", border_style="green", header_style="bold magenta")
    table.add_column("City", style="bold white")
    table.add_column("Country", style="cyan")
    table.add_column("Coordinates", style="dim")
    table.add_column("Temperature", justify="right", style="bold yellow")
    table.add_column("Condition", style="green")
    table.add_column("API Latency", justify="right", style="blue")

    for resp in reports:
        if resp.success and resp.data:
            report: WeatherReport = resp.data
            loc = report.location
            w = report.weather
            temp_str = f"{w.temperature_celsius:.1f}°C ({w.temperature_fahrenheit:.1f}°F)"
            coords_str = f"{loc.latitude:.2f}, {loc.longitude:.2f}"
            table.add_row(
                loc.city_name,
                loc.country,
                coords_str,
                temp_str,
                w.condition_summary,
                f"{report.retrieval_latency_ms:.1f} ms"
            )
        else:
            table.add_row("Error", "-", "-", "-", f"[red]{resp.error_message}[/red]", f"{resp.latency_ms:.1f} ms")

    console.print(table)
    console.print(
        f"[bold green]✓[/bold green] Fetched {len(cities)} cities concurrently in "
        f"[bold cyan]{total_duration:.2f}s[/bold cyan] "
        f"[dim](Estimated sequential time: ~{sum(r.latency_ms for r in reports)/1000:.2f}s)[/dim]"
    )


def demo_pydantic_validation():
    """Demonstrates strict schema enforcement and validation error handling."""
    console.print("\n[bold yellow]═══ 2. Pydantic v2 Schema Enforcement & Validation ═══[/bold yellow]")
    console.print("[dim]Testing edge case: Invalid latitude (120.0°) and empty city name...[/dim]\n")

    try:
        GeoLocation(
            city_name="   ",  # Invalid (fails min_length or whitespace strip)
            country="Wonderland",
            latitude=120.0,   # Invalid: Exceeds 90 degrees
            longitude=-45.0
        )
    except ValidationError as err:
        console.print("[bold red]Pydantic Caught Validation Error Gracefully:[/bold red]")
        for e in err.errors():
            loc = " -> ".join(str(p) for p in e["loc"])
            msg = e["msg"]
            console.print(f"  [red]• Field '[bold]{loc}[/bold]': {msg}[/red]")

    # Valid instantiation
    valid_loc = GeoLocation(
        city_name="Berlin",
        country="Germany",
        latitude=52.52,
        longitude=13.405
    )
    console.print(f"\n[bold green]✓ Valid Model Instantiation:[/bold green] {valid_loc.city_name} ({valid_loc.latitude}, {valid_loc.longitude})")


def demo_tool_calling_schema():
    """Demonstrates generating OpenAI/Anthropic tool calling JSON schemas."""
    console.print("\n[bold yellow]═══ 3. Dynamic Tool Definition Schema (OpenAI / Anthropic Spec) ═══[/bold yellow]")
    console.print("[dim]Generating function schema directly from GeoLocation Pydantic model for LLMs...[/dim]\n")

    tool_def = ToolDefinition.from_pydantic_model(
        name="lookup_location_coordinates",
        description="Resolves geographic coordinates and timezone information for a given city.",
        model_cls=GeoLocation
    )

    schema_json = json.dumps(tool_def.model_dump(), indent=2)
    syntax = Syntax(schema_json, "json", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, title="LLM Function Tool Specification", border_style="magenta"))
    console.print("[bold green]✓[/bold green] LLM can now accurately invoke this tool with type-checked arguments.")


async def demo_token_streaming():
    """Demonstrates real-time token streaming with cost and token estimation."""
    console.print("\n[bold yellow]═══ 4. Real-Time LLM Token Streaming Simulation ═══[/bold yellow]")
    console.print("[dim]Streaming response token-by-token with live token counting and USD cost estimation...[/dim]\n")

    client = ResilientAIClient()
    prompt = "Why is asynchronous programming essential for AI Engineers?"

    console.print(f"[bold cyan]User Prompt:[/bold cyan] {prompt}\n")
    console.print("[bold green]Assistant:[/bold green] ", end="")

    last_metrics = None
    async for chunk, metrics in client.stream_simulated_ai_response(prompt):
        console.print(chunk, end="", style="bold white")
        sys.stdout.flush()
        last_metrics = metrics

    console.print("\n")
    if last_metrics:
        cost = last_metrics.calculate_estimated_cost_usd()
        stats = (
            f"[dim]Model:[/dim] [bold]{last_metrics.model_name}[/bold] | "
            f"[dim]Prompt Tokens:[/dim] [bold]{last_metrics.prompt_tokens}[/bold] | "
            f"[dim]Output Tokens:[/dim] [bold]{last_metrics.completion_tokens}[/bold] | "
            f"[dim]Total Tokens:[/dim] [bold]{last_metrics.total_tokens}[/bold] | "
            f"[dim]Estimated Cost:[/dim] [bold yellow]${cost:.6f} USD[/bold yellow]"
        )
        console.print(Panel(stats, border_style="blue"))


async def main():
    print_banner()
    await demo_concurrent_api()
    demo_pydantic_validation()
    demo_tool_calling_schema()
    await demo_token_streaming()

    console.print("\n" + "═" * 70)
    console.print(
        "[bold green]✨ Phase 1 Python Foundations Complete![/bold green]\n"
        "You now have rock-solid mastery of:\n"
        " • Async network I/O with concurrency throttling\n"
        " • Pydantic v2 data models, validation & schema generation\n"
        " • Resilient error handling with exponential backoff\n"
        " • Token stream consumption and metrics tracking\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
