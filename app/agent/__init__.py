"""The agent. Owner: workstream B (Garlic). Importing this package registers it."""

from app.agent.agent import LoanAgent
from app.agent.llm import GEMINI, GROQ, build_llm, close_client
from app.core.registry import DEFAULT, register

register("agent", DEFAULT, lambda ctx: LoanAgent(ctx))
register("llm", GROQ, lambda lang: build_llm(GROQ))
register("llm", GEMINI, lambda lang: build_llm(GEMINI))


async def shutdown() -> None:
    """Called by app/pipeline/plugins.py when the app stops."""
    await close_client()


__all__ = ["LoanAgent", "shutdown"]
