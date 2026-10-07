"""The agent. Owner: workstream B (Garlic). Importing this package registers it."""

from app.agent.agent import LoanAgent
from app.core.registry import DEFAULT, register

register("agent", DEFAULT, lambda ctx: LoanAgent(ctx))

__all__ = ["LoanAgent"]
