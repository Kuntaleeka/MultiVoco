"""The loan tools the agent can call. Owner: workstream B (Garlic)."""

from collections.abc import Callable
from datetime import date

from app.db.session import SessionFactory
from app.tools.account import (
    GetLoanSummary,
    GetNextEmi,
    GetPaymentHistory,
    RecordPaymentPromise,
    RequestCallback,
)
from app.tools.base import MAX_VERIFICATION_ATTEMPTS, LoanTool
from app.tools.handoff import HANDOFF_TOOL, HandoffToHuman
from app.tools.identity import VerifyIdentity

TOOL_CLASSES: tuple[type[LoanTool], ...] = (
    VerifyIdentity,
    GetLoanSummary,
    GetNextEmi,
    GetPaymentHistory,
    RecordPaymentPromise,
    RequestCallback,
    HandoffToHuman,
)


def build_tools(
    sessions: SessionFactory | None = None, today: Callable[[], date] = date.today
) -> dict[str, LoanTool]:
    """One set of tools, keyed by name. Tools hold no per-call state, so a set can be shared."""
    tools = [cls(sessions, today) for cls in TOOL_CLASSES]
    return {tool.spec.name: tool for tool in tools}


VERIFY_TOOL = VerifyIdentity.spec.name

__all__ = ["HANDOFF_TOOL", "MAX_VERIFICATION_ATTEMPTS", "VERIFY_TOOL", "LoanTool", "build_tools"]
