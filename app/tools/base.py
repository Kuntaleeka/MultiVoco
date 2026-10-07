"""Shared pieces for the loan tools, including the identity gate (guardrail 1).

Tool names, arguments, and results are always in English, whatever the call language.
Results are read by the LLM, and every number in them becomes a figure the agent is
allowed to say (guardrail 2), so return exact values and nothing speculative.
"""

import re
from collections.abc import Callable
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.interfaces import CallContext, ToolSpec
from app.core.languages import to_ascii_digits
from app.db.models import Loan
from app.db.session import SessionFactory, get_session_factory

MAX_VERIFICATION_ATTEMPTS = 3

_DAY_FIRST = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})")


def error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": code, "message": message, **extra}


def money(value: Decimal) -> str:
    return f"{value:.2f}"


def parse_date(value: Any) -> date | None:
    """ISO (2026-11-05) or day-first (05/11/2026), with numerals from any of the four languages."""
    text = to_ascii_digits(str(value)).strip()
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    match = _DAY_FIRST.fullmatch(text)
    if match is None:
        return None
    day, month, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_amount(value: Any) -> Decimal | None:
    try:
        amount = Decimal(to_ascii_digits(str(value)).replace(",", "").strip())
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount <= 0:
        return None
    return amount.quantize(Decimal("0.01"))


def parse_int(value: Any) -> int | None:
    try:
        return int(to_ascii_digits(str(value)).strip())
    except ValueError:
        return None


class LoanTool:
    """Base for every tool. Account tools refuse to run until the caller is verified.

    The check is here, in code, so a model that ignores its prompt still gets nothing.
    """

    spec: ToolSpec
    requires_verification = True

    def __init__(
        self,
        sessions: SessionFactory | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._sessions = sessions
        self._today = today

    def _session(self) -> AsyncSession:
        return (self._sessions or get_session_factory())()

    async def run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        if self.requires_verification and ctx.verified_customer_id is None:
            return error(
                "not_verified",
                "The caller is not verified. Call verify_identity first, with the last 4 "
                "digits of their phone number and their date of birth.",
            )
        return await self._run(args, ctx)

    async def _run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        raise NotImplementedError


async def pick_loan(session: AsyncSession, ctx: CallContext, raw_loan_id: Any) -> Loan | dict:
    """The verified caller's loan, or an error result. Never another customer's loan."""
    loans = (
        await session.scalars(
            select(Loan).where(Loan.customer_id == ctx.verified_customer_id).order_by(Loan.id)
        )
    ).all()
    if raw_loan_id not in (None, ""):
        loan_id = parse_int(raw_loan_id)
        for loan in loans:
            if loan.id == loan_id:
                return loan
        return error("loan_not_found", "This caller has no loan with that ID.")
    candidates = [loan for loan in loans if loan.status != "closed"] or list(loans)
    if not candidates:
        return error("no_loans", "This caller has no loans.")
    if len(candidates) > 1:
        return error(
            "loan_id_required",
            "This caller has more than one loan. Ask which one, then pass loan_id.",
            loan_ids=[loan.id for loan in candidates],
        )
    return candidates[0]


LOAN_ID_PARAM = {
    "type": "integer",
    "description": "Which loan. Leave out if the caller has only one.",
}
