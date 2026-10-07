"""Account tools. All of them sit behind the identity gate in LoanTool.run."""

from typing import Any

from sqlalchemy import select

from app.core.interfaces import CallContext, ToolSpec
from app.db.models import Loan, Payment
from app.tools.base import (
    LOAN_ID_PARAM,
    LoanTool,
    error,
    money,
    parse_amount,
    parse_date,
    parse_int,
    pick_loan,
)

MAX_HISTORY = 12
MAX_PROMISE_DAYS = 30


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


class GetLoanSummary(LoanTool):
    spec = ToolSpec(
        name="get_loan_summary",
        description="All of the verified caller's loans: principal, outstanding, EMI, status.",
        parameters={"type": "object", "properties": {}},
    )

    async def _run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        async with self._session() as session:
            loans = (
                await session.scalars(
                    select(Loan)
                    .where(Loan.customer_id == ctx.verified_customer_id)
                    .order_by(Loan.id)
                )
            ).all()
        return {
            "loans": [
                {
                    "loan_id": loan.id,
                    "principal": money(loan.principal),
                    "outstanding": money(loan.outstanding),
                    "emi_amount": money(loan.emi_amount),
                    "next_due_date": _iso(loan.next_due_date),
                    "status": loan.status,
                }
                for loan in loans
            ]
        }


class GetNextEmi(LoanTool):
    spec = ToolSpec(
        name="get_next_emi",
        description="The amount and due date of the verified caller's next EMI.",
        parameters={"type": "object", "properties": {"loan_id": LOAN_ID_PARAM}},
    )

    async def _run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        async with self._session() as session:
            loan = await pick_loan(session, ctx, args.get("loan_id"))
        if isinstance(loan, dict):
            return loan
        return {
            "loan_id": loan.id,
            "emi_amount": money(loan.emi_amount),
            "next_due_date": _iso(loan.next_due_date),
            "status": loan.status,
        }


class GetPaymentHistory(LoanTool):
    spec = ToolSpec(
        name="get_payment_history",
        description="The verified caller's most recent payments on a loan, newest first.",
        parameters={
            "type": "object",
            "properties": {
                "loan_id": LOAN_ID_PARAM,
                "limit": {"type": "integer", "description": "How many payments. Default 5."},
            },
        },
    )

    async def _run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        limit = parse_int(args.get("limit", 5)) or 5
        limit = max(1, min(limit, MAX_HISTORY))
        async with self._session() as session:
            loan = await pick_loan(session, ctx, args.get("loan_id"))
            if isinstance(loan, dict):
                return loan
            payments = (
                await session.scalars(
                    select(Payment)
                    .where(Payment.loan_id == loan.id)
                    .order_by(Payment.paid_on.desc(), Payment.id.desc())
                    .limit(limit)
                )
            ).all()
        return {
            "loan_id": loan.id,
            "payments": [
                {"amount": money(p.amount), "date": _iso(p.paid_on), "status": p.status}
                for p in payments
            ],
        }


class RecordPaymentPromise(LoanTool):
    spec = ToolSpec(
        name="record_payment_promise",
        description=(
            "Record the verified caller's promise to pay an amount on a date. "
            "Call it as soon as the caller has given both."
        ),
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Rupees the caller promises to pay."},
                "promise_date": {"type": "string", "description": "YYYY-MM-DD."},
                "loan_id": LOAN_ID_PARAM,
            },
            "required": ["amount", "promise_date"],
        },
    )

    async def _run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        amount = parse_amount(args.get("amount", ""))
        promise_date = parse_date(args.get("promise_date", ""))
        if amount is None or promise_date is None:
            return error(
                "invalid_arguments",
                "amount must be a positive number and promise_date must be YYYY-MM-DD.",
            )
        days_ahead = (promise_date - self._today()).days
        if days_ahead < 0:
            return error("date_in_past", "The promise date is in the past. Ask for a new date.")
        if days_ahead > MAX_PROMISE_DAYS:
            return error(
                "date_too_far",
                f"A promise can be at most {MAX_PROMISE_DAYS} days ahead. Ask for an earlier date.",
            )
        async with self._session() as session:
            loan = await pick_loan(session, ctx, args.get("loan_id"))
            if isinstance(loan, dict):
                return loan
            session.add(
                Payment(loan_id=loan.id, amount=amount, paid_on=promise_date, status="promised")
            )
            await session.commit()
        return {
            "recorded": True,
            "loan_id": loan.id,
            "amount": money(amount),
            "promise_date": promise_date.isoformat(),
        }
