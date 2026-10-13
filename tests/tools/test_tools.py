from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.core.interfaces import CallContext, Tool
from app.core.languages import Lang
from app.db.models import Callback, Payment
from app.tools import MAX_VERIFICATION_ATTEMPTS, build_tools
from tests.loan_fixtures import (
    ASHA,
    ASHA_ID,
    ASHA_LOAN,
    RAVI,
    RAVI_ACTIVE_LOAN,
    RAVI_OVERDUE_LOAN,
    TODAY,
)

ACCOUNT_TOOLS = [
    "get_loan_summary",
    "get_next_emi",
    "get_payment_history",
    "record_payment_promise",
    "request_callback",
]


@pytest.fixture
def tools(loan_db):
    return build_tools(loan_db, today=lambda: TODAY)


@pytest.fixture
def ctx():
    return CallContext("call-1", Lang.EN)


async def verified(tools, ctx, who):
    result = await tools["verify_identity"].run(dict(who), ctx)
    assert result["verified"] is True
    return ctx


def test_every_tool_meets_the_contract(tools):
    assert set(tools) == {*ACCOUNT_TOOLS, "verify_identity", "handoff_to_human"}
    for name, tool in tools.items():
        assert isinstance(tool, Tool)
        assert tool.spec.name == name
        assert tool.spec.parameters["type"] == "object"


# --- Guardrail 1: the identity gate ------------------------------------------------


@pytest.mark.parametrize("name", ACCOUNT_TOOLS)
async def test_account_tools_refuse_before_verification(tools, ctx, name, loan_db):
    args = {
        "loan_id": ASHA_LOAN,
        "amount": 5000,
        "promise_date": "2026-10-10",
        "callback_date": "2026-10-10",
    }
    result = await tools[name].run(args, ctx)
    assert result["error"] == "not_verified"
    assert not {"loans", "emi_amount", "payments", "recorded", "booked"} & set(result)
    async with loan_db() as session:
        assert (await session.scalars(select(Callback))).all() == []
    async with loan_db() as session:
        promised = await session.scalars(select(Payment).where(Payment.status == "promised"))
        assert promised.all() == []


async def test_verification_sets_the_customer_on_the_call(tools, ctx):
    result = await tools["verify_identity"].run(dict(ASHA), ctx)
    assert result == {"verified": True, "customer_name": "Asha Rao", "preferred_language": "en"}
    assert ctx.verified_customer_id == ASHA_ID
    assert ctx.failed_verifications == 0


@pytest.mark.parametrize(
    "args",
    [
        {"phone_last4": "४८२१", "dob": "१९९०-०४-१२"},  # Devanagari numerals
        {"phone_last4": "೪೮೨೧", "dob": "೧೯೯೦-೦೪-೧೨"},  # Kannada numerals
        {"phone_last4": "৪৮২১", "dob": "১৯৯০-০৪-১২"},  # Bengali numerals
        {"phone_last4": "48 21", "dob": " 1990-04-12 "},  # spaced digits
    ],
)
async def test_verification_reads_digits_from_every_language(tools, ctx, args):
    assert (await tools["verify_identity"].run(args, ctx))["verified"] is True
    assert ctx.verified_customer_id == ASHA_ID


@pytest.mark.parametrize("dob", ["04/12/1990", "12/04/1990", "12 April 1990"])
async def test_a_date_that_is_not_iso_is_refused_not_guessed_at(tools, ctx, dob):
    """Speech recognition writes dates month first, so a slashed date cannot be trusted."""
    result = await tools["verify_identity"].run({"phone_last4": "4821", "dob": dob}, ctx)
    assert result["error"] == "invalid_arguments"
    assert ctx.failed_verifications == 0


async def test_right_phone_with_wrong_dob_does_not_verify(tools, ctx):
    result = await tools["verify_identity"].run({**ASHA, "dob": "1990-04-13"}, ctx)
    assert result == {"verified": False, "attempts_left": 2}
    assert ctx.verified_customer_id is None
    assert ctx.failed_verifications == 1


async def test_malformed_arguments_are_not_a_failed_attempt(tools, ctx):
    for args in ({"phone_last4": "821", "dob": "1990-04-12"}, {"phone_last4": "4821"}, {}):
        assert (await tools["verify_identity"].run(args, ctx))["error"] == "invalid_arguments"
    assert ctx.failed_verifications == 0


async def test_verification_locks_after_three_failures(tools, ctx):
    wrong = {"phone_last4": "0000", "dob": "2000-01-01"}
    results = [
        await tools["verify_identity"].run(wrong, ctx) for _ in range(MAX_VERIFICATION_ATTEMPTS)
    ]
    assert [r.get("attempts_left") for r in results] == [2, 1, None]
    assert results[-1]["locked"] is True

    # The right details no longer work, and the account stays closed.
    after = await tools["verify_identity"].run(dict(ASHA), ctx)
    assert after["locked"] is True
    assert ctx.verified_customer_id is None
    assert (await tools["get_loan_summary"].run({}, ctx))["error"] == "not_verified"


# --- Account tools -----------------------------------------------------------------


async def test_loan_summary_lists_only_the_callers_loans(tools, ctx):
    await verified(tools, ctx, ASHA)
    result = await tools["get_loan_summary"].run({}, ctx)
    assert result == {
        "loans": [
            {
                "loan_id": ASHA_LOAN,
                "principal": "250000.00",
                "outstanding": "181250.50",
                "emi_amount": "8450.00",
                "next_due_date": "2026-11-05",
                "status": "active",
            }
        ]
    }


async def test_next_emi_for_a_single_loan_needs_no_loan_id(tools, ctx):
    await verified(tools, ctx, ASHA)
    assert await tools["get_next_emi"].run({}, ctx) == {
        "loan_id": ASHA_LOAN,
        "emi_amount": "8450.00",
        "next_due_date": "2026-11-05",
        "status": "active",
    }


async def test_two_open_loans_need_a_loan_id(tools, ctx):
    await verified(tools, ctx, RAVI)
    result = await tools["get_next_emi"].run({}, ctx)
    assert result["error"] == "loan_id_required"
    assert result["loan_ids"] == [
        RAVI_OVERDUE_LOAN,
        RAVI_ACTIVE_LOAN,
    ]  # the closed loan is left out

    chosen = await tools["get_next_emi"].run({"loan_id": RAVI_ACTIVE_LOAN}, ctx)
    assert chosen["emi_amount"] == "5100.00"


@pytest.mark.parametrize("name", ["get_next_emi", "get_payment_history", "record_payment_promise"])
async def test_another_customers_loan_is_never_returned(tools, ctx, name):
    await verified(tools, ctx, RAVI)
    args = {"loan_id": ASHA_LOAN, "amount": 5000, "promise_date": "2026-10-10"}
    result = await tools[name].run(args, ctx)
    assert result["error"] == "loan_not_found"
    assert "8450" not in str(result)


async def test_payment_history_is_newest_first_and_limited(tools, ctx):
    await verified(tools, ctx, ASHA)
    result = await tools["get_payment_history"].run({"limit": 2}, ctx)
    assert result == {
        "loan_id": ASHA_LOAN,
        "payments": [
            {"amount": "8450.00", "date": "2026-10-05", "status": "paid"},
            {"amount": "8450.00", "date": "2026-09-09", "status": "late"},
        ],
    }


async def test_payment_promise_is_stored(tools, ctx, loan_db):
    await verified(tools, ctx, ASHA)
    result = await tools["record_payment_promise"].run(
        {"amount": "5,000", "promise_date": "2026-10-10"}, ctx
    )
    assert result == {
        "recorded": True,
        "loan_id": ASHA_LOAN,
        "amount": "5000.00",
        "promise_date": "2026-10-10",
    }
    async with loan_db() as session:
        row = await session.scalar(select(Payment).where(Payment.status == "promised"))
    assert (row.loan_id, row.amount, row.paid_on) == (
        ASHA_LOAN,
        Decimal("5000.00"),
        date(2026, 10, 10),
    )


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"amount": 5000, "promise_date": "2026-10-06"}, "date_in_past"),
        ({"amount": 5000, "promise_date": "2026-12-25"}, "date_too_far"),
        ({"amount": -5, "promise_date": "2026-10-10"}, "invalid_arguments"),
        ({"amount": "lots", "promise_date": "2026-10-10"}, "invalid_arguments"),
        ({"amount": 5000, "promise_date": "next week"}, "invalid_arguments"),
    ],
)
async def test_a_bad_payment_promise_is_refused_and_not_stored(tools, ctx, loan_db, args, code):
    await verified(tools, ctx, ASHA)
    assert (await tools["record_payment_promise"].run(args, ctx))["error"] == code
    async with loan_db() as session:
        assert await session.scalar(select(Payment).where(Payment.status == "promised")) is None


async def test_a_callback_is_booked_for_the_caller_on_this_call(tools, ctx, loan_db):
    await verified(tools, ctx, ASHA)
    result = await tools["request_callback"].run(
        {"callback_date": "2026-10-09", "time_of_day": "evening", "note": "Wants to prepay"}, ctx
    )
    assert result == {"booked": True, "callback_date": "2026-10-09", "time_of_day": "evening"}
    async with loan_db() as session:
        row = await session.scalar(select(Callback))
    assert (row.customer_id, row.call_id, row.requested_for) == (
        ASHA_ID,
        "call-1",
        date(2026, 10, 9),
    )
    assert (row.time_of_day, row.note) == ("evening", "Wants to prepay")
    assert row.created_at is not None


async def test_a_callback_needs_only_a_date(tools, ctx):
    await verified(tools, ctx, ASHA)
    result = await tools["request_callback"].run(
        {"callback_date": "2026-10-07", "time_of_day": "whenever", "note": None}, ctx
    )
    assert result == {"booked": True, "callback_date": "2026-10-07", "time_of_day": None}


@pytest.mark.parametrize(
    ("callback_date", "code"),
    [("2026-10-06", "date_in_past"), ("2026-11-30", "date_too_far"), ("soon", "invalid_arguments")],
)
async def test_a_bad_callback_date_is_refused_and_not_stored(
    tools, ctx, loan_db, callback_date, code
):
    await verified(tools, ctx, ASHA)
    result = await tools["request_callback"].run({"callback_date": callback_date}, ctx)
    assert result["error"] == code
    async with loan_db() as session:
        assert await session.scalar(select(Callback)) is None


async def test_handoff_needs_no_verification_and_cleans_the_reason(tools, ctx):
    result = await tools["handoff_to_human"].run({"reason": "distress"}, ctx)
    assert (result["handoff"], result["reason"]) == (True, "distress")
    odd = await tools["handoff_to_human"].run({"reason": "because"}, ctx)
    assert odd["reason"] == "other"
    assert ctx.verified_customer_id is None
