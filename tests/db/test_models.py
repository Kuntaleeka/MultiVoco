from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.core.languages import Lang
from app.core.trace import Mark, TurnTrace
from app.db.models import Base, Call, Customer, Loan, Payment, ToolCallRow, Turn


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


def test_schema_has_the_agreed_tables():
    assert set(Base.metadata.tables) == {
        "customers",
        "loans",
        "payments",
        "calls",
        "turns",
        "tool_calls",
    }


async def test_loan_data_round_trips(session):
    customer = Customer(name="ಅನಿತಾ ರಾವ್", phone_last4="4821", dob=date(1990, 4, 12), language="kn")
    loan = Loan(
        customer=customer,
        principal=Decimal("250000.00"),
        outstanding=Decimal("181250.50"),
        emi_amount=Decimal("8450.00"),
        next_due_date=date(2026, 11, 5),
        status="active",
    )
    loan.payments.append(
        Payment(amount=Decimal("8450.00"), paid_on=date(2026, 10, 5), status="paid")
    )
    session.add(customer)
    await session.commit()

    stored = (await session.execute(select(Loan).options(selectinload(Loan.payments)))).scalar_one()
    assert stored.outstanding == Decimal("181250.50")
    assert stored.payments[0].status == "paid"
    assert (await session.get(Customer, stored.customer_id)).name == "ಅನಿತಾ ರಾವ್"


async def test_phone_last4_and_dob_identify_one_customer(session):
    session.add(Customer(name="Asha Rao", phone_last4="4821", dob=date(1990, 4, 12), language="en"))
    await session.commit()

    session.add(Customer(name="Ravi Rao", phone_last4="4821", dob=date(1990, 4, 12), language="hi"))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()

    session.add(Customer(name="Ravi Rao", phone_last4="4821", dob=date(1991, 4, 12), language="hi"))
    await session.commit()


async def test_call_with_turn_trace_round_trips(session):
    trace = TurnTrace(call_id="call-1", idx=0, language=Lang.BN, user_text="আমার পরের EMI কবে?")
    trace.mark(Mark.SPEECH_END, 100.0)
    trace.mark(Mark.AUDIO_SENT, 820.0)

    call = Call(id="call-1", started_at=datetime.now(UTC), requested_lang="auto", final_lang="bn")
    turn = Turn(
        idx=trace.idx,
        language=trace.language.value,
        user_text=trace.user_text,
        agent_text="",
        interrupted=True,
        ms_played=640.0,
        trace=trace.to_dict(),
    )
    turn.tool_calls.append(
        ToolCallRow(
            name="get_next_emi", args={"loan_id": 1}, result={"amount": "8450.00"}, duration_ms=9.5
        )
    )
    call.turns.append(turn)
    session.add(call)
    await session.commit()

    stored = (
        await session.execute(select(Turn).options(selectinload(Turn.tool_calls)))
    ).scalar_one()
    assert stored.call_id == "call-1"
    assert stored.trace["time_to_first_audio_ms"] == 720.0
    assert stored.user_text == "আমার পরের EMI কবে?"
    assert stored.tool_calls[0].result == {"amount": "8450.00"}
