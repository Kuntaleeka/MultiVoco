"""A small synthetic loan book for the tools and agent tests (workstreams B and E).

Asha has one loan. Ravi has an overdue loan, an active loan, and a closed one.
"""

from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Customer, Loan, Payment

TODAY = date(2026, 10, 7)

ASHA = {"phone_last4": "4821", "dob": "1990-04-12"}
RAVI = {"phone_last4": "7310", "dob": "1985-01-30"}
ASHA_ID, RAVI_ID = 1, 2
ASHA_LOAN, RAVI_OVERDUE_LOAN, RAVI_ACTIVE_LOAN, RAVI_CLOSED_LOAN = 1, 2, 3, 4


def _loan(loan_id, customer_id, principal, outstanding, emi, due, status) -> Loan:
    return Loan(
        id=loan_id,
        customer_id=customer_id,
        principal=Decimal(principal),
        outstanding=Decimal(outstanding),
        emi_amount=Decimal(emi),
        next_due_date=due,
        status=status,
    )


@pytest.fixture
async def loan_db():
    """A session factory over a fresh in-memory database holding the loan book."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        session.add_all(
            [
                Customer(
                    id=ASHA_ID,
                    name="Asha Rao",
                    phone_last4="4821",
                    dob=date(1990, 4, 12),
                    language="en",
                ),
                Customer(
                    id=RAVI_ID,
                    name="ರವಿ ಕುಮಾರ್",
                    phone_last4="7310",
                    dob=date(1985, 1, 30),
                    language="kn",
                ),
                _loan(
                    ASHA_LOAN,
                    ASHA_ID,
                    "250000.00",
                    "181250.50",
                    "8450.00",
                    date(2026, 11, 5),
                    "active",
                ),
                _loan(
                    RAVI_OVERDUE_LOAN,
                    RAVI_ID,
                    "500000.00",
                    "420000.00",
                    "15200.00",
                    date(2026, 10, 1),
                    "overdue",
                ),
                _loan(
                    RAVI_ACTIVE_LOAN,
                    RAVI_ID,
                    "100000.00",
                    "64000.00",
                    "5100.00",
                    date(2026, 10, 20),
                    "active",
                ),
                _loan(RAVI_CLOSED_LOAN, RAVI_ID, "80000.00", "0.00", "4000.00", None, "closed"),
                Payment(
                    loan_id=ASHA_LOAN,
                    amount=Decimal("8450.00"),
                    paid_on=date(2026, 8, 5),
                    status="paid",
                ),
                Payment(
                    loan_id=ASHA_LOAN,
                    amount=Decimal("8450.00"),
                    paid_on=date(2026, 9, 9),
                    status="late",
                ),
                Payment(
                    loan_id=ASHA_LOAN,
                    amount=Decimal("8450.00"),
                    paid_on=date(2026, 10, 5),
                    status="paid",
                ),
            ]
        )
        await session.commit()
    yield sessions
    await engine.dispose()
