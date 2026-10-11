from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.languages import Lang
from app.db.models import Base, Customer, Loan, Payment
from scripts.seed import CUSTOMERS, build, seed

TODAY = date(2026, 10, 7)


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        yield session
    await engine.dispose()


def test_customers_are_spread_evenly_over_the_four_languages():
    languages = [language for _, language, _, _ in CUSTOMERS]
    assert len(CUSTOMERS) == 20
    assert {lang: languages.count(lang.value) for lang in Lang} == dict.fromkeys(Lang, 5)


def test_every_customer_can_be_told_apart_by_phone_and_dob():
    pairs = [(phone, dob) for _, _, phone, dob in CUSTOMERS]
    assert len(set(pairs)) == len(pairs)
    assert all(len(phone) == 4 and phone.isdigit() for phone, _ in pairs)


def test_the_book_covers_the_loan_states_the_agent_must_handle():
    customers = build(TODAY)
    statuses = [loan.status for customer in customers for loan in customer.loans]
    assert {"active", "overdue", "closed"} <= set(statuses)
    assert any(len(customer.loans) > 1 for customer in customers)
    for customer in customers:
        for loan in customer.loans:
            assert 0 <= loan.outstanding <= loan.principal
            assert (loan.next_due_date is None) == (loan.status == "closed")
            if loan.status == "overdue":
                assert loan.next_due_date < TODAY
                assert loan.payments[-1].status == "bounced"
            if loan.status == "active":
                assert loan.next_due_date > TODAY


async def test_seeding_twice_adds_nothing_and_reset_starts_over(session):
    assert await seed(session, TODAY) == 20
    assert await seed(session, TODAY) == 0
    assert await seed(session, TODAY, reset=True) == 20
    for table, expected in ((Customer, 20), (Loan, 28), (Payment, 84)):
        assert await session.scalar(select(func.count()).select_from(table)) == expected
