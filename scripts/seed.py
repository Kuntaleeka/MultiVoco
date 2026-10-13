"""Fill the database with synthetic customers and loans. Owner: workstream E (Garlic).

    uv run python -m scripts.seed            # create tables and seed if empty
    uv run python -m scripts.seed --reset    # wipe the loan data and seed again

Twenty customers, five per language, with names typical of each. Every customer has a
different phone_last4 + dob pair, since that is how verify_identity finds them. Nobody
here is a real person. Due dates are set relative to --today, so a fresh seed always
has loans that are due soon and loans that are overdue.
"""

import argparse
import asyncio
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Base, Callback, Customer, Loan, Payment
from app.db.session import dispose_engine, get_engine, get_session_factory

# (name, language, phone_last4, date of birth)
CUSTOMERS: list[tuple[str, str, str, date]] = [
    ("Asha Rao", "en", "4821", date(1990, 4, 12)),
    ("Rahul Mehta", "en", "7754", date(1984, 11, 3)),
    ("Priya Nair", "en", "3069", date(1995, 7, 28)),
    ("Imran Sheikh", "en", "9142", date(1979, 2, 15)),
    ("Deepa Thomas", "en", "5580", date(1988, 9, 6)),
    ("सुनीता शर्मा", "hi", "2217", date(1982, 1, 20)),
    ("राजेश वर्मा", "hi", "6405", date(1975, 6, 9)),
    ("पूजा गुप्ता", "hi", "8836", date(1993, 12, 1)),
    ("अमित यादव", "hi", "1973", date(1987, 3, 25)),
    ("नेहा सिंह", "hi", "4390", date(1998, 8, 17)),
    ("ರವಿ ಕುಮಾರ್", "kn", "7310", date(1985, 1, 30)),
    ("ಲಕ್ಷ್ಮೀ ಗೌಡ", "kn", "5628", date(1991, 5, 14)),
    ("ಮಂಜುನಾಥ ಹೆಗ್ಡೆ", "kn", "9057", date(1978, 10, 22)),
    ("ಸವಿತಾ ಶೆಟ್ಟಿ", "kn", "3481", date(1989, 7, 4)),
    ("ಅನಿತಾ ಭಟ್", "kn", "6192", date(1996, 2, 11)),
    ("অনির্বাণ দাস", "bn", "8243", date(1983, 4, 27)),
    ("মৌসুমী ঘোষ", "bn", "1506", date(1992, 9, 19)),
    ("সুব্রত মুখার্জি", "bn", "7968", date(1976, 12, 8)),
    ("তানিয়া বসু", "bn", "2735", date(1994, 6, 2)),
    ("দেবাশীষ রায়", "bn", "4014", date(1986, 11, 23)),
]

# What each customer holds, cycling through the list: (status, days until the next EMI).
# A closed loan has no next EMI.
PORTFOLIOS: list[list[tuple[str, int | None]]] = [
    [("active", 12)],
    [("overdue", -6)],
    [("active", 20), ("closed", None)],
    [("overdue", -15), ("active", 9)],
    [("active", 3)],
]


def _money(value: int) -> Decimal:
    return Decimal(value).quantize(Decimal("0.01"))


def build(today: date) -> list[Customer]:
    """The whole loan book, as unsaved objects."""
    customers = []
    for index, (name, language, phone_last4, dob) in enumerate(CUSTOMERS):
        customer = Customer(name=name, phone_last4=phone_last4, dob=dob, language=language)
        for slot, (status, due_in) in enumerate(PORTFOLIOS[index % len(PORTFOLIOS)]):
            principal = 100_000 + 25_000 * ((index * 3 + slot * 7) % 13)
            emi = principal // 24 // 50 * 50
            paid_months = 0 if status == "closed" else 4 + (index + slot) % 9
            outstanding = 0 if status == "closed" else principal - emi * paid_months
            due = None if due_in is None else today + timedelta(days=due_in)
            loan = Loan(
                principal=_money(principal),
                outstanding=_money(outstanding),
                emi_amount=_money(emi),
                next_due_date=due,
                status=status,
            )
            # The three most recent payments. An overdue loan has a bounced one last.
            anchor = due or today
            for months_back in (3, 2, 1):
                paid_on = anchor - timedelta(days=30 * months_back)
                payment_status = "paid"
                if status == "overdue" and months_back == 1:
                    payment_status = "bounced"
                elif (index + months_back) % 4 == 0:
                    payment_status = "late"
                    paid_on += timedelta(days=5)
                loan.payments.append(
                    Payment(amount=_money(emi), paid_on=paid_on, status=payment_status)
                )
            customer.loans.append(loan)
        customers.append(customer)
    return customers


async def seed(session: AsyncSession, today: date, *, reset: bool = False) -> int:
    """Seed if the customers table is empty, or after wiping it. Returns customers added."""
    if reset:
        for table in (Callback, Payment, Loan, Customer):
            await session.execute(delete(table))
    elif await session.scalar(select(func.count()).select_from(Customer)):
        return 0
    customers = build(today)
    session.add_all(customers)
    await session.commit()
    return len(customers)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="wipe the loan data first")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD")
    args = parser.parse_args()

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with get_session_factory()() as session:
        added = await seed(session, args.today, reset=args.reset)
    await dispose_engine()
    print(f"Seeded {added} customers." if added else "Already seeded. Use --reset to start over.")


if __name__ == "__main__":
    asyncio.run(main())
