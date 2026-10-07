"""Database schema. Runs unchanged on SQLite (local, tests) and Postgres (Neon).

All customer and loan data is synthetic.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

Money = Numeric(12, 2)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    phone_last4: Mapped[str] = mapped_column(String(4))
    dob: Mapped[date] = mapped_column(Date)
    language: Mapped[str] = mapped_column(String(8))  # preferred Lang value

    loans: Mapped[list["Loan"]] = relationship(back_populates="customer")


class Loan(Base):
    __tablename__ = "loans"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    principal: Mapped[Decimal] = mapped_column(Money)
    outstanding: Mapped[Decimal] = mapped_column(Money)
    emi_amount: Mapped[Decimal] = mapped_column(Money)
    next_due_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(24))  # "active", "overdue", "closed"

    customer: Mapped[Customer] = relationship(back_populates="loans")
    payments: Mapped[list["Payment"]] = relationship(back_populates="loan")


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(primary_key=True)
    loan_id: Mapped[int] = mapped_column(ForeignKey("loans.id"), index=True)
    amount: Mapped[Decimal] = mapped_column(Money)
    paid_on: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(24))  # "paid", "late", "bounced", "promised"

    loan: Mapped[Loan] = relationship(back_populates="payments")


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    verified_customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"))
    outcome: Mapped[str | None] = mapped_column(String(24))
    handoff_reason: Mapped[str | None] = mapped_column(Text)
    requested_lang: Mapped[str] = mapped_column(String(8))  # "auto" or a Lang value
    final_lang: Mapped[str | None] = mapped_column(String(8))

    turns: Mapped[list["Turn"]] = relationship(back_populates="call", order_by="Turn.idx")


class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (UniqueConstraint("call_id", "idx"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"), index=True)
    idx: Mapped[int]
    language: Mapped[str] = mapped_column(String(8), index=True)
    user_text: Mapped[str] = mapped_column(Text, default="")
    agent_text: Mapped[str] = mapped_column(Text, default="")
    interrupted: Mapped[bool] = mapped_column(default=False)
    ms_played: Mapped[float | None]
    trace: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # TurnTrace.to_dict()

    call: Mapped[Call] = relationship(back_populates="turns")
    tool_calls: Mapped[list["ToolCallRow"]] = relationship(back_populates="turn")


class ToolCallRow(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("turns.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    args: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    duration_ms: Mapped[float]

    turn: Mapped[Turn] = relationship(back_populates="tool_calls")
