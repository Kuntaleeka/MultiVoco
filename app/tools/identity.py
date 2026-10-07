"""verify_identity: the only way a call becomes verified."""

from typing import Any

from sqlalchemy import select

from app.core.interfaces import CallContext, ToolSpec
from app.core.languages import to_ascii_digits
from app.db.models import Customer
from app.tools.base import MAX_VERIFICATION_ATTEMPTS, LoanTool, error, parse_date


class VerifyIdentity(LoanTool):
    requires_verification = False
    spec = ToolSpec(
        name="verify_identity",
        description=(
            "Verify the caller. Needs the last 4 digits of their registered phone number "
            "and their date of birth. Call it only once the caller has given both."
        ),
        parameters={
            "type": "object",
            "properties": {
                "phone_last4": {
                    "type": "string",
                    "description": "Exactly 4 digits, for example 4821.",
                },
                "dob": {
                    "type": "string",
                    "description": "Date of birth as YYYY-MM-DD, for example 1990-04-12.",
                },
            },
            "required": ["phone_last4", "dob"],
        },
    )

    async def _run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        if ctx.verified_customer_id is not None:
            return {"verified": True, "already_verified": True}
        if ctx.failed_verifications >= MAX_VERIFICATION_ATTEMPTS:
            return _locked()

        phone = "".join(
            ch for ch in to_ascii_digits(str(args.get("phone_last4", ""))) if ch.isdigit()
        )
        dob = parse_date(args.get("dob", ""))
        if len(phone) != 4 or dob is None:
            # A malformed call is the model's mistake, not a failed attempt by the caller.
            return error(
                "invalid_arguments",
                "phone_last4 must be exactly 4 digits and dob must be YYYY-MM-DD. "
                "Ask the caller again for whichever is missing.",
            )

        async with self._session() as session:
            customer = await session.scalar(
                select(Customer).where(Customer.phone_last4 == phone, Customer.dob == dob)
            )
        if customer is None:
            ctx.failed_verifications += 1
            if ctx.failed_verifications >= MAX_VERIFICATION_ATTEMPTS:
                return _locked()
            return {
                "verified": False,
                "attempts_left": MAX_VERIFICATION_ATTEMPTS - ctx.failed_verifications,
            }

        ctx.verified_customer_id = customer.id
        return {
            "verified": True,
            "customer_name": customer.name,
            "preferred_language": customer.language,
        }


def _locked() -> dict[str, Any]:
    return {
        "verified": False,
        "locked": True,
        "message": "Too many failed attempts. The call is being handed to a human agent.",
    }
