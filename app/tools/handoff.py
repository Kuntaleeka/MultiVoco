"""handoff_to_human: the model's way to end the automated part of the call."""

from typing import Any

from app.core.interfaces import CallContext, ToolSpec
from app.tools.base import LoanTool

HANDOFF_TOOL = "handoff_to_human"

REASONS = (
    "caller_request",
    "distress",
    "dispute",
    "out_of_scope",
    "repeated_misunderstanding",
    "verification_failed",
    "other",
)


class HandoffToHuman(LoanTool):
    requires_verification = False
    spec = ToolSpec(
        name=HANDOFF_TOOL,
        description=(
            "Transfer the call to a human agent. Use it when the caller asks for a person, "
            "is distressed, disputes a charge or a record, or wants something these tools "
            "cannot do."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "enum": list(REASONS)},
                "note": {"type": "string", "description": "One line for the human agent."},
            },
            "required": ["reason"],
        },
    )

    async def _run(self, args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        reason = str(args.get("reason", "other"))
        if reason not in REASONS:
            reason = "other"
        return {
            "handoff": True,
            "reason": reason,
            "message": "Transfer started. Tell the caller briefly, then stop.",
        }
