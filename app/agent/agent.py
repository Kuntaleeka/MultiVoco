"""The loan agent: one per call. Owns the history, the tool loop, and the guardrails.

Text reaches the orchestrator one checked sentence at a time. A sentence is held until
it has passed the figure check and the language check, because a TextDelta cannot be
taken back once it is yielded.
"""

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.agent.guardrails import FigureLedger, SentenceBuffer, wrong_language
from app.agent.lines import GREETINGS, HANDOFF_LINE, NEUTRAL_GREETING, SAFE_REPLY
from app.agent.prompt import build_system_prompt
from app.core.interfaces import (
    LLM,
    AgentEvent,
    CallContext,
    Done,
    Handoff,
    Message,
    ModelFirstToken,
    TextDelta,
    ToolCall,
    ToolResult,
)
from app.core.languages import LANGUAGES, Lang
from app.core.registry import get_llm
from app.tools import HANDOFF_TOOL, MAX_VERIFICATION_ATTEMPTS, LoanTool, build_tools

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4
# Turns in a row that ended in the safe reply, or with nothing heard, before a human takes over.
MAX_MISSES = 3
# Set by the orchestrator in ctx.extra while the session language is still being detected.
LANGUAGE_PENDING = "language_pending"


@dataclass
class _Attempt:
    """What one pass through the tool loop left behind."""

    spoken: list[str] = field(default_factory=list)
    rejected: tuple[str, str] | None = None  # (sentence, problem)
    handoff: str | None = None


class LoanAgent:
    def __init__(
        self,
        ctx: CallContext | None = None,
        *,
        tools: dict[str, LoanTool] | None = None,
        llm_for: Callable[[Lang], LLM] = get_llm,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._tools = tools if tools is not None else build_tools(today=today)
        self._llm_for = llm_for
        self._today = today
        self._ledger = FigureLedger()
        self._ledger.add(today().isoformat())
        self.history: list[Message] = []
        # The current turn's user message and tool exchanges, kept apart until commit_spoken.
        self._pending: list[Message] = []
        self._misses = 0
        self._handed_off: str | None = None
        self._first_token_sent = False

    def greeting(self, ctx: CallContext) -> str:
        if ctx.extra.get(LANGUAGE_PENDING):
            return NEUTRAL_GREETING
        return GREETINGS[ctx.language]

    def commit_spoken(self, spoken_text: str, interrupted: bool) -> None:
        self.history.extend(self._pending)
        self._pending = []
        if spoken_text.strip():
            self.history.append(Message("assistant", spoken_text.strip()))

    async def respond(self, user_text: str, ctx: CallContext) -> AsyncIterator[AgentEvent]:
        # A turn the orchestrator never committed was not heard: keep its facts, not its reply.
        self.history.extend(self._pending)
        self._pending = [Message("user", user_text)]
        self._first_token_sent = False

        if self._handed_off or ctx.failed_verifications >= MAX_VERIFICATION_ATTEMPTS:
            reason = self._handed_off or "verification_failed"
            async for event in self._hand_off(ctx, reason, say_line=True):
                yield event
            return

        if not user_text.strip():
            async for event in self._miss(ctx):
                yield event
            return

        spoken: list[str] = []
        rejected: tuple[str, str] | None = None
        handoff: str | None = None
        for _ in range(2):
            attempt = _Attempt(spoken=spoken)
            async for event in self._attempt(ctx, attempt, rejected):
                yield event
            # Only the first attempt reports the model's first token.
            self._first_token_sent = True
            rejected = attempt.rejected
            handoff = handoff or attempt.handoff
            if rejected is None:
                break
            log.warning("reply rejected by guardrail: %s", rejected[1])

        if handoff:
            async for event in self._hand_off(ctx, handoff, say_line=not spoken):
                yield event
            return
        if rejected is not None or not spoken:
            async for event in self._miss(ctx, prefix=_gap(spoken)):
                yield event
            return
        self._misses = 0
        yield Done("stop")

    async def _attempt(
        self, ctx: CallContext, attempt: _Attempt, rejected: tuple[str, str] | None
    ) -> AsyncIterator[AgentEvent]:
        """One pass: stream the model, release checked sentences, run tools, repeat."""
        llm = self._llm_for(ctx.language)
        specs = [tool.spec for tool in self._tools.values()]
        for _ in range(MAX_TOOL_ROUNDS):
            system = build_system_prompt(
                ctx, self._today(), rejected=rejected, spoken="".join(attempt.spoken)
            )
            messages = [Message("system", system), *self.history, *self._pending]
            buffer = SentenceBuffer()
            calls: list[ToolCall] = []

            async with aclosing(llm.stream(messages, specs)) as stream:
                async for event in stream:
                    if isinstance(event, ToolCall):
                        calls.append(event)
                    elif isinstance(event, TextDelta):
                        if not self._first_token_sent:
                            self._first_token_sent = True
                            yield ModelFirstToken()
                        for sentence in buffer.feed(event.text):
                            if delta := self._release(sentence, ctx, attempt):
                                yield delta
                            else:
                                return
            if tail := buffer.flush():
                if delta := self._release(tail, ctx, attempt):
                    yield delta
                else:
                    return
            if not calls:
                return

            self._pending.append(Message("assistant", "", tool_calls=tuple(calls)))
            for call in calls:
                result, duration_ms = await self._run_tool(call, ctx)
                self._ledger.add(result)
                self._pending.append(
                    Message("tool", json.dumps(result, ensure_ascii=False), tool_call_id=call.id)
                )
                yield ToolResult(call.name, call.args, result, duration_ms)
                if call.name == HANDOFF_TOOL and result.get("handoff"):
                    attempt.handoff = str(result.get("reason", "other"))
                elif result.get("locked"):
                    attempt.handoff = "verification_failed"
        log.warning("tool loop did not finish in %d rounds", MAX_TOOL_ROUNDS)

    def _release(self, sentence: str, ctx: CallContext, attempt: _Attempt) -> TextDelta | None:
        """The sentence as a delta if it passes the guardrails. Otherwise record why not."""
        if figures := self._ledger.unsupported(sentence):
            attempt.rejected = (
                sentence.strip(),
                f"it states {', '.join(figures)}, which no tool result in this call contains. "
                "Call the tool for the figure, or reply without it. Write figures in digits.",
            )
            return None
        if wrong_language(sentence, ctx.language):
            attempt.rejected = (
                sentence.strip(),
                f"it is not in {LANGUAGES[ctx.language].name}, the session language.",
            )
            return None
        text = _gap(attempt.spoken) + sentence
        attempt.spoken.append(text)
        return TextDelta(text)

    async def _run_tool(self, call: ToolCall, ctx: CallContext) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            result: dict[str, Any] = {"error": "unknown_tool", "message": "No such tool."}
        else:
            try:
                result = await tool.run(call.args if isinstance(call.args, dict) else {}, ctx)
            except Exception:
                # A broken tool or database must not end the call. No details go to the model.
                log.exception("tool %s failed", call.name)
                result = {
                    "error": "tool_failed",
                    "message": "The tool is unavailable right now. Apologise, and do not guess.",
                }
        return result, (time.perf_counter() - started) * 1000

    async def _miss(self, ctx: CallContext, prefix: str = "") -> AsyncIterator[AgentEvent]:
        """Nothing usable this turn: say the safe line, or hand over after too many in a row."""
        self._misses += 1
        if self._misses >= MAX_MISSES:
            async for event in self._hand_off(
                ctx, "repeated_misunderstanding", say_line=True, prefix=prefix
            ):
                yield event
            return
        yield TextDelta(prefix + SAFE_REPLY[ctx.language])
        yield Done("stop")

    async def _hand_off(
        self, ctx: CallContext, reason: str, *, say_line: bool, prefix: str = ""
    ) -> AsyncIterator[AgentEvent]:
        self._handed_off = reason
        if say_line:
            yield TextDelta(prefix + HANDOFF_LINE[ctx.language])
        yield Handoff(reason)
        yield Done("handoff")


def _gap(spoken: list[str]) -> str:
    """A space to put before the next piece, if what was already spoken does not end in one."""
    return " " if spoken and not spoken[-1][-1:].isspace() else ""
