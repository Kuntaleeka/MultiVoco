"""The agent in text mode, driven by a scripted LLM. No network, no keys."""

import pytest

import app.agent  # noqa: F401  (registers the agent)
from app.agent.agent import LANGUAGE_PENDING, MAX_MISSES, LoanAgent
from app.agent.lines import GREETINGS, HANDOFF_LINE, NEUTRAL_GREETING, SAFE_REPLY
from app.core import registry
from app.core.interfaces import (
    Agent,
    CallContext,
    Done,
    Handoff,
    ModelFirstToken,
    TextDelta,
    ToolCall,
    ToolResult,
)
from app.core.languages import Lang
from app.tools import build_tools
from tests.loan_fixtures import ASHA, ASHA_ID, TODAY

# A reply with no figure, a reply stating the real EMI, and one inventing a figure.
PLAIN = {
    Lang.EN: "Sure, let me check that for you.",
    Lang.HI: "जी, मैं अभी देखती हूँ।",
    Lang.KN: "ಖಂಡಿತ, ನಾನು ಈಗ ನೋಡುತ್ತೇನೆ.",
    Lang.BN: "নিশ্চয়ই, আমি এখনই দেখছি।",
}
REAL_EMI = {
    Lang.EN: "Your next EMI is 8450.00, due on 5 November 2026.",
    Lang.HI: "आपकी अगली EMI ८४५० रुपये है, ५ नवंबर २०२६ को।",
    Lang.KN: "ನಿಮ್ಮ ಮುಂದಿನ EMI ೮೪೫೦ ರೂಪಾಯಿ, ನವೆಂಬರ್ ೫ ರಂದು.",
    Lang.BN: "আপনার পরের EMI ৮৪৫০ টাকা, ৫ নভেম্বর।",
}
INVENTED = {
    Lang.EN: "Your outstanding balance is 99999 rupees.",
    Lang.HI: "आपका बकाया ९९९९९ रुपये है।",
    Lang.KN: "ನಿಮ್ಮ ಬಾಕಿ ೯೯೯೯೯ ರೂಪಾಯಿ.",
    Lang.BN: "আপনার বকেয়া ৯৯৯৯৯ টাকা।",
}
ALL_LANGS = list(Lang)


def say(text: str) -> list[TextDelta]:
    """A reply arriving the way a model streams it: in small pieces."""
    words = text.split(" ")
    return [TextDelta(word + (" " if i < len(words) - 1 else "")) for i, word in enumerate(words)]


def call(name: str, **args) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, args=args)


class ScriptedLLM:
    """Plays back one list of events per stream() call, and records what it was sent."""

    def __init__(self, *rounds: list) -> None:
        self.rounds = list(rounds)
        self.seen: list[list] = []

    async def stream(self, messages, tools):
        self.seen.append(list(messages))
        assert self.rounds, "the agent asked the model for more than the test scripted"
        for event in self.rounds.pop(0):
            yield event
        yield Done()


def make_agent(loan_db, lang: Lang, *rounds: list, verified: bool = False):
    ctx = CallContext("call-1", lang)
    if verified:
        ctx.verified_customer_id = ASHA_ID
    llm = ScriptedLLM(*rounds)
    agent = LoanAgent(
        ctx,
        tools=build_tools(loan_db, today=lambda: TODAY),
        llm_for=lambda _lang: llm,
        today=lambda: TODAY,
    )
    return agent, ctx, llm


async def run(agent, ctx, user_text: str = "When is my next EMI due?") -> list:
    events = [event async for event in agent.respond(user_text, ctx)]
    # The orchestrator's splitter holds a sentence until it sees what follows the full
    # stop, so every delta must end in whitespace (section 12, note 12).
    assert all(e.text[-1:].isspace() for e in events if isinstance(e, TextDelta))
    return events


def text_of(events: list) -> str:
    """What the caller hears, without the space that ends the last delta."""
    return "".join(event.text for event in events if isinstance(event, TextDelta)).rstrip()


def kinds(events: list) -> list[type]:
    return [type(event) for event in events]


# --- The contract ------------------------------------------------------------------


async def test_the_registry_hands_out_the_real_agent_when_it_is_not_mocked(settings_env):
    settings_env(MOCK="vad,stt,langid,llm,tts,trace_sink")
    ctx = CallContext("call-1", Lang.EN)
    agent = registry.get_agent(ctx)
    assert isinstance(agent, LoanAgent)
    assert isinstance(agent, Agent)


async def test_a_reply_reaches_the_caller_one_checked_sentence_at_a_time(loan_db):
    agent, ctx, llm = make_agent(
        loan_db,
        Lang.EN,
        [call("verify_identity", **ASHA)],
        [call("get_next_emi")],
        say("Thank you, Asha. " + REAL_EMI[Lang.EN]),
    )
    events = await run(agent, ctx, "4821 and 12 April 1990. When is my next EMI?")

    assert kinds(events) == [ToolResult, ToolResult, ModelFirstToken, TextDelta, TextDelta, Done]
    assert [e.text for e in events if isinstance(e, TextDelta)] == [
        "Thank you, Asha. ",
        REAL_EMI[Lang.EN] + " ",
    ]
    assert events[0].result["verified"] is True
    assert events[1].result["emi_amount"] == "8450.00"
    assert events[1].duration_ms >= 0
    assert ctx.verified_customer_id == ASHA_ID
    # The model saw each tool result before it wrote the reply.
    assert [m.role for m in llm.seen[2]][-4:] == ["assistant", "tool", "assistant", "tool"]


async def test_words_before_a_tool_call_are_released_before_the_tool_runs(loan_db):
    agent, ctx, _ = make_agent(
        loan_db,
        Lang.EN,
        [*say("Let me check."), call("get_next_emi")],
        say(REAL_EMI[Lang.EN]),
        verified=True,
    )
    events = await run(agent, ctx)
    assert kinds(events) == [ModelFirstToken, TextDelta, ToolResult, TextDelta, Done]
    assert events[1] == TextDelta("Let me check. ")
    assert text_of(events) == "Let me check. " + REAL_EMI[Lang.EN]


@pytest.mark.parametrize("lang", ALL_LANGS)
async def test_greeting_is_in_the_session_language_or_neutral_while_detecting(loan_db, lang):
    agent, ctx, _ = make_agent(loan_db, lang)
    assert agent.greeting(ctx) == GREETINGS[lang]
    ctx.extra[LANGUAGE_PENDING] = True
    assert agent.greeting(ctx) == NEUTRAL_GREETING


# --- Guardrail 1: the identity gate ------------------------------------------------


@pytest.mark.parametrize("lang", ALL_LANGS)
async def test_account_data_cannot_be_reached_before_verification(loan_db, lang):
    """The model ignores its prompt and goes straight for the account. It gets nothing."""
    agent, ctx, llm = make_agent(
        loan_db,
        lang,
        [call("get_next_emi"), call("get_loan_summary"), call("get_payment_history")],
        say(REAL_EMI[lang]),  # it then tries to state a figure anyway
        say(PLAIN[lang]),
    )
    events = await run(agent, ctx)

    results = [e.result for e in events if isinstance(e, ToolResult)]
    assert [r["error"] for r in results] == ["not_verified"] * 3
    assert "8450" not in str(results)
    assert text_of(events) == PLAIN[lang]
    assert ctx.verified_customer_id is None


async def test_three_failed_verifications_hand_the_call_to_a_human(loan_db):
    wrong = {"phone_last4": "0000", "dob": "2000-01-01"}
    agent, ctx, _ = make_agent(
        loan_db,
        Lang.EN,
        [call("verify_identity", **wrong)],
        say("That did not match. Could you try once more?"),
        [call("verify_identity", **wrong)],
        say("That did not match. Could you try once more?"),
        [call("verify_identity", **wrong)],
        say("I could not verify you, so I am transferring you to a colleague."),
    )
    first = await run(agent, ctx, "0000, 1 January 2000")
    second = await run(agent, ctx, "0000, 1 January 2000")
    third = await run(agent, ctx, "0000, 1 January 2000")

    assert Handoff not in kinds(first) + kinds(second)
    assert third[-2:] == [Handoff("verification_failed"), Done("handoff")]
    assert ctx.failed_verifications == 3

    # Nothing more reaches the model, even with the right details.
    after = await run(agent, ctx, "4821, 12 April 1990")
    assert text_of(after) == HANDOFF_LINE[Lang.EN]
    assert after[-2:] == [Handoff("verification_failed"), Done("handoff")]
    assert ctx.verified_customer_id is None


# --- Guardrail 2: tool-only figures ------------------------------------------------


@pytest.mark.parametrize("lang", ALL_LANGS)
async def test_a_figure_from_a_tool_result_is_spoken(loan_db, lang):
    agent, ctx, _ = make_agent(
        loan_db, lang, [call("get_next_emi")], say(REAL_EMI[lang]), verified=True
    )
    events = await run(agent, ctx)
    assert text_of(events) == REAL_EMI[lang]
    assert events[-1] == Done("stop")


@pytest.mark.parametrize("lang", ALL_LANGS)
async def test_an_invented_figure_is_never_spoken(loan_db, lang):
    """The model invents a figure twice. The caller hears the safe line instead."""
    agent, ctx, llm = make_agent(
        loan_db, lang, say(INVENTED[lang]), say(INVENTED[lang]), verified=True
    )
    events = await run(agent, ctx, "Just tell me roughly what I owe.")

    assert text_of(events) == SAFE_REPLY[lang]
    assert kinds(events) == [ModelFirstToken, TextDelta, Done]  # first attempt only
    assert len(llm.seen) == 2
    # The second attempt was told what was wrong with the first.
    assert "99999" in llm.seen[1][0].content
    assert "99999" not in llm.seen[0][0].content


async def test_a_rejected_reply_is_regenerated_and_can_then_use_the_tool(loan_db):
    agent, ctx, llm = make_agent(
        loan_db,
        Lang.EN,
        say("Sure. Your EMI is around 8000 rupees."),
        [call("get_next_emi")],
        say(REAL_EMI[Lang.EN]),
        verified=True,
    )
    events = await run(agent, ctx)

    assert text_of(events) == "Sure. " + REAL_EMI[Lang.EN]
    assert kinds(events).count(ModelFirstToken) == 1
    # The retry knows what the caller already heard, so it does not repeat it.
    assert "already heard: 'Sure. '" in llm.seen[1][0].content
    assert events[-1] == Done("stop")


async def test_a_figure_the_caller_supplied_is_not_confirmed(loan_db):
    agent, ctx, _ = make_agent(
        loan_db,
        Lang.EN,
        say("Yes, your balance is 50000."),
        say("Yes, that is right, it is 50000."),
        verified=True,
    )
    events = await run(agent, ctx, "My balance is 50000, right?")
    assert text_of(events) == SAFE_REPLY[Lang.EN]


async def test_figures_stay_allowed_on_later_turns_of_the_same_call(loan_db):
    agent, ctx, _ = make_agent(
        loan_db,
        Lang.EN,
        [call("get_next_emi")],
        say(REAL_EMI[Lang.EN]),
        say("It is 8450.00, as I said."),
        verified=True,
    )
    first = await run(agent, ctx)
    agent.commit_spoken(text_of(first), interrupted=False)
    second = await run(agent, ctx, "Sorry, how much?")
    assert text_of(second) == "It is 8450.00, as I said."


# --- Reply language ----------------------------------------------------------------


@pytest.mark.parametrize("lang", [Lang.KN, Lang.BN])
async def test_a_reply_in_the_wrong_language_is_regenerated(loan_db, lang):
    agent, ctx, llm = make_agent(
        loan_db, lang, say(PLAIN[Lang.EN]), say(PLAIN[lang]), verified=True
    )
    events = await run(agent, ctx)
    assert text_of(events) == PLAIN[lang]
    assert "session language" in llm.seen[1][0].content


async def test_the_prompt_follows_a_language_switch_between_turns(loan_db):
    agent, ctx, llm = make_agent(loan_db, Lang.EN, say(PLAIN[Lang.EN]), say(PLAIN[Lang.KN]))
    agent.commit_spoken(text_of(await run(agent, ctx)), interrupted=False)
    ctx.language = Lang.KN
    events = await run(agent, ctx, "ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?")

    assert text_of(events) == PLAIN[Lang.KN]
    assert "Session language: English" in llm.seen[0][0].content
    assert "Session language: Kannada" in llm.seen[1][0].content
    # The history survives the switch.
    assert [m.content for m in llm.seen[1][1:]] == [
        "When is my next EMI due?",
        PLAIN[Lang.EN],
        "ನನ್ನ ಮುಂದಿನ EMI ಯಾವಾಗ?",
    ]


# --- Guardrail 3: handoff ----------------------------------------------------------


@pytest.mark.parametrize("lang", ALL_LANGS)
async def test_the_model_can_hand_the_call_to_a_human(loan_db, lang):
    agent, ctx, _ = make_agent(
        loan_db, lang, [call("handoff_to_human", reason="caller_request")], say(PLAIN[lang])
    )
    events = await run(agent, ctx, "I want to talk to a person.")
    assert kinds(events) == [ToolResult, ModelFirstToken, TextDelta, Handoff, Done]
    assert events[-2:] == [Handoff("caller_request"), Done("handoff")]

    # The call stays handed off.
    again = await run(agent, ctx, "Hello?")
    assert text_of(again) == HANDOFF_LINE[lang]
    assert again[-2:] == [Handoff("caller_request"), Done("handoff")]


async def test_a_handoff_with_no_closing_words_still_tells_the_caller(loan_db):
    agent, ctx, _ = make_agent(loan_db, Lang.HI, [call("handoff_to_human", reason="distress")], [])
    events = await run(agent, ctx, "मैं बहुत परेशान हूँ।")
    assert text_of(events) == HANDOFF_LINE[Lang.HI]
    assert events[-2:] == [Handoff("distress"), Done("handoff")]


@pytest.mark.parametrize("lang", ALL_LANGS)
async def test_repeated_misunderstanding_hands_the_call_to_a_human(loan_db, lang):
    agent, ctx, llm = make_agent(loan_db, lang)
    for _ in range(MAX_MISSES - 1):
        events = await run(agent, ctx, "   ")
        assert text_of(events) == SAFE_REPLY[lang]
        assert events[-1] == Done("stop")
    events = await run(agent, ctx, "")
    assert text_of(events) == HANDOFF_LINE[lang]
    assert events[-2:] == [Handoff("repeated_misunderstanding"), Done("handoff")]
    assert llm.seen == []


async def test_a_good_turn_resets_the_misunderstanding_count(loan_db):
    agent, ctx, _ = make_agent(loan_db, Lang.EN, say(PLAIN[Lang.EN]))
    await run(agent, ctx, "")
    await run(agent, ctx, "")
    await run(agent, ctx, "Hello, can you hear me?")
    events = await run(agent, ctx, "")
    assert Handoff not in kinds(events)


# --- History and failures ----------------------------------------------------------


async def test_history_keeps_what_was_heard_and_the_tool_facts(loan_db):
    agent, ctx, llm = make_agent(
        loan_db,
        Lang.EN,
        [call("get_next_emi")],
        say(REAL_EMI[Lang.EN]),
        say("It is due on 5 November 2026."),
        verified=True,
    )
    await run(agent, ctx)
    agent.commit_spoken("Your next EMI is", interrupted=True)  # barge-in

    assert [(m.role, m.content) for m in agent.history if m.role != "tool"] == [
        ("user", "When is my next EMI due?"),
        ("assistant", ""),  # the tool call
        ("assistant", "Your next EMI is"),
    ]
    assert REAL_EMI[Lang.EN] not in [m.content for m in agent.history]

    # The tool result is still there, so the next reply can state the date.
    events = await run(agent, ctx, "Sorry, when?")
    assert text_of(events) == "It is due on 5 November 2026."
    assert any(m.role == "tool" and "2026-11-05" in m.content for m in llm.seen[-1])


async def test_a_turn_that_was_never_committed_leaves_no_reply_behind(loan_db):
    agent, ctx, llm = make_agent(loan_db, Lang.EN, say(PLAIN[Lang.EN]), say(PLAIN[Lang.EN]))
    await run(agent, ctx, "first")
    await run(agent, ctx, "second")
    assert [(m.role, m.content) for m in llm.seen[1][1:]] == [("user", "first"), ("user", "second")]


async def test_a_failing_tool_does_not_end_the_call(loan_db):
    class Broken:
        spec = build_tools(loan_db)["get_next_emi"].spec

        async def run(self, args, ctx):
            raise RuntimeError("database is down: host=db.internal password=hunter2")

    ctx = CallContext("call-1", Lang.EN, verified_customer_id=ASHA_ID)
    llm = ScriptedLLM([call("get_next_emi")], say("Sorry, I cannot check that right now."))
    agent = LoanAgent(ctx, tools={"get_next_emi": Broken()}, llm_for=lambda _: llm)
    events = await run(agent, ctx)

    assert events[0].result["error"] == "tool_failed"
    assert "hunter2" not in str(events[0].result)
    assert text_of(events) == "Sorry, I cannot check that right now."
    assert events[-1] == Done("stop")


async def test_an_unknown_tool_is_reported_to_the_model_not_raised(loan_db):
    agent, ctx, _ = make_agent(loan_db, Lang.EN, [call("delete_loan")], say(PLAIN[Lang.EN]))
    events = await run(agent, ctx)
    assert events[0].result["error"] == "unknown_tool"
    assert text_of(events) == PLAIN[Lang.EN]
