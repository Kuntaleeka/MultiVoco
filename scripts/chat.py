"""Talk to the agent in text, with a real LLM. Owner: workstream B (Garlic).

    uv run python -m scripts.chat --lang en
    uv run python -m scripts.chat --lang kn --provider gemini

Needs MULTIVOCO_GROQ_API_KEY or MULTIVOCO_GEMINI_API_KEY in .env, and a seeded database
(uv run python -m scripts.seed). Tool calls are printed as they happen. Type /quit to
stop, or /lang hi to switch language mid-call, as the orchestrator would.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid

from app.agent import LoanAgent, shutdown
from app.agent.llm import GEMINI, GROQ, build_llm
from app.core.interfaces import (
    CallContext,
    Handoff,
    ModelFirstToken,
    ProviderError,
    TextDelta,
    ToolResult,
)
from app.core.languages import Lang, route_for
from app.db.session import dispose_engine


async def reply(agent: LoanAgent, ctx: CallContext, user_text: str) -> bool:
    """Print one reply. Returns False once the call has been handed to a human."""
    started = time.perf_counter()
    spoken, first_token, first_text, handed_off = "", None, None, False
    try:
        async for event in agent.respond(user_text, ctx):
            elapsed = (time.perf_counter() - started) * 1000
            if isinstance(event, ModelFirstToken):
                first_token = elapsed
            elif isinstance(event, TextDelta):
                first_text = first_text or elapsed
                if not spoken:
                    print("agent> ", end="")
                print(event.text, end="", flush=True)
                spoken += event.text
            elif isinstance(event, ToolResult):
                args = json.dumps(event.args, ensure_ascii=False)
                result = json.dumps(event.result, ensure_ascii=False)
                print(f"  [tool {event.name}({args}) -> {result}  {event.duration_ms:.0f} ms]")
            elif isinstance(event, Handoff):
                handed_off = True
                print(f"\n  [handoff: {event.reason}]", end="")
    except ProviderError as error:
        print(f"\n  [provider error: {error}]")
    print()
    if first_text is not None:
        token = f"{first_token:.0f}" if first_token is not None else "-"
        print(f"  [model first token {token} ms, first sentence cleared {first_text:.0f} ms]")
    agent.commit_spoken(spoken, interrupted=False)
    return not handed_off


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=[lang.value for lang in Lang], default="en")
    parser.add_argument(
        "--provider", choices=[GROQ, GEMINI], help="default: the route for the language"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    ctx = CallContext(call_id=str(uuid.uuid4()), language=Lang(args.lang))
    # Straight to the real provider, whatever MULTIVOCO_MOCK says: this script is for
    # trying the agent against a model.
    agent = LoanAgent(ctx, llm_for=lambda lang: build_llm(args.provider or route_for(lang).llm))

    greeting = agent.greeting(ctx)
    print(f"agent> {greeting}")
    agent.commit_spoken(greeting, interrupted=False)
    try:
        while True:
            try:
                user_text = (await asyncio.to_thread(input, "you> ")).strip()
            except EOFError:
                break
            if user_text == "/quit":
                break
            if user_text.startswith("/lang "):
                ctx.language = Lang(user_text.split()[1])
                print(f"  [session language is now {ctx.language.value}]")
                continue
            if not await reply(agent, ctx, user_text):
                break
    finally:
        await shutdown()
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
