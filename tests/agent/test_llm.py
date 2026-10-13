"""The Groq and Gemini client against a fake HTTP transport. No network, no keys."""

import json

import httpx2 as httpx
import pytest

from app.agent import llm as llm_module
from app.agent.config import AgentSettings, get_agent_settings
from app.agent.llm import GEMINI, GROQ, FallbackLLM, OpenAICompatLLM, build_llm
from app.core import registry
from app.core.interfaces import (
    LLM,
    Done,
    Message,
    ProviderError,
    QuotaExceeded,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from app.core.languages import Lang

KEY = "test-key-not-real"
SPEC = ToolSpec("get_next_emi", "The next EMI.", {"type": "object", "properties": {}})
SETTINGS = AgentSettings(llm_retries=1, llm_retry_delay_s=0.0, _env_file=None)


def sse(*chunks: dict | str) -> httpx.Response:
    lines = [f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks]
    return httpx.Response(
        200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"}
    )


def delta(finish: str | None = None, **fields) -> dict:
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": finish}]}


def tool_part(index: int, arguments: str, call_id: str | None = None, name: str | None = None):
    part = {"index": index, "function": {"arguments": arguments}}
    if call_id:
        part["id"] = call_id
    if name:
        part["function"]["name"] = name
    return part


class Server:
    """Answers each request with the next scripted response, and records the requests."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def llm(self, name: str = GROQ) -> OpenAICompatLLM:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self))
        return OpenAICompatLLM(name, KEY, "test-model", client=client, settings=SETTINGS)


async def collect(llm, messages=None, tools=None) -> list:
    messages = messages or [Message("user", "hello")]
    return [event async for event in llm.stream(messages, tools or [])]


async def test_text_is_streamed_as_it_arrives():
    server = Server(
        sse(
            delta(role="assistant", content=""),
            delta(reasoning="thinking about it"),  # the model's thinking is not spoken
            delta(content="Your EMI"),
            delta(content=" is due."),
            delta(finish="stop"),
            "[DONE]",
        )
    )
    events = await collect(server.llm())
    assert events == [TextDelta("Your EMI"), TextDelta(" is due."), Done("stop")]
    assert isinstance(server.llm(), LLM)


async def test_tool_calls_are_put_together_from_their_pieces():
    server = Server(
        sse(
            delta(tool_calls=[tool_part(0, "", "call_a", "verify_identity")]),
            delta(tool_calls=[tool_part(0, '{"phone_last4": "48')]),
            delta(tool_calls=[tool_part(0, '21", "dob": "1990-04-12"}')]),
            delta(tool_calls=[tool_part(1, "{}", "call_b", "get_next_emi")]),
            delta(finish="tool_calls"),
            "[DONE]",
        )
    )
    assert await collect(server.llm()) == [
        ToolCall("call_a", "verify_identity", {"phone_last4": "4821", "dob": "1990-04-12"}),
        ToolCall("call_b", "get_next_emi", {}),
        Done("tool_calls"),
    ]


async def test_broken_tool_arguments_become_empty_not_an_error():
    server = Server(
        sse(delta(tool_calls=[tool_part(0, '{"loan_id": ', None, "get_next_emi")]), "[DONE]")
    )
    (call, done) = await collect(server.llm())
    assert (call.name, call.args) == ("get_next_emi", {})
    assert call.id  # one is made up when the provider sends none


async def test_the_request_carries_the_conversation_and_the_tools():
    server = Server(sse(delta(content="ok"), "[DONE]"))
    messages = [
        Message("system", "Be brief."),
        Message("user", "ನನ್ನ EMI?"),
        Message("assistant", "", tool_calls=(ToolCall("call_a", "get_next_emi", {"loan_id": 1}),)),
        Message("tool", '{"emi_amount": "8450.00"}', tool_call_id="call_a"),
    ]
    await collect(server.llm(GROQ), messages, [SPEC])

    request = server.requests[0]
    assert str(request.url) == f"{llm_module.BASE_URLS[GROQ]}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert (body["model"], body["stream"]) == ("test-model", True)
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "ನನ್ನ EMI?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "get_next_emi", "arguments": '{"loan_id": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": '{"emi_amount": "8450.00"}'},
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_next_emi",
                "description": "The next EMI.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


async def test_no_tools_key_is_sent_when_there_are_no_tools():
    server = Server(sse(delta(content="ok"), "[DONE]"))
    await collect(server.llm())
    assert "tools" not in json.loads(server.requests[0].content)


# --- Failures ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(503, text="overloaded"),
        httpx.Response(400, text='{"error": {"code": "tool_use_failed"}}'),
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("no route"),
    ],
)
async def test_a_passing_failure_is_retried_once(failure):
    server = Server(failure, sse(delta(content="ok"), "[DONE]"))
    assert await collect(server.llm()) == [TextDelta("ok"), Done("stop")]
    assert len(server.requests) == 2


async def test_a_failure_that_does_not_pass_becomes_a_provider_error():
    server = Server(httpx.Response(503, text="overloaded"), httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError, match="groq: timed out") as raised:
        await collect(server.llm())
    assert not isinstance(raised.value, QuotaExceeded)
    assert len(server.requests) == 2


async def test_a_rate_limit_is_quota_exceeded_and_is_not_retried():
    server = Server(httpx.Response(429, text="rate limit reached"))
    with pytest.raises(QuotaExceeded, match="429"):
        await collect(server.llm())
    assert len(server.requests) == 1


async def test_a_rejected_request_is_not_retried_and_does_not_leak_the_key():
    server = Server(httpx.Response(401, text="invalid api key"))
    with pytest.raises(ProviderError) as raised:
        await collect(server.llm())
    assert len(server.requests) == 1
    assert KEY not in str(raised.value)


async def test_a_failure_after_text_has_gone_out_is_not_retried():
    """A retry would make the caller hear the start of the reply twice."""
    server = Server(
        sse(delta(content="Your EMI"), {"error": {"message": "upstream died"}}),
        sse(delta(content="never used"), "[DONE]"),
    )
    events = []
    with pytest.raises(ProviderError, match="upstream died"):
        async for event in server.llm().stream([Message("user", "hello")], []):
            events.append(event)
    assert events == [TextDelta("Your EMI")]
    assert len(server.requests) == 1


# --- The backup provider -----------------------------------------------------------


async def test_the_backup_answers_when_the_primary_is_over_quota():
    primary = Server(httpx.Response(429, text="rate limit reached"))
    backup = Server(sse(delta(content="from the backup"), "[DONE]"))
    events = await collect(FallbackLLM(primary.llm(GROQ), backup.llm(GEMINI)))
    assert events == [TextDelta("from the backup"), Done("stop")]


async def test_the_backup_is_not_used_once_the_primary_has_spoken():
    primary = Server(sse(delta(content="Your EMI"), {"error": {"message": "upstream died"}}))
    backup = Server(sse(delta(content="from the backup"), "[DONE]"))
    with pytest.raises(ProviderError):
        await collect(FallbackLLM(primary.llm(GROQ), backup.llm(GEMINI)))
    assert backup.requests == []


async def test_both_failing_is_a_provider_error():
    primary = Server(httpx.Response(429, text="rate limit reached"))
    backup = Server(httpx.Response(401, text="bad key"))
    with pytest.raises(ProviderError, match="gemini"):
        await collect(FallbackLLM(primary.llm(GROQ), backup.llm(GEMINI)))


# --- Registration ------------------------------------------------------------------


@pytest.fixture
def keys(monkeypatch, settings_env):
    """Set the provider keys for one test: keys(GROQ="k") leaves Gemini without one."""

    def apply(**values: str) -> None:
        settings_env(
            MOCK="vad,stt,langid,tts,agent,trace_sink",
            GROQ_API_KEY=values.get("GROQ", ""),
            GEMINI_API_KEY=values.get("GEMINI", ""),
        )
        build_llm.cache_clear()

    yield apply
    build_llm.cache_clear()


def test_the_registry_routes_every_language_to_groq_with_gemini_behind_it(keys):
    keys(GROQ="g", GEMINI="m")
    for lang in Lang:
        llm = registry.get_llm(lang)
        assert isinstance(llm, FallbackLLM)
        assert [p.name for p in llm.providers] == [GROQ, GEMINI]
    assert registry.get_llm(Lang.EN) is registry.get_llm(Lang.KN)  # one shared instance


def test_one_key_means_only_that_provider(keys):
    keys(GROQ="g")
    assert registry.get_llm(Lang.EN).name == GROQ
    keys(GEMINI="m")
    assert build_llm(GEMINI).name == GEMINI


def test_a_second_groq_model_can_be_put_before_gemini(keys, monkeypatch):
    """Each Groq model has its own rate limit. Off by default: see note 20 in the plan."""
    monkeypatch.setenv("MULTIVOCO_GROQ_BACKUP_MODEL", "some/other-model")
    get_agent_settings.cache_clear()
    keys(GROQ="g", GEMINI="m")
    try:
        llm = registry.get_llm(Lang.EN)
        assert [(p.name, p.model) for p in llm.providers][1] == (GROQ, "some/other-model")
        assert [p.name for p in llm.providers] == [GROQ, GROQ, GEMINI]
    finally:
        monkeypatch.undo()
        get_agent_settings.cache_clear()


async def test_a_chain_moves_on_until_one_provider_answers():
    first = Server(httpx.Response(429, text="rate limit reached"))
    second = Server(httpx.Response(429, text="rate limit reached"))
    third = Server(sse(delta(content="third time lucky"), "[DONE]"))
    chain = FallbackLLM(first.llm(GROQ), second.llm(GROQ), third.llm(GEMINI))
    assert await collect(chain) == [TextDelta("third time lucky"), Done("stop")]


# --- Gemini's thought signatures ---------------------------------------------------


def signed(call_id: str, name: str, signature: str) -> dict:
    part = tool_part(0, "{}", call_id, name)
    part["extra_content"] = {"google": {"thought_signature": signature}}
    return part


async def test_gemini_gets_its_own_signature_back_and_a_placeholder_for_other_calls():
    server = Server(
        sse(delta(tool_calls=[signed("call_g", "get_next_emi", "sig-123")]), "[DONE]"),
        sse(delta(content="ok"), "[DONE]"),
    )
    gemini = server.llm(GEMINI)
    (own_call, _) = await collect(gemini)
    history = [
        Message("user", "hello"),
        # One call Gemini made, and one made by Groq before a fallback.
        Message("assistant", "", tool_calls=(own_call, ToolCall("call_x", "get_next_emi", {}))),
        Message("tool", "{}", tool_call_id="call_g"),
        Message("tool", "{}", tool_call_id="call_x"),
    ]
    await collect(gemini, history)

    sent = json.loads(server.requests[1].content)["messages"][1]["tool_calls"]
    assert [call["extra_content"]["google"]["thought_signature"] for call in sent] == [
        "sig-123",
        llm_module._NO_SIGNATURE,
    ]


async def test_groq_is_not_sent_gemini_fields():
    server = Server(sse(delta(content="ok"), "[DONE]"))
    history = [
        Message("assistant", "", tool_calls=(ToolCall("call_x", "get_next_emi", {}),)),
        Message("tool", "{}", tool_call_id="call_x"),
    ]
    await collect(server.llm(GROQ), history)
    sent = json.loads(server.requests[0].content)["messages"][0]["tool_calls"][0]
    assert "extra_content" not in sent


def test_a_missing_key_is_a_provider_error_not_a_crash_at_import(keys):
    keys()
    with pytest.raises(ProviderError, match="MULTIVOCO_GROQ_API_KEY"):
        registry.get_llm(Lang.EN)
