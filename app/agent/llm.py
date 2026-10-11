"""Streaming LLM clients for Groq and Gemini.

Both speak the OpenAI chat-completions protocol, so one client serves both. Failures
are raised as ProviderError or QuotaExceeded (app/core/interfaces.py), which cost the
caller one turn, not the call.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from functools import cache
from typing import Any

import httpx2 as httpx

from app.agent.config import AgentSettings, get_agent_settings
from app.core.config import get_settings
from app.core.interfaces import (
    LLM,
    Done,
    LLMEvent,
    Message,
    ProviderError,
    QuotaExceeded,
    TextDelta,
    ToolCall,
    ToolSpec,
)

log = logging.getLogger(__name__)

GROQ = "groq"
GEMINI = "gemini"

BASE_URLS = {
    GROQ: "https://api.groq.com/openai/v1",
    GEMINI: "https://generativelanguage.googleapis.com/v1beta/openai",
}

_RETRYABLE_STATUS = {408, 409, 500, 502, 503, 504}

# Stands in for the thought signature of a tool call Gemini did not make itself.
_NO_SIGNATURE = "context_engineering_is_the_way_to_go"
_MAX_SIGNATURES = 500


class _Retryable(Exception):
    """A failure worth one more try on the same provider."""


def to_wire_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["content"] = message.content or None
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.args, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return wire


def to_wire_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


@dataclass
class _PartialCall:
    id: str = ""
    name: str = ""
    arguments: str = ""
    signature: str = ""

    def finish(self) -> ToolCall:
        try:
            args = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        # Malformed arguments become {}: the tool then answers "invalid_arguments" and
        # the model gets a chance to call it properly.
        return ToolCall(
            id=self.id or f"call_{uuid.uuid4().hex[:12]}",
            name=self.name,
            args=args if isinstance(args, dict) else {},
        )


class OpenAICompatLLM:
    """One provider. Holds no per-call state, so one instance serves every call."""

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        *,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        settings: AgentSettings | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self.model = model
        self._url = f"{base_url or BASE_URLS[name]}/chat/completions"
        self._client = client
        self._settings = settings or get_agent_settings()
        self._signatures: dict[str, str] = {}  # Gemini only: tool call id -> signature

    def _wire_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire = [to_wire_message(message) for message in messages]
        if self.name == GEMINI:
            # Gemini rejects a tool call in the history that lacks its thought signature.
            # Calls it made itself get theirs back. Calls made by another provider, after
            # a fallback, get the placeholder Google documents for that case.
            for message in wire:
                for call in message.get("tool_calls", ()):
                    signature = self._signatures.get(call["id"], _NO_SIGNATURE)
                    call["extra_content"] = {"google": {"thought_signature": signature}}
        return wire

    def _remember_signature(self, call_id: str, signature: str) -> None:
        if len(self._signatures) >= _MAX_SIGNATURES:
            self._signatures.pop(next(iter(self._signatures)))
        self._signatures[call_id] = signature

    def _payload(self, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._wire_messages(messages),
            "stream": True,
            "temperature": self._settings.llm_temperature,
            "max_completion_tokens": self._settings.llm_max_tokens,
        }
        if self._settings.llm_reasoning_effort:
            payload["reasoning_effort"] = self._settings.llm_reasoning_effort
        if tools:
            payload["tools"] = [to_wire_tool(spec) for spec in tools]
            payload["tool_choice"] = "auto"
        return payload

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        payload = self._payload(messages, tools)
        tries = self._settings.llm_retries + 1
        for attempt in range(1, tries + 1):
            started = False
            try:
                async with aclosing(self._once(payload)) as events:
                    async for event in events:
                        started = True
                        yield event
                return
            except _Retryable as error:
                # Once an event has gone out, a retry would repeat it. Give up instead.
                if started or attempt == tries:
                    raise ProviderError(f"{self.name}: {error}") from error
                log.warning("%s failed (%s), retrying", self.name, error)
                await asyncio.sleep(self._settings.llm_retry_delay_s)

    async def _once(self, payload: dict[str, Any]) -> AsyncIterator[LLMEvent]:
        client = self._client or _shared_client()
        timeout = httpx.Timeout(
            self._settings.llm_read_timeout_s, connect=self._settings.llm_connect_timeout_s
        )
        headers = {"Authorization": f"Bearer {self._api_key}"}
        calls: dict[int, _PartialCall] = {}
        finish_reason = "stop"
        try:
            async with client.stream(
                "POST", self._url, json=payload, headers=headers, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise self._http_error(response.status_code, body)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise _Retryable(f"error in stream: {_brief(chunk['error'])}")
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if text := delta.get("content"):
                            yield TextDelta(text)
                        for position, part in enumerate(delta.get("tool_calls") or []):
                            call = calls.setdefault(part.get("index", position), _PartialCall())
                            function = part.get("function") or {}
                            call.id = part.get("id") or call.id
                            call.name = function.get("name") or call.name
                            call.arguments += function.get("arguments") or ""
                            google = (part.get("extra_content") or {}).get("google") or {}
                            call.signature = google.get("thought_signature") or call.signature
                        finish_reason = choice.get("finish_reason") or finish_reason
        except httpx.TimeoutException as error:
            raise _Retryable(f"timed out ({type(error).__name__})") from error
        except httpx.HTTPError as error:
            raise _Retryable(f"connection failed ({type(error).__name__})") from error

        for index in sorted(calls):
            if calls[index].name:
                call = calls[index].finish()
                if calls[index].signature:
                    self._remember_signature(call.id, calls[index].signature)
                yield call
        yield Done(finish_reason)

    def _http_error(self, status: int, body: str) -> Exception:
        detail = f"HTTP {status}: {_brief(body)}"
        if status == 429:
            return QuotaExceeded(f"{self.name}: {detail}")
        # Groq answers 400 "tool_use_failed" when the model wrote a tool call it could
        # not parse. A second try usually works.
        if status in _RETRYABLE_STATUS or (status == 400 and "tool_use_failed" in body):
            return _Retryable(detail)
        return ProviderError(f"{self.name}: {detail}")


class FallbackLLM:
    """Tries each provider in turn, moving on only when one fails before producing anything."""

    def __init__(self, *providers: LLM) -> None:
        self.providers = providers

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        for position, provider in enumerate(self.providers, start=1):
            started = False
            try:
                async with aclosing(provider.stream(messages, tools)) as events:
                    async for event in events:
                        started = True
                        yield event
                return
            except ProviderError as error:
                if started or position == len(self.providers):
                    raise
                log.warning(
                    "LLM %d of %d failed, trying the next: %s", position, len(self.providers), error
                )


def _brief(value: Any) -> str:
    return " ".join(str(value).split())[:300]


_client: httpx.AsyncClient | None = None


def _shared_client() -> httpx.AsyncClient:
    """One connection pool per process, so later turns skip the TLS handshake."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
    _client = None


def provider_clients(name: str) -> list[OpenAICompatLLM]:
    """One client per model of a provider, in order of preference. Empty without a key."""
    key = {GROQ: get_settings().groq_api_key, GEMINI: get_settings().gemini_api_key}[name]
    if not key:
        return []
    settings = get_agent_settings()
    models = {
        # Groq's rate limit is per model, so a second model is a second allowance.
        GROQ: [settings.groq_model, settings.groq_backup_model],
        GEMINI: [settings.gemini_model],
    }[name]
    return [OpenAICompatLLM(name, key, model) for model in dict.fromkeys(models) if model]


@cache
def build_llm(name: str) -> LLM:
    """The LLM registered as `name`, backed by the other provider if that one has a key."""
    primary = provider_clients(name)
    if not primary:
        raise ProviderError(f"{name}: no API key set (MULTIVOCO_{name.upper()}_API_KEY)")
    chain = [*primary, *provider_clients(GEMINI if name == GROQ else GROQ)]
    return FallbackLLM(*chain) if len(chain) > 1 else chain[0]
