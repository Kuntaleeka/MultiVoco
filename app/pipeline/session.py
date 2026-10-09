"""One call: audio in, turn-taking, barge-in, language state, audio out.

Three things run at once for a session:

- The receive path (handle_audio, handle_message), driven by the socket. It runs the
  VAD on every frame and must stay fast, so anything slow is spawned as a task.
- At most one reply task, which streams agent text through TTS to the socket.
- Small helper tasks: the STT event consumer, barge-in, language detection, limits.

A reply is "busy" from the moment it is created until its turn is finished. Speech
that starts while a reply is busy is a possible interruption; it becomes a barge-in
once it has lasted `barge_in_min_ms`.
"""

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from app.core import protocol as p
from app.core import registry
from app.core.config import Settings, get_settings
from app.core.interfaces import (
    FRAME_BYTES,
    FRAME_MS,
    STT,
    TTS,
    AgentEvent,
    CallContext,
    Done,
    Handoff,
    LangGuess,
    ModelFirstToken,
    ProviderError,
    QuotaExceeded,
    TextDelta,
    ToolResult,
    VADEventKind,
)
from app.core.languages import AUTO, Lang
from app.core.trace import CallRecord, Mark, SessionClock, ToolCallTrace, TurnTrace
from app.pipeline.config import ASK_LANGUAGE, PipelineConfig, language_named_in, same_group
from app.pipeline.splitter import SentenceSplitter
from app.pipeline.turn import Segment, Turn

log = logging.getLogger(__name__)


class Transport(Protocol):
    async def receive(self) -> dict[str, Any]: ...
    async def send_text(self, data: str) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def close(self, code: int = 1000) -> None: ...


@dataclass
class _Utterance:
    """Caller speech between a VAD start and end."""

    confirmed: bool  # False while it may still be too short to count as an interruption
    buffered: bool  # True if no STT was open when it began: frames are kept, not streamed
    speech_frames: int = 0
    frames: list[bytes] = field(default_factory=list)
    stt_failed: bool = False


@dataclass
class _Pending:
    """A final transcript the STT owes us, and what to do with it."""

    speech_end_ms: float = 0.0
    lang_detected_ms: float | None = None
    discard: bool = False


async def _static(text: str) -> AsyncIterator[AgentEvent]:
    """A fixed line (greeting, language question) in the shape of an agent reply."""
    yield TextDelta(text + " ")
    yield Done()


class CallSession:
    def __init__(
        self,
        transport: Transport,
        config: PipelineConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.closed = asyncio.Event()
        self._transport = transport
        self._cfg = config or PipelineConfig()
        self._settings = settings or get_settings()
        self._clock = SessionClock()
        self._send_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

        self._started = False
        self._open = True
        self._closing = False
        self._outcome: str | None = None
        self._state: p.SessionState | None = None

        self._lang: Lang | None = None
        self._locked = False  # the caller chose the language: no automatic switching
        self._wanted: Lang | None = None  # a manual choice waiting for the turn to end
        self._asking = False  # the language question is out and unanswered
        self._mismatches = 0
        self._lang_lock = asyncio.Lock()

        self._stt: STT | None = None
        self._expected: deque[_Pending] = deque()
        self._preroll: deque[bytes] = deque(maxlen=self._cfg.preroll_frames)
        self._utt: _Utterance | None = None
        self._carry = ""  # a finished utterance waiting to be joined to the next one

        self._turn: Turn | None = None
        self._reply_task: asyncio.Task | None = None
        self._bargein_task: asyncio.Task | None = None
        self._position_waiter: tuple[int, asyncio.Future[float]] | None = None
        self._next_turn_id = 0

    # --- Socket ------------------------------------------------------------------

    async def run(self) -> None:
        """Read the socket until either side ends the call."""
        try:
            while not self._closing:
                message = await self._transport.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if (data := message.get("bytes")) is not None:
                    await self.handle_audio(data)
                elif (text := message.get("text")) is not None:
                    await self.handle_message(text)
        except WebSocketDisconnect:
            return
        except RuntimeError:
            if not self._closing:  # otherwise: receive() after our own close
                raise

    async def _send(self, message: BaseModel) -> None:
        await self._send_raw(text=message.model_dump_json())

    async def _send_raw(self, *, text: str | None = None, data: bytes | None = None) -> None:
        if not self._open:
            return
        async with self._send_lock:
            try:
                if text is not None:
                    await self._transport.send_text(text)
                else:
                    await self._transport.send_bytes(data or b"")
            except (WebSocketDisconnect, RuntimeError):
                self._open = False

    async def _set_state(self, state: p.SessionState) -> None:
        if state is not self._state:
            self._state = state
            await self._send(p.State(value=state))

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled() or task.exception() is None:
            return
        log.error("session %s: task failed", self.id, exc_info=task.exception())
        if not self._closing:
            self._spawn(self._fail(p.ErrorCode.INTERNAL, "Internal error.", p.CLOSE_INTERNAL))

    # --- Control messages --------------------------------------------------------

    async def handle_message(self, raw: str) -> None:
        try:
            message = p.parse_client_message(raw)
        except ValidationError:
            await self._bad_message("Unrecognised message.")
            return
        if isinstance(message, p.Start):
            if self._started:
                await self._bad_message("The call has already started.")
            else:
                await self._start(message.lang)
            return
        if not self._started:
            await self._bad_message("Send start first.")
        elif isinstance(message, p.SetLanguage):
            self._locked = True
            self._wanted = message.lang
            await self._apply_wanted_language()
        elif isinstance(message, p.PlaybackStarted):
            turn = self._turn
            if turn is not None and turn.id == message.turn_id and turn.playback_anchor is None:
                turn.playback_anchor = asyncio.get_running_loop().time()
                turn.trace.client_playback_started_ms = message.t_client_ms
        elif isinstance(message, p.PlaybackPosition):
            waiter = self._position_waiter
            if waiter is not None and waiter[0] == message.turn_id and not waiter[1].done():
                waiter[1].set_result(message.ms_played)
        elif isinstance(message, p.End):
            await self._close(p.CLOSE_NORMAL, "completed")

    async def _start(self, lang: Lang | str) -> None:
        self._started = True
        loop = asyncio.get_running_loop()
        self._started_at = self._last_activity = loop.time()
        auto = lang == AUTO
        self._call = CallRecord(self.id, datetime.now(UTC), requested_lang=str(lang))
        self._ctx = CallContext(call_id=self.id, language=Lang.EN if auto else Lang(lang))
        # True until the language is known. The agent's greeting must be language-neutral
        # while it is set.
        self._ctx.extra["language_pending"] = auto
        self._vad = registry.get_vad()
        self._detector = registry.get_language_detector()
        self._sink = registry.get_trace_sink()
        self._agent = registry.get_agent(self._ctx)
        await self._sink_call(self._sink.call_started(self._call))
        await self._send(p.Ready(session_id=self.id))
        if not auto:
            self._locked = True
            await self._set_language(Lang(lang), p.LanguageSource.MANUAL, None)
        self._spawn(self._watch_limits())
        self._begin_reply(_static(self._agent.greeting(self._ctx)))

    # --- Audio in ----------------------------------------------------------------

    async def handle_audio(self, frame: bytes) -> None:
        if not self._started:
            await self._bad_message("Send start before audio.")
            return
        if self._closing:
            return
        if len(frame) != FRAME_BYTES:
            await self._bad_message(f"Audio frames must be {FRAME_BYTES} bytes.")
            return
        event = await self._vad.process(frame)
        kind = event.kind if event else None
        if kind is VADEventKind.SPEECH_START and self._utt is None:
            self._utt = _Utterance(confirmed=not self._busy(), buffered=self._stt is None)
            for earlier in self._preroll:
                await self._feed(self._utt, earlier, speech=False)
            self._preroll.clear()
        if self._utt is None:
            self._preroll.append(frame)
            return
        await self._feed(self._utt, frame, speech=self._vad.is_speech)
        if kind is VADEventKind.SPEECH_END:
            await self._end_utterance()

    async def _feed(self, utt: _Utterance, frame: bytes, *, speech: bool) -> None:
        self._last_activity = asyncio.get_running_loop().time()
        if utt.buffered:
            utt.frames.append(frame)
        elif self._stt is not None and not utt.stt_failed:
            try:
                await self._stt.send_audio(frame)
            except ProviderError as error:
                utt.stt_failed = True
                await self._report(error)
        if not speech:
            return
        utt.speech_frames += 1
        if utt.confirmed:
            return
        if not self._busy():
            utt.confirmed = True  # the reply ended while they were starting to speak
        elif utt.speech_frames * FRAME_MS >= self._cfg.barge_in_min_ms:
            utt.confirmed = True
            if self._turn is not None:
                self._bargein_task = self._spawn(self._barge_in(self._turn))

    async def _end_utterance(self) -> None:
        utt, self._utt = self._utt, None
        assert utt is not None
        if not utt.confirmed:
            # Too short to be an interruption. The STT still owes a final for it.
            if not utt.buffered and self._stt is not None:
                self._expected.append(_Pending(discard=True))
                await self._finish_stt()
            return
        speech_end_ms = self._clock.now_ms()
        if utt.buffered:
            self._spawn(self._handle_buffered(utt, speech_end_ms))
        elif self._stt is not None:
            self._expected.append(_Pending(speech_end_ms))
            await self._finish_stt()

    async def _finish_stt(self) -> None:
        assert self._stt is not None
        try:
            await self._stt.finish()
        except ProviderError as error:
            self._expected.pop()
            await self._report(error)

    async def _consume_stt(self, stt: STT) -> None:
        try:
            async for transcript in stt.events():
                if not transcript.final:
                    if self._utt is not None and self._utt.confirmed:
                        text = f"{self._carry} {transcript.text}".strip()
                        await self._send_user_transcript(text, final=False)
                    continue
                pending = self._expected.popleft() if self._expected else None
                if pending is None or pending.discard:
                    continue
                await self._on_user_final(transcript.text, pending)
        except ProviderError as error:
            await self._report(error)

    async def _send_user_transcript(self, text: str, *, final: bool) -> None:
        await self._send(
            p.TranscriptMessage(role="user", text=text, final=final, turn_id=self._next_turn_id)
        )

    async def _on_user_final(self, text: str, pending: _Pending) -> None:
        text = f"{self._carry} {text}".strip()
        if not text:
            return
        if self._utt is not None and self._utt.confirmed:
            # They paused and went on. Answer once, when they have finished.
            self._carry = text
            return
        self._carry = ""
        await self._stop_reply()

        switched = False
        if not self._locked and self._lang is not None:
            guess = self._detector.detect_text(text)
            if guess.lang is not None:
                if same_group(self._lang, guess.lang):
                    self._mismatches = 0
                else:
                    self._mismatches += 1
                    if self._mismatches >= self._cfg.switch_after_mismatches:
                        switched = await self._set_language(
                            guess.lang, p.LanguageSource.SCRIPT, guess.confidence
                        )

        await self._send_user_transcript(text, final=True)
        self._begin_reply(
            self._agent.respond(text, self._ctx),
            user_text=text,
            pending=pending,
            switched=switched,
        )

    # --- Language ----------------------------------------------------------------

    async def _set_language(
        self, lang: Lang, source: p.LanguageSource, confidence: float | None
    ) -> bool:
        if self._stt is None or not same_group(self._lang, lang):
            try:
                stt = registry.get_stt(lang)
                await stt.start(lang)
            except (ProviderError, LookupError) as error:
                log.warning("session %s: no STT for %s: %s", self.id, lang, error)
                await self._send(
                    p.Error(
                        code=p.ErrorCode.LANGUAGE_UNAVAILABLE,
                        message=f"{lang.value} is not available right now.",
                    )
                )
                return False
            await self._drop_stt()
            self._stt = stt
            self._spawn(self._consume_stt(stt))
        self._lang = lang
        self._ctx.language = lang
        self._ctx.extra["language_pending"] = False
        self._call.final_lang = lang
        self._mismatches = 0
        self._asking = False
        await self._send(p.LanguageMessage(lang=lang, source=source, confidence=confidence))
        return True

    async def _drop_stt(self) -> None:
        stt, self._stt = self._stt, None
        self._expected.clear()
        if stt is not None:
            try:
                await stt.close()  # ends its events(), and with it the consumer task
            except ProviderError:
                pass

    async def _apply_wanted_language(self) -> None:
        """Apply a manual choice between turns, never during one."""
        if self._wanted is None or self._busy() or self._utt is not None:
            return
        lang, self._wanted = self._wanted, None
        await self._set_language(lang, p.LanguageSource.MANUAL, None)

    async def _handle_buffered(self, utt: _Utterance, speech_end_ms: float) -> None:
        """An utterance that arrived before any STT was open: decide the language first."""
        async with self._lang_lock:
            detected_ms = None
            if self._lang is None:
                if self._asking:
                    lang = await self._language_from_answer(utt)
                    if await self._set_language(lang, p.LanguageSource.ASKED, None):
                        await self._stop_reply()
                        self._begin_reply(_static(self._agent.greeting(self._ctx)))
                    return
                guess = await self._detect(utt)
                detected_ms = self._clock.now_ms()
                unsure = (
                    guess.lang is None
                    or guess.confidence < self._cfg.min_lang_confidence
                    or utt.speech_frames * FRAME_MS < self._cfg.min_detect_ms
                )
                if unsure or guess.lang is None:
                    self._asking = True
                    await self._stop_reply()
                    self._begin_reply(_static(ASK_LANGUAGE))
                    return
                if not await self._set_language(
                    guess.lang, p.LanguageSource.AUDIO, guess.confidence
                ):
                    return
            if self._stt is None:
                return
            pending = _Pending(speech_end_ms, detected_ms)
            try:
                for frame in utt.frames:
                    await self._stt.send_audio(frame)
                self._expected.append(pending)
                await self._stt.finish()
            except ProviderError as error:
                if pending in self._expected:
                    self._expected.remove(pending)
                await self._report(error)

    async def _detect(self, utt: _Utterance) -> LangGuess:
        try:
            return await self._detector.detect_audio(b"".join(utt.frames))
        except ProviderError as error:
            log.warning("session %s: language detection failed: %s", self.id, error)
            return LangGuess(None, 0.0)

    async def _language_from_answer(self, utt: _Utterance) -> Lang:
        """The caller was asked which language they want. Work out what they said."""
        text = ""
        try:
            stt = registry.get_stt(Lang.EN)
            await stt.start(Lang.EN)
            try:
                for frame in utt.frames:
                    await stt.send_audio(frame)
                await stt.finish()
                async for transcript in stt.events():
                    if transcript.final:
                        text = transcript.text
                        break
            finally:
                await stt.close()
        except (ProviderError, LookupError) as error:
            log.warning("session %s: could not transcribe the language answer: %s", self.id, error)
        return language_named_in(text) or (await self._detect(utt)).lang or Lang.EN

    # --- Replies -----------------------------------------------------------------

    def _busy(self) -> bool:
        return self._turn is not None and not self._turn.finished

    def _begin_reply(
        self,
        events: AsyncIterator[AgentEvent],
        *,
        user_text: str = "",
        pending: _Pending | None = None,
        switched: bool = False,
    ) -> None:
        turn_id = self._next_turn_id
        self._next_turn_id += 1
        lang = self._lang or Lang.EN
        trace = TurnTrace(
            call_id=self.id,
            idx=turn_id,
            language=lang,
            user_text=user_text,
            language_switched=switched,
            region=self._settings.region,
            providers={kind: registry.provider_name(kind, lang) for kind in registry.ROUTED_KINDS},
        )
        if pending is not None:
            trace.mark(Mark.SPEECH_END, pending.speech_end_ms)
            if pending.lang_detected_ms is not None:
                trace.mark(Mark.LANG_DETECTED, pending.lang_detected_ms)
            trace.mark(Mark.STT_FINAL, self._clock.now_ms())
        self._turn = Turn(id=turn_id, lang=lang, trace=trace)
        self._reply_task = self._spawn(self._run_reply(self._turn, events))

    async def _run_reply(self, turn: Turn, events: AsyncIterator[AgentEvent]) -> None:
        if turn.id > self._settings.max_turns:
            turn.finished = True
            await self._limit(f"Call limit of {self._settings.max_turns} turns reached.")
            return
        await self._set_state(p.SessionState.THINKING)
        sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self._cfg.sentence_queue)
        producer = asyncio.create_task(self._produce(turn, events, sentences))
        try:
            tts = registry.get_tts(turn.lang)
            turn.sample_rate = tts.sample_rate(turn.lang)
            while (sentence := await sentences.get()) is not None:
                await self._speak(turn, tts, sentence)
            await producer
            if turn.audio_started:
                await self._send(p.AudioEnd(turn_id=turn.id))
                await self._wait_for_playback(turn)
            await self._finish_turn(turn, turn.spoken_text(), interrupted=False, ms_played=None)
            if turn.handoff is not None:
                self._call.handoff_reason = turn.handoff
                await self._set_state(p.SessionState.HANDOFF)
                await self._close(p.CLOSE_NORMAL, "handoff")
        except (ProviderError, LookupError) as error:
            await self._report(error)
            if turn.audio_started:
                await self._send(p.AudioEnd(turn_id=turn.id))
            await self._finish_turn(
                turn, turn.spoken_text(complete_only=True), interrupted=True, ms_played=None
            )
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _produce(
        self, turn: Turn, events: AsyncIterator[AgentEvent], out: asyncio.Queue[str | None]
    ) -> None:
        """Agent events in, sentences out. Always ends the queue unless cancelled."""
        splitter = SentenceSplitter()
        trace = turn.trace

        async def emit(sentence: str) -> None:
            trace.mark(Mark.LLM_FIRST_SENTENCE, self._clock.now_ms())
            await out.put(sentence)

        try:
            async for event in events:
                if isinstance(event, TextDelta):
                    trace.mark(Mark.LLM_FIRST_TOKEN, self._clock.now_ms())
                    for sentence in splitter.feed(event.text):
                        await emit(sentence)
                elif isinstance(event, ModelFirstToken):
                    trace.mark(Mark.LLM_RAW_FIRST_TOKEN, self._clock.now_ms())
                elif isinstance(event, ToolResult):
                    trace.tool_calls.append(
                        ToolCallTrace(event.name, event.args, event.result, event.duration_ms)
                    )
                elif isinstance(event, Handoff):
                    turn.handoff = event.reason
                elif isinstance(event, Done):
                    break
            if (rest := splitter.flush()) is not None:
                await emit(rest)
        except asyncio.CancelledError:
            raise
        except BaseException:
            await out.put(None)
            raise
        await out.put(None)

    async def _speak(self, turn: Turn, tts: TTS, sentence: str) -> None:
        if not turn.audio_started:
            turn.audio_started = True
            await self._send(p.AudioStart(turn_id=turn.id, sample_rate=turn.sample_rate))
            await self._set_state(p.SessionState.SPEAKING)
        segment = Segment(sentence)
        turn.segments.append(segment)
        await self._send(
            p.TranscriptMessage(role="agent", text=turn.spoken_text(), final=False, turn_id=turn.id)
        )
        bytes_per_ms = turn.sample_rate * 2 / 1000
        async for chunk in tts.synthesize(sentence, turn.lang):
            turn.trace.mark(Mark.TTS_FIRST_CHUNK, self._clock.now_ms())
            await self._send_raw(data=chunk)
            turn.trace.mark(Mark.AUDIO_SENT, self._clock.now_ms())
            if turn.first_audio_at is None:
                turn.first_audio_at = asyncio.get_running_loop().time()
            segment.ms += len(chunk) / bytes_per_ms
        segment.complete = True

    async def _wait_for_playback(self, turn: Turn) -> None:
        """Audio is sent faster than it plays. Stay interruptible until it has played."""
        loop = asyncio.get_running_loop()
        while True:
            anchor = turn.playback_anchor or turn.first_audio_at or loop.time()
            remaining = anchor + turn.audio_ms / 1000 + self._cfg.playback_margin_s - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    async def _finish_turn(
        self, turn: Turn, text: str, *, interrupted: bool, ms_played: float | None
    ) -> None:
        if turn.finished:
            return
        turn.finished = True
        self._last_activity = asyncio.get_running_loop().time()
        # Everything the caller heard goes into the history, whoever wrote it: agent
        # replies, the greeting, and the language question. An empty string means the
        # reply was cut off before any of it was heard.
        self._agent.commit_spoken(text, interrupted)
        turn.trace.agent_text = text
        turn.trace.interrupted = interrupted
        turn.trace.ms_played = ms_played
        if turn.audio_started:
            await self._send(
                p.TranscriptMessage(role="agent", text=text, final=True, turn_id=turn.id)
            )
        await self._sink_call(self._sink.turn_finished(turn.trace))
        await self._apply_wanted_language()
        if not self._closing:
            await self._set_state(p.SessionState.LISTENING)

    async def _barge_in(self, turn: Turn) -> None:
        if turn.finished:
            return
        task = self._reply_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if turn.finished:
            return
        ms_played: float | None = None
        if turn.audio_started:
            future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
            self._position_waiter = (turn.id, future)
            await self._send(p.Flush(turn_id=turn.id))
            try:
                reported = await asyncio.wait_for(future, self._cfg.position_timeout_s)
            except TimeoutError:
                reported = 0.0
            finally:
                self._position_waiter = None
            ms_played = max(0.0, min(reported, turn.audio_ms))
        await self._finish_turn(
            turn, turn.heard_text(ms_played or 0.0), interrupted=True, ms_played=ms_played
        )

    async def _stop_reply(self) -> None:
        """Make sure no reply is running before another one starts."""
        if self._bargein_task is not None and not self._bargein_task.done():
            await asyncio.gather(self._bargein_task, return_exceptions=True)
        if self._turn is not None and not self._turn.finished:
            await self._barge_in(self._turn)

    # --- Errors, limits, and the end of the call ---------------------------------

    async def _report(self, error: Exception) -> None:
        log.warning("session %s: %s: %s", self.id, type(error).__name__, error)
        if isinstance(error, QuotaExceeded):
            code, message = p.ErrorCode.LANGUAGE_UNAVAILABLE, "This language is over its quota."
        else:
            code, message = p.ErrorCode.PROVIDER_ERROR, "A speech service failed. Please repeat."
        await self._send(p.Error(code=code, message=message))

    async def _sink_call(self, awaitable: Coroutine[Any, Any, None]) -> None:
        try:
            await awaitable
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("session %s: trace sink failed", self.id)

    async def _watch_limits(self) -> None:
        loop = asyncio.get_running_loop()
        settings = self._settings
        while not self._closing:
            await asyncio.sleep(0.25)
            now = loop.time()
            if now - self._started_at > settings.max_call_seconds:
                await self._limit(f"Call limit of {settings.max_call_seconds} seconds reached.")
            elif (
                not self._busy()
                and self._utt is None
                and now - self._last_activity > settings.idle_timeout_seconds
            ):
                await self._limit("The call was idle for too long.")

    async def _limit(self, message: str) -> None:
        await self._fail(p.ErrorCode.LIMIT_REACHED, message, p.CLOSE_LIMIT_REACHED, "limit")

    async def _bad_message(self, message: str) -> None:
        await self._fail(p.ErrorCode.BAD_MESSAGE, message, p.CLOSE_BAD_MESSAGE)

    async def _fail(
        self, code: p.ErrorCode, message: str, close_code: int, outcome: str = "error"
    ) -> None:
        if not self._closing:
            await self._send(p.Error(code=code, message=message))
            await self._close(close_code, outcome)

    async def _close(self, code: int, outcome: str) -> None:
        if self._closing:
            return
        self._closing = True
        self._outcome = outcome
        async with self._send_lock:
            try:
                await self._transport.close(code)
            except (WebSocketDisconnect, RuntimeError):
                pass
        self._open = False
        self.closed.set()

    async def teardown(self) -> None:
        """Stop every task and record the end of the call. Safe to call once, always."""
        self._closing = True
        self._open = False
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current and not task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._drop_stt()
        self.closed.set()
        if not self._started:
            return
        turn = self._turn
        if turn is not None and not turn.finished:
            turn.finished = True
            turn.trace.agent_text = turn.spoken_text(complete_only=True)
            turn.trace.interrupted = True
            await self._sink_call(self._sink.turn_finished(turn.trace))
        self._call.ended_at = datetime.now(UTC)
        self._call.outcome = self._outcome or "abandoned"
        self._call.verified_customer_id = self._ctx.verified_customer_id
        await self._sink_call(self._sink.call_ended(self._call))
