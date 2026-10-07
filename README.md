# MultiVoco

A voice AI loan servicing agent that speaks English, Hindi/Hinglish, Kannada, and
Bengali. Work in progress: see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for
the plan and current status.

## Setup

Needs [uv](https://docs.astral.sh/uv/). No API keys are needed to run the tests or
the server: with no `.env`, every component is a mock.

```sh
uv sync
uv run pytest
uv run uvicorn app.main:app --reload
```

To use real providers, copy `.env.example` to `.env`, fill in the keys, and set
`MULTIVOCO_MOCK` to the component kinds that should stay mocked.

## Where things are

| Path | What |
|---|---|
| `app/core/` | Contracts shared by every workstream: interfaces, mocks, registry, languages, trace, protocol models. Frozen. |
| `docs/protocol.md` | The WebSocket protocol between the browser and the server |
| `tests/core/test_mock_e2e.py` | One turn through every contract on mocks. Must always pass. |
