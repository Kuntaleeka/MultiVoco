# Agent

Owner: Garlic (workstream B).

## Models

Checked live on 2026-10-11 and 2026-10-13, on the free tiers.

| Order | Provider | Model | What we saw |
|---|---|---|---|
| 1 | Groq | `openai/gpt-oss-120b` | 0.6 to 1.5 s to the first checked sentence, correct tool calls in all four languages |
| 2 | Gemini | `gemini-3.5-flash-lite` | Slow and patchy: about 1 s on a small request, 15 to 17 s on a real Bengali turn, timeouts on Kannada |

- Groq no longer serves Llama models.
- **Groq allows 8000 tokens a minute per model**, and one agent request costs about 900
  and grows with the call. One call at talking pace reaches the limit around its fifth
  turn. After that, turns go to Gemini or fail with `provider_error`.
- `openai/gpt-oss-20b` was tried as a second Groq model and is off. In a real call it
  wrote the caller's lines as well as its own. Set `MULTIVOCO_GROQ_BACKUP_MODEL` to
  try another.
- Kannada and Bengali stay on Groq. A comparison with Gemini was attempted on
  2026-10-13: every Kannada turn on Gemini timed out, and the Bengali turns took over
  15 s, so there was nothing usable to compare. In one Bengali turn Gemini also booked
  a callback the caller had not asked for.

## What the code enforces, whatever the model does

| Rule | Where |
|---|---|
| No account tool runs before `verify_identity` succeeds. Three failures lock the call and hand it to a human. | `app/tools/base.py`, `app/tools/identity.py` |
| `verify_identity` is not run with a phone number or date the caller never said. | `identity_details_not_given` in `app/agent/guardrails.py` |
| A sentence with a figure no tool returned in this call is not spoken. The reply is regenerated once, then replaced by a fixed safe line. | `FigureLedger` |
| A sentence in the wrong script for the session language is not spoken. | `wrong_language` |
| A reply ends at its first question, at four sentences, or at a line written for the caller ("User:"). | `LoanAgent._release` |
| Three turns in a row with nothing usable hand the call to a human. | `LoanAgent._miss` |
| "साथ" after two or more Hindi digit words is read as "सात". | `repair_misheard_digits` |

## Live check, 2026-10-13

One turn each, in text, on Groq. Script and raw output are not in the repo.

| Case | Result |
|---|---|
| English, digits and date said as words | Verified |
| Bengali, digits and date said as words | Verified |
| English, date written month first (`04/12/1990`) | Verified, read as 12 April |
| Hindi, digits as words with "सात" heard as "साथ" | Not verified: the model asked for the digits again. The repair above was added after this and has not been re-run live. |
| **Kannada, digits and date said as words** | **Failed, and cost the caller an attempt.** The phone digits were right. The year, said as "ಸಾವಿರದ ಒಂಬೈನೂರ ತೊಂಬತ್ತೊಂದು" (1991), was turned into 1989. |
| "Get me a real person" | Handed off, reason `caller_request` |
| "Roughly when will the loan be finished?" | Handed off, reason `out_of_scope`, no figure given |
| Hindi, "I do not agree with this charge" | Handed off, reason `dispute` |
| "When is my next EMI due?" | Answered from the tool, no handoff |
| Kannada and Bengali, EMI question | Answered from the tool, in the right script, one or two sentences |

Also seen: the model sometimes writes a date as the tool returned it (`2026-10-23`), or
an English month inside a Kannada sentence. The TTS layer turns an ISO date into words.

## Known limits

- **Dates and numbers said as words are converted by the model, and it gets Kannada
  years wrong.** Verification from spoken words is reliable only when speech
  recognition writes digits. The fix is to convert number words to digits in code,
  per language, which needs a native speaker for Kannada and Bengali.
- Numbers below a hundred written as words pass the figure check.
- The agent will not repeat a figure the caller said until a tool has returned it, so
  it cannot read back a promised amount before recording it.
- The Kannada and Bengali fixed lines, prompt examples, and number-word lists are
  Garlic's own wording and have not been checked by a native speaker.
