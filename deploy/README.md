# Deploying to Render

Owner: Onion. Render builds the `Dockerfile` at the repo root and runs it as a free web
service. With no keys set, it runs the mock pipeline: the full call flow with canned
replies and a tone for a voice.

Hugging Face Spaces was the original plan. Since mid-2026 a Docker Space needs a paid
PRO plan, so it is no longer a free option.

## What the free plan gives

Checked on 2026-10-08. Confirm on Render's pricing page before relying on it.

| | Free web service |
|---|---|
| Card needed | No |
| Memory | 512 MB |
| CPU | Shared |
| Hours | 750 a month, enough for one always-on service |
| Sleep | After 15 minutes with no traffic. The next visit waits about a minute. |
| WebSockets | Supported |

## First deploy

1. Push the branch you want to deploy to GitHub.
2. In the Render dashboard, choose **New > Blueprint**, connect the GitHub repository,
   and pick that branch. Render reads `render.yaml` and proposes one service,
   `multivoco`, on the free plan in Singapore.
3. Apply it and wait for the first build. This is also the first time the Dockerfile
   is built anywhere, so read the build log if it fails.
4. Check, with the service's `onrender.com` URL:
   - `/healthz` returns `{"status": "ok", ...}`
   - the page at `/` loads, and "Start call" plays a greeting tone and shows a transcript
   - a second browser tab can call at the same time, and a third is told the line is busy

After that, every push to the branch redeploys.

## Checking WebSockets through the proxy

`/ws/echo` sends back whatever it receives. If calls fail on Render but work locally,
test this first:

```js
// In the browser console, on the deployed page
const ws = new WebSocket(`wss://${location.host}/ws/echo`);
ws.onmessage = (e) => console.log("echo:", e.data);
ws.onopen = () => ws.send("hello");
```

## Settings

All settings are environment variables: see `.env.example`. Non-secret ones are in
`render.yaml`. Add provider keys in the dashboard under **Environment**, never in
`render.yaml`. `MULTIVOCO_MOCK` controls which components are mocked. Leave it at `all`
until real providers are being switched on.

## Limits to plan around

- **512 MB of memory.** The mock pipeline is far below it. Silero and Piper will have
  to fit too: use ONNX Runtime, not PyTorch, and load one voice per language.
  Measure memory when each one is added.
- **Sleep after 15 minutes.** The first visitor after a quiet spell waits about a
  minute. The page needs a "waking up" state, and the demo needs a recorded fallback.
- **One instance.** Sessions live in memory, which is fine for a single instance.

## Running the image locally

```sh
docker build -t multivoco .
docker run --rm -p 8000:8000 multivoco
```

The page needs a secure context for the microphone. `http://localhost:8000` counts as
one; any other plain-http address does not.
