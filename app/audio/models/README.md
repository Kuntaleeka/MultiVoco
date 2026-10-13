# Models

Owner: Onion.

| File | What | Source | Licence |
|---|---|---|---|
| `silero_vad.onnx` | Silero VAD v6.2.3, 16 kHz and 8 kHz, about 2.3 MB | https://github.com/snakers4/silero-vad (`src/silero_vad/data/silero_vad.onnx`) | MIT, Copyright (c) 2020-present Silero Team |

SHA-256 of `silero_vad.onnx`:
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`

The file is committed so that tests and the Docker build need no download. To update
it, replace the file, update the version and hash above, and run
`uv run pytest tests/audio`.
