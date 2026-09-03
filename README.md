# Breeze TTS API

An OpenAI-shaped, single-GPU HTTP server for Breeze TTS 2. This project adapts
the upstream inference code for API serving and uses the ConvRot INT8 hybrid
checkpoint from the ComfyUI ecosystem to reduce VRAM use.

It supports voice design, reference voice cloning/direction, streaming PCM or
SSE audio, persistent voice profiles, long-text splitting, CUDA-graph fast
paths, and explicit GPU model load/unload controls.

> [!IMPORTANT]
> The server code is Apache-2.0. Breeze model weights, derivative checkpoints,
> and generated outputs are governed by the BreezeBlue research/non-commercial
> license in [MODEL_LICENSE](MODEL_LICENSE). Obtain all necessary rights and
> consent before using reference audio or cloning a voice.

## Highlights

- `POST /v1/audio/speech` for voice design, cloning, and direction.
- Hybrid INT8 ConvRot backbone/text encoder with BF16 depth decoder.
- Automatic download of missing default model assets from
  [`drbaph/Breeze-TTS-2-comfyui`](https://huggingface.co/drbaph/Breeze-TTS-2-comfyui).
- 24 kHz mono PCM streaming or SSE audio chunks.
- Persistent profiles and cached reference audio codes.
- Explicit GPU model lifecycle: load and unload without restarting the API.
- Swagger UI at `http://HOST:7860/docs`.

## Requirements

- Linux, NVIDIA CUDA GPU, and a compatible NVIDIA driver.
- About 12 GB VRAM for practical eager serving. Fast CUDA-graph stages require
  more VRAM and add a cold-start warmup.
- Python 3.12 is used by the Docker image. A local install needs compatible
  CUDA PyTorch, compiler tooling, and Python dependencies.
- Internet access on the first default-model load, unless files are present.

## Model files and automatic download

All model assets are stored together:

```text
models/Breeze-TTS-2/
├── Breeze-TTS-2-int8-hybrid.safetensors
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── audio_tokenizer/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

When the default `Breeze-TTS-2-int8-hybrid.safetensors` is selected, missing
files are downloaded automatically. This includes the Qwen audio-tokenizer
codec needed to encode reference audio and decode generated speech. The
`qwen-tts` Python package is installed as a runtime dependency.

Automatic download does not apply to custom `--weights` or
`BREEZE_WEIGHTS_PATH` values; provide all custom-checkpoint assets yourself.

## Local deployment

Install a CUDA PyTorch build appropriate for your host, then install project
dependencies. The Dockerfile is the tested CUDA 12.8 configuration.

```bash
git clone https://github.com/remominor/breeze-tts-api.git
cd breeze-tts-api
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start the API. The first launch downloads missing default assets to
`models/Breeze-TTS-2/`.

```bash
python -m breeze_infer.api \
  models/Breeze-TTS-2 \
  --weights models/Breeze-TTS-2/Breeze-TTS-2-int8-hybrid.safetensors \
  --host 0.0.0.0 --port 7860 \
  --fast-backbone-decode --fast-depth-decoder --fast-codec
```

Useful URLs:

- `http://127.0.0.1:7860/docs` — interactive API documentation
- `http://127.0.0.1:7860/health` — readiness and model state
- `http://127.0.0.1:7860/metrics` — process/request metrics

If the initial load fails, such as after a CUDA OOM, the API stays running in
an unloaded `503` state. Free VRAM and retry `POST /v1/model/load`, or send a
GPU-using request; a failed retry returns HTTP 500.

## Docker and Compose deployment

Docker is recommended: the image pins and verifies its CUDA, PyTorch,
FlashAttention, and `comfy-kitchen` setup.

```bash
docker compose up -d --build
docker compose logs -f breeze-tts-api
```

Compose persists these host paths:

```text
./models  -> /models        # writable: first-run model download persists here
./voices  -> /data/profiles # voice profiles and cached reference codes
./cache   -> /cache         # Hugging Face, Triton, and TorchInductor caches
```

For Unraid or another managed host:

```bash
BREEZE_MODELS_HOST_PATH=/mnt/user/appdata/breeze-tts/models \
BREEZE_PROFILES_HOST_PATH=/mnt/user/appdata/breeze-tts/profiles \
BREEZE_CACHE_HOST_PATH=/mnt/user/appdata/breeze-tts/cache \
docker compose up -d --build
```

Common overrides:

```text
BREEZE_HOST_PORT=7860
BREEZE_MODEL_DIR=/models/Breeze-TTS-2
BREEZE_WEIGHTS_FILE=Breeze-TTS-2-int8-hybrid.safetensors
BREEZE_WEIGHTS_PATH=/models/custom.safetensors  # custom: no auto-download
BREEZE_PROFILE_DIR=/data/profiles
NVIDIA_VISIBLE_DEVICES=0
```

The model mount must be writable when automatic download is enabled.

## API examples

### Voice design

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome aboard. Your journey begins now.",
    "instructions": "A warm, calm narrator with a clear, thoughtful delivery.",
    "guidance_scale": 4,
    "seed": 42,
    "response_format": "wav"
  }' --output voice-design.wav
```

### Voice clone or direction

Reference audio requires its exact transcript. Add an instruction and CFG 4 to
direct the cloned delivery.

```bash
curl -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F 'ref_audio=@reference.wav' \
  -F 'ref_text=This is the exact transcript of the reference audio.' \
  -F 'text=We need to discuss what happened last night.' \
  -F 'instruction=Speak slowly with a restrained, serious tone.' \
  -F 'cfg_scale=4' -F 'seed=42' -F 'response_format=wav' \
  --output directed-clone.wav
```

Use `stream=true` for a streaming response. Raw streaming output is PCM; use
the `X-Sample-Rate: 24000` and `X-Sample-Format: s16le` headers. Set
`stream_format=sse` for server-sent audio chunk events.

### GPU model lifecycle

```bash
curl -X POST http://127.0.0.1:7860/v1/model/unload
curl -X POST http://127.0.0.1:7860/v1/model/load
```

Only GPU-using requests auto-load: speech synthesis and voice upload with
`preload=true`. Health, metrics, model metadata, and voice/profile CRUD remain
available while unloaded.

## Profiles and endpoint summary

`POST /upload_voice` or `POST /v1/upload_voice` saves a named profile. Its
default `preload=true` encodes reference audio immediately; set `preload=false`
to save it without loading the model.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness and model state |
| `GET /docs` | Swagger UI |
| `POST /v1/audio/speech` | Design, clone, or direct speech |
| `POST /v1/model/load` | Load the model onto GPU |
| `POST /v1/model/unload` | Release model and CUDA-graph memory |
| `GET /v1/audio/voices` | List built-in and saved voices |
| `POST /v1/upload_voice` | Create a reusable voice profile |

## Fast inference switches

The Docker entrypoint enables fast backbone decode, depth decoder, and codec
stages. Local deployments can opt in individually or use `--fast-all` when
there is enough VRAM:

```text
--fast-text-encoder
--fast-backbone-prefill
--fast-backbone-decode
--fast-depth-decoder
--fast-codec
--fast-all
```

Fast modes warm CUDA graphs before serving and are single-concurrency by
design. Use eager stages when debugging or conserving VRAM.

## Development

```bash
uv run --offline python -m pytest -q
uv run --offline python -m ruff check breeze_infer models tests infer.py
```

Do not commit checkpoints, voice profiles, or generated audio containing user
data.

## License and responsible use

Source code is [Apache License 2.0](LICENSE). Model materials, checkpoints,
derivatives, and self-hosted outputs are governed by [MODEL_LICENSE](MODEL_LICENSE),
including its research/non-commercial terms. You are responsible for legal
compliance and permission to use all reference voices and audio inputs.
