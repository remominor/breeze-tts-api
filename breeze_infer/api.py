"""OpenAI/OmniVoice-shaped HTTP API for Breeze TTS 2."""
from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import logging
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from breeze_infer.audio import encode_prompt_audio
from breeze_infer.profiles import ProfileExistsError, ProfileNotFoundError, ProfileStore
from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from breeze_infer.text_chunks import estimate_speech_frames, split_text_to_fit
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
DEFAULT_INSTRUCTION = "Speak clearly and naturally."
CONTEXT_SAFETY_FRAMES = 64
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiSettings:
    model: Path
    weights: Path | None = None
    voice_dir: Path = Path.home() / ".local/share/breeze-tts/voices"
    max_ref_audio_bytes: int = 25 * 1024 * 1024
    cors_origins: tuple[str, ...] = ("http://localhost:7860", "http://127.0.0.1:7860")
    fast_all: bool | None = None
    fast_text_encoder: bool = False
    fast_backbone_prefill: bool = False
    fast_backbone_decode: bool = False
    fast_depth_decoder: bool = False
    fast_codec: bool = False


_settings: ApiSettings | None = None
_request_lock = threading.Lock()


def _quiet_expected_torch_compile_warnings() -> None:
    """Hide expected dynamic-shape/compiler chatter from the eager CFG=1 path."""
    # These warnings are emitted while torch.compile specializes the codec and
    # first eager backbone request.  They do not indicate a synthesis error;
    # keep the rest of torch and API logging at their normal levels.
    logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(
        logging.ERROR
    )
    logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
    try:
        import torch

        torch._logging.set_logs(dynamic=logging.ERROR, inductor=logging.ERROR)
    except (AttributeError, ImportError):
        # Older torch releases do not expose torch._logging.  The logger-level
        # filters above still cover the known messages.
        pass


def _pcm16(audio: np.ndarray) -> bytes:
    values = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (values * 32767.0).astype("<i2", copy=False).tobytes()


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    import struct
    return b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm)) + pcm


def _prompt_token_count(inputs: dict) -> int:
    prompt_keys = (
        "input_ids",
        "cfg_negative_prompt_ids",
        "cfg_uncond_prompt_ids",
        "cfg_ref_prompt_ids",
        "cfg_ins_prompt_ids",
    )
    lengths = [
        int(inputs[key].shape[1])
        for key in prompt_keys
        if inputs.get(key) is not None
    ]
    return max(lengths, default=0)


def _parse_seed(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, "seed must be an integer") from exc


def _parse_stream(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise HTTPException(422, "stream must be a boolean")


def _download_voice_url(url: str, limit: int) -> bytes:
    with urlopen(url, timeout=30) as response:
        return response.read(limit + 1)


def _encode_reference_bytes(audio_tokenizer, data: bytes):
    with tempfile.NamedTemporaryFile(prefix="breeze-upload-", suffix=".wav") as handle:
        handle.write(data)
        handle.flush()
        return encode_prompt_audio(audio_tokenizer, Path(handle.name))


async def _upload_bytes(upload: UploadFile | None, limit: int) -> bytes | None:
    if upload is None or not upload.filename:
        return None
    data = await upload.read(limit + 1)
    if not data:
        raise HTTPException(422, "Reference audio is empty")
    if len(data) > limit:
        raise HTTPException(413, "Reference audio exceeds the configured size limit")
    return data


def _profile_request(store: ProfileStore, voice: str | None, *, ref_text: str | None, instruction: str | None) -> tuple[dict, str]:
    voice = (voice or "voice-design").strip()
    try:
        profile_id = store.resolve(voice.removeprefix("clone:"))
    except (ProfileNotFoundError, ValueError):
        profile_id = None
    if profile_id:
        profile = store.get(profile_id)
        effective_ref_text = ref_text.strip() if ref_text and ref_text.strip() else profile["ref_text"]
        request = {"ref_text": effective_ref_text, "speaker": "S0"}
        codes = store.load_codes(profile_id)
        if codes is not None:
            request["ref_audio_codes"] = codes
        else:
            request["ref_audio_path"] = str(store.audio_path(profile_id))
        request["profile_id"] = profile_id
        return request, "ref_edit_tata" if instruction else "ref_clone_tata"
    if voice not in {"default", "voice-design"}:
        raise HTTPException(404, f"Voice '{voice}' not found")
    return {"speaker": "S0"}, "tts_instruction" if instruction else "tts_plain"


def _normalise_instruction_for_cfg(
    instruction: str,
    cfg_value: object,
    *,
    fast_enabled: bool,
) -> tuple[str, float]:
    """Return a valid instruction/CFG pairing for the Breeze templates.

    CFG needs both conditional and negative prompt branches.  The plain and
    clone templates deliberately have no negative branch, so an empty design
    instruction together with CFG > 1 used to reach ``prepare_inputs`` as an
    invalid combination.  Treat a blank instruction as a request for the
    neutral instruction in that case.  The low-VRAM fast profile only captures
    its two-branch CFG path, so CFG=1 is promoted to its safe CFG=4 default
    while that profile is active instead of falling back to eager inference.
    """
    try:
        scale = (
            float(cfg_value)
            if cfg_value is not None and str(cfg_value).strip() != ""
            else (4.0 if instruction else 1.0)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            422, "guidance_scale/cfg_scale must be a number"
        ) from exc
    if not np.isfinite(scale) or scale <= 0:
        raise HTTPException(
            422, "guidance_scale/cfg_scale must be finite and greater than 0"
        )
    if fast_enabled and scale == 1.0:
        return instruction or DEFAULT_INSTRUCTION, 4.0
    if not instruction and scale != 1.0:
        return DEFAULT_INSTRUCTION, scale
    return instruction, scale


def _load_app(app: FastAPI, settings: ApiSettings) -> None:
    _quiet_expected_torch_compile_warnings()
    tokenizer, model, audio_tokenizer = load_runtime(settings.model, device=resolve_device(), attn_implementation="eager", weights_path=settings.weights)
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, FastStreamingConfig(max_new_tokens=MAX_NEW_TOKENS, max_seq_len=MAX_SEQ_LEN, fast_all=settings.fast_all, fast_text_encoder=settings.fast_text_encoder, fast_backbone_prefill=settings.fast_backbone_prefill, fast_backbone_decode=settings.fast_backbone_decode, fast_depth_decoder=settings.fast_depth_decoder, fast_codec=settings.fast_codec, repetition_penalty=REPETITION_PENALTY), tokenizer=tokenizer)
    if runtime.fast_enabled:
        profile = replace(load_warmup_profile(FAST_CONFIG), codec_chunk_frames=runtime.codec_chunk_frames)
        runtime.warmup_from_profile(profile)
    eager_runtime = None
    if runtime.fast_enabled:
        eager_runtime = FastBreezeStreamingRuntime(
            model,
            audio_tokenizer,
            FastStreamingConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN,
                # Leave the master switch unset so only fast_codec below is
                # enabled; ``fast_all=False`` would override every stage.
                fast_all=None,
                # Keep the requested one-frame codec path for CFG=1 fallback
                # requests.  Backbone/depth stay eager, while this avoids the
                # harmless-but-noisy residual-tail codec path.
                fast_codec=settings.fast_codec,
                repetition_penalty=REPETITION_PENALTY,
            ),
            tokenizer=tokenizer,
        )
    app.state.tokenizer, app.state.model, app.state.audio_tokenizer, app.state.runtime = tokenizer, model, audio_tokenizer, runtime
    app.state.eager_runtime = eager_runtime


def _model_is_loaded(app: FastAPI) -> bool:
    return getattr(app.state, "runtime", None) is not None


def _ensure_model_loaded(app: FastAPI) -> bool:
    """Load model resources when absent; callers must hold ``_request_lock``."""
    if _model_is_loaded(app):
        return False
    try:
        _load_app(app, app.state.cfg)
    except Exception as exc:
        # ``_load_app`` publishes state only after all components are ready,
        # but a failed CUDA allocation can leave allocator cache behind.
        app.state.model_load_error = str(exc)
        _release_cuda_memory()
        raise
    app.state.model_load_error = None
    return True


def _release_cuda_memory() -> None:
    """Best-effort collection of tensors left by an unsuccessful transition."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        logger.warning("CUDA cleanup after model transition was incomplete", exc_info=True)


def _unload_app(app: FastAPI) -> bool:
    """Release all model and CUDA-graph references; caller holds the lock."""
    if not _model_is_loaded(app):
        return False

    # Clear application references before collecting so compiled modules and
    # CUDA graph pools can be reclaimed instead of remaining reachable.
    app.state.runtime = None
    app.state.eager_runtime = None
    app.state.tokenizer = None
    app.state.model = None
    app.state.audio_tokenizer = None
    app.state.model_load_error = None
    _release_cuda_memory()
    return True


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if _settings is None:
        raise RuntimeError("API settings are not initialized")
    app.state.cfg = _settings
    app.state.profiles = ProfileStore(_settings.voice_dir)
    app.state.metrics = {"requests_total": 0, "requests_success": 0, "requests_error": 0, "requests_busy": 0, "streaming_total": 0, "streaming_design": 0, "streaming_clone": 0, "ttfa_ms": [], "latency_ms": []}
    app.state.runtime = None
    app.state.eager_runtime = None
    app.state.tokenizer = None
    app.state.model = None
    app.state.audio_tokenizer = None
    app.state.model_load_error = None
    app.state.start_time = time.monotonic()
    try:
        _ensure_model_loaded(app)
    except Exception:
        # Keep the HTTP service alive after e.g. a transient CUDA OOM. The
        # explicit load endpoint and GPU-using requests can retry later.
        logger.exception("Breeze model was not loaded at startup; server is idle")
    yield
    if _model_is_loaded(app):
        _unload_app(app)


app = FastAPI(title="Breeze TTS API", lifespan=_lifespan)


@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    origin = request.headers.get("origin")
    configured = _settings.cors_origins if _settings is not None else ApiSettings.cors_origins
    allowed = origin and ("*" in configured or origin in configured)
    if request.method == "OPTIONS" and allowed:
        response = Response(status_code=204)
    else:
        response = await call_next(request)
    if allowed:
        response.headers["Access-Control-Allow-Origin"] = "*" if "*" in configured else origin
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "X-Sample-Rate, X-Audio-Sample-Rate, X-Request-Id, X-Sample-Format"
        response.headers["Vary"] = "Origin"
    return response


@app.get("/health")
def health() -> JSONResponse:
    if not _model_is_loaded(app):
        status = "load_failed" if getattr(app.state, "model_load_error", None) else "unloaded"
        return JSONResponse({"status": status, "ready": False, "model_loaded": False}, status_code=503)
    runtime = app.state.runtime
    return JSONResponse({"status": "healthy", "ready": True, "model_loaded": True, "uptime_s": round(time.monotonic() - app.state.start_time, 1), "model_id": "breeze-tts-2", "sample_rate": runtime.sample_rate, "hybrid_quantized": app.state.cfg.weights is not None, "fast_enabled": runtime.fast_enabled, "profile_count": len(app.state.profiles.list()), "memory_rss_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1)})


@app.post("/v1/model/load")
def load_model() -> dict:
    """Load Breeze onto the configured device, including enabled fast graphs."""
    if not _request_lock.acquire(blocking=False):
        raise HTTPException(409, "An inference request or model transition is already running.")
    try:
        loaded = _ensure_model_loaded(app)
    except Exception as exc:
        logger.exception("Breeze model load failed")
        raise HTTPException(500, f"Model load failed: {exc}") from exc
    finally:
        _request_lock.release()
    return {"status": "loaded", "model_loaded": True, "already_loaded": not loaded}


@app.post("/v1/model/unload")
def unload_model() -> dict:
    """Unload Breeze model resources from GPU memory."""
    if not _request_lock.acquire(blocking=False):
        raise HTTPException(409, "An inference request or model transition is already running.")
    try:
        unloaded = _unload_app(app)
    finally:
        _request_lock.release()
    return {"status": "unloaded", "model_loaded": False, "was_loaded": unloaded}


@app.get("/metrics")
def metrics() -> dict:
    values = app.state.metrics
    return {**{key: value for key, value in values.items() if key not in {"ttfa_ms", "latency_ms"}}, "latency_ms_mean": round(sum(values["latency_ms"]) / len(values["latency_ms"]), 1) if values["latency_ms"] else 0.0, "streaming_ttfa_ms_mean": round(sum(values["ttfa_ms"]) / len(values["ttfa_ms"]), 1) if values["ttfa_ms"] else 0.0, "memory_rss_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1), "profile_cache_entries": sum(1 for p in app.state.profiles.list() if p.get("cached"))}


@app.get("/v1/audio/models")
@app.get("/v1/models")
def models() -> dict:
    return {"object": "list", "data": [{"id": "breeze-tts-2", "object": "model", "owned_by": "breezeblue"}]}


def _voice_item(profile: dict) -> dict:
    return {"id": profile["profile_id"], "voice_id": profile["profile_id"], "name": profile["name"], "object": "voice", "owned_by": "breezeblue", "ref_text": profile.get("ref_text", ""), "cached": profile.get("cached", False)}


@app.get("/v1/audio/voices")
def voices() -> dict:
    builtins = [{"id": "voice-design", "voice_id": "voice-design", "name": "voice-design", "object": "voice", "owned_by": "breezeblue"}]
    return {"object": "list", "data": builtins + [_voice_item(p) for p in app.state.profiles.list()]}


@app.post("/upload_voice")
@app.post("/v1/upload_voice")
async def upload_voice(request: Request) -> dict:
    form = await request.form()
    upload = form.get("voice_file")
    data = await _upload_bytes(upload if hasattr(upload, "read") else None, app.state.cfg.max_ref_audio_bytes)
    if data is None:
        url = str(form.get("voice_url") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "voice_file or voice_url is required")
        try:
            data = await asyncio.to_thread(
                _download_voice_url, url, app.state.cfg.max_ref_audio_bytes
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"Unable to download voice_url: {exc}") from exc
        if len(data) > app.state.cfg.max_ref_audio_bytes:
            raise HTTPException(413, "voice_url response exceeds the configured size limit")
    ref_text = str(form.get("ref_text") or form.get("reference_text") or "").strip()
    if not ref_text:
        raise HTTPException(422, "ref_text is required for Breeze voice profiles")
    name = str(form.get("name") or form.get("voice_name") or uuid.uuid4().hex[:12]).strip()
    profile_id = name if all(c.isalnum() or c in "-_" for c in name) and len(name) <= 64 else uuid.uuid4().hex
    overwrite = str(form.get("overwrite", "false")).lower() == "true"
    if not overwrite:
        try:
            app.state.profiles.get(profile_id)
        except ProfileNotFoundError:
            pass
        else:
            raise HTTPException(409, "Voice profile already exists")
    preload = str(form.get("preload", "true")).lower() != "false"
    reference_codes = None
    if preload:
        if not _request_lock.acquire(blocking=False):
            raise HTTPException(409, "An inference request is already running.")
        try:
            try:
                _ensure_model_loaded(app)
            except Exception as exc:
                logger.exception("Breeze model load failed while preloading voice")
                raise HTTPException(500, f"Model load failed: {exc}") from exc
            reference_codes = await asyncio.to_thread(
                _encode_reference_bytes, app.state.audio_tokenizer, data
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(422, f"Invalid reference audio: {exc}") from exc
        finally:
            _request_lock.release()
    try:
        result = app.state.profiles.save(profile_id, data, ref_text, name=name, overwrite=overwrite)
    except ProfileExistsError:
        raise HTTPException(409, "Voice profile already exists")
    if reference_codes is not None:
        app.state.profiles.save_codes(profile_id, reference_codes)
        result = app.state.profiles.get(profile_id)
    return {"voice_id": profile_id, "id": profile_id, "name": result["name"], "cached": result.get("cached", False)}


def _profile_id(voice_id: str) -> str:
    try:
        return app.state.profiles.resolve(voice_id)
    except (ProfileNotFoundError, ValueError) as exc:
        raise HTTPException(404, f"Voice '{voice_id}' not found") from exc


@app.get("/v1/audio/voices/{voice_id}")
def get_voice(voice_id: str) -> dict:
    if voice_id == "voice-design":
        return voices()["data"][0]
    return _voice_item(app.state.profiles.get(_profile_id(voice_id)))


@app.patch("/v1/audio/voices/{voice_id}")
async def update_voice(voice_id: str, request: Request) -> dict:
    if voice_id == "voice-design": raise HTTPException(403, "Built-in voices cannot be modified")
    profile_id = _profile_id(voice_id); body = await request.json()
    result = app.state.profiles.update(profile_id, ref_text=body.get("ref_text"), name=body.get("name"))
    return _voice_item(result)


@app.delete("/v1/audio/voices/{voice_id}")
def delete_voice(voice_id: str) -> dict:
    if voice_id == "voice-design": raise HTTPException(403, "Built-in voices cannot be modified")
    profile_id = _profile_id(voice_id); app.state.profiles.delete(profile_id)
    return {"deleted": True, "voice_id": profile_id}


@app.post("/v1/voices/profiles")
async def create_profile(request: Request) -> dict:
    form = await request.form(); upload = form.get("ref_audio")
    data = await _upload_bytes(upload if hasattr(upload, "read") else None, app.state.cfg.max_ref_audio_bytes)
    profile_id = str(form.get("profile_id") or "").strip(); ref_text = str(form.get("ref_text") or "").strip()
    if not profile_id or not data or not ref_text: raise HTTPException(422, "profile_id, ref_audio, and ref_text are required")
    try: result = app.state.profiles.save(profile_id, data, ref_text, overwrite=str(form.get("overwrite", "false")).lower() == "true")
    except ProfileExistsError as exc: raise HTTPException(409, "Voice profile already exists") from exc
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    return result


@app.get("/v1/voices/profiles")
def list_profiles() -> dict:
    return {"profiles": app.state.profiles.list(), "total": len(app.state.profiles.list())}


@app.get("/v1/voices/profiles/{profile_id}")
def get_profile(profile_id: str) -> dict: return app.state.profiles.get(_profile_id(profile_id))


@app.patch("/v1/voices/profiles/{profile_id}")
async def patch_profile(profile_id: str, request: Request) -> dict:
    pid = _profile_id(profile_id); form = await request.form(); upload = form.get("ref_audio")
    data = await _upload_bytes(upload if hasattr(upload, "read") else None, app.state.cfg.max_ref_audio_bytes)
    return app.state.profiles.update(pid, audio=data, ref_text=str(form["ref_text"]) if "ref_text" in form else None)


@app.delete("/v1/voices/profiles/{profile_id}")
def remove_profile(profile_id: str) -> Response:
    app.state.profiles.delete(_profile_id(profile_id)); return Response(status_code=204)


@app.post("/v1/audio/speech")
@app.post("/v1/audio/speech/clone")
async def speech(request: Request):
    app.state.metrics["requests_total"] += 1
    if not _request_lock.acquire(blocking=False):
        app.state.metrics["requests_busy"] += 1
        raise HTTPException(409, "An inference request is already running.")
    temp_path: Path | None = None
    started = time.perf_counter()
    finish_guard = threading.Lock()
    finished = False

    def finish_request() -> None:
        nonlocal finished
        with finish_guard:
            if finished:
                return
            finished = True
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            _request_lock.release()

    try:
        try:
            _ensure_model_loaded(app)
        except Exception as exc:
            logger.exception("Breeze model load failed while serving request")
            raise HTTPException(500, f"Model load failed: {exc}") from exc
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type: form = await request.json(); upload = None
        else:
            parsed = await request.form(); form = dict(parsed); upload = parsed.get("ref_audio")
        text = str(form.get("input") or form.get("text") or "").strip()
        if not text: raise HTTPException(422, "input/text is required")
        instruction = str(form.get("instructions") if form.get("instructions") is not None else form.get("instruction") or "").strip()
        ref_text = form.get("reference_text") if form.get("reference_text") is not None else form.get("ref_text")
        ref_text = str(ref_text).strip() if ref_text is not None else None
        cfg_value = form.get("guidance_scale", form.get("cfg_scale")); voice = str(form.get("voice") or "voice-design")
        instruction, scale = _normalise_instruction_for_cfg(
            instruction, cfg_value, fast_enabled=app.state.runtime.fast_enabled
        )
        ref_request, template_name = _profile_request(app.state.profiles, voice, ref_text=ref_text, instruction=instruction)
        if hasattr(upload, "read"):
            data = await _upload_bytes(upload, app.state.cfg.max_ref_audio_bytes)
            if not ref_text: raise HTTPException(422, "ref_text is required with ref_audio")
            with tempfile.NamedTemporaryFile(prefix="breeze-ref-", suffix=".wav", delete=False) as handle:
                handle.write(data or b""); temp_path = Path(handle.name)
            ref_request.update(ref_audio_path=str(temp_path), ref_text=ref_text); template_name = "ref_edit_tata" if instruction else "ref_clone_tata"
        # Encode any uncached profile or one-off upload exactly once.  Adaptive
        # fit probes and split segments must all reuse identical reference
        # codes rather than repeatedly invoking the codec encoder.
        if ref_request.get("ref_audio_path"):
            reference_codes = encode_prompt_audio(
                app.state.audio_tokenizer, Path(ref_request["ref_audio_path"])
            )
            ref_request["ref_audio_codes"] = reference_codes
            ref_request.pop("ref_audio_path", None)
            if ref_request.get("profile_id"):
                app.state.profiles.save_codes(
                    ref_request["profile_id"], reference_codes
                )
        ref_request.update(id=uuid.uuid4().hex, instruction=instruction, speaker="S0")
        # The bundled low-VRAM fast profile warms the CFG branch.  The
        # normaliser above supplies the neutral instruction for an automatic
        # request and for any explicit CFG value that otherwise lacks one.
        seed = _parse_seed(form.get("seed", 42))
        set_all_seeds(seed)
        def prepare_segment(segment_text: str, segment_index: int = 0):
            segment_request = {
                **ref_request,
                "id": f"{ref_request['id']}-{segment_index}",
                "text": segment_text,
            }
            return prepare_inputs(
                app.state.tokenizer,
                app.state.audio_tokenizer,
                app.state.model,
                [segment_request],
                get_template(template_name),
                guidance_scale=scale,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )

        def segment_fits(segment_text: str, inputs: dict | None = None) -> bool:
            prepared = inputs if inputs is not None else prepare_segment(segment_text)
            prompt_tokens = _prompt_token_count(prepared)
            available_frames = min(
                MAX_NEW_TOKENS, MAX_SEQ_LEN - 1 - prompt_tokens
            )
            estimated_frames = estimate_speech_frames(
                app.state.tokenizer, segment_text
            )
            return estimated_frames + CONTEXT_SAFETY_FRAMES <= available_frames

        # Keep native streaming as one generation whenever the actual prompt
        # and estimated speech fit the model's 2048-token context.  Only then
        # fall back to sentence-aware long-form segmentation.
        full_inputs = prepare_segment(text)
        if segment_fits(text, full_inputs):
            text_segments = [text]
            inputs_by_segment = [full_inputs]
        else:
            text_segments = split_text_to_fit(text, segment_fits)
            inputs_by_segment = [
                prepare_segment(segment_text, segment_index)
                for segment_index, segment_text in enumerate(text_segments)
            ]
            if any(
                not segment_fits(segment_text, inputs)
                for segment_text, inputs in zip(text_segments, inputs_by_segment)
            ):
                raise HTTPException(
                    422,
                    "Reference/instruction prompt leaves too little context for synthesis",
                )
        logger.info(
            "Breeze request plan: segments=%d prompt_tokens=%d "
            "estimated_frames=%d max_seq_len=%d",
            len(text_segments),
            _prompt_token_count(inputs_by_segment[0]),
            estimate_speech_frames(app.state.tokenizer, text_segments[0]),
            MAX_SEQ_LEN,
        )
        # Every prompt is constructed before sending a streaming response, so
        # invalid combinations cannot become truncated responses after headers.
        request_id = f"api-{uuid.uuid4().hex}"
        response_format = str(form.get("response_format") or "wav").lower()
        stream_format = str(form.get("stream_format") or "audio").lower()
        if stream_format not in {"audio", "sse"}:
            raise HTTPException(422, "stream_format must be audio or sse")
        stream = _parse_stream(form.get("stream", False)) or stream_format == "sse"
        sample_rate = app.state.runtime.sample_rate
        if response_format not in {"wav", "pcm"}:
            raise HTTPException(422, "Breeze supports response_format=wav or pcm")
        if stream:
            app.state.metrics["streaming_total"] += 1
            app.state.metrics["streaming_clone" if "ref_text" in ref_request else "streaming_design"] += 1

            def stream_body() -> Iterator[bytes]:
                chunk_count = 0
                pending: bytes | None = None
                first_audio_at: float | None = None
                try:
                    active_runtime = getattr(app.state, "eager_runtime", None) if scale == 1.0 else None
                    active_runtime = active_runtime or app.state.runtime
                    for segment_index, inputs in enumerate(inputs_by_segment):
                        segment_request_id = f"{request_id}-{segment_index}"
                        # Breeze sampling can drift noticeably when one long
                        # request carries RNG state across independent model
                        # calls.  Restart each segment from the request seed;
                        # its different text still produces different speech,
                        # while the cloned speaker remains much more stable.
                        set_all_seeds(seed)
                        logger.info(
                            "Breeze segment synthesis: request_id=%s "
                            "segment=%d/%d clone=%s profile_id=%s "
                            "cached_ref_codes=%s",
                            request_id,
                            segment_index + 1,
                            len(inputs_by_segment),
                            "ref_text" in ref_request,
                            ref_request.get("profile_id"),
                            "ref_audio_codes" in ref_request,
                        )
                        for chunk in active_runtime.iter_audio_chunks(
                            inputs, request_id=segment_request_id, seed=seed
                        ):
                            item = _pcm16(chunk.audio)
                            if not item:
                                continue
                            if first_audio_at is None:
                                first_audio_at = time.perf_counter()
                                app.state.metrics["ttfa_ms"].append(
                                    (first_audio_at - started) * 1000
                                )
                            chunk_count += 1
                            if stream_format == "sse":
                                if pending is not None:
                                    payload = {"type": "audio.chunk", "data": base64.b64encode(_wav(pending, sample_rate) if response_format == "wav" else pending).decode(), "format": response_format, "sample_rate": sample_rate, "chunk_index": chunk_count - 2, "final": False}
                                    yield (f"data: {json.dumps(payload)}\n\n").encode()
                                pending = item
                            else:
                                # A RIFF/WAV header carries a final data
                                # length that is unknown while streaming.  A
                                # header for only the first chunk makes strict
                                # players truncate the rest of the response.
                                # Raw PCM is the compatible streaming format;
                                # the sample-rate headers describe it.
                                yield item
                    if chunk_count == 0:
                        raise RuntimeError("Breeze produced an empty audio response")
                    if stream_format == "sse":
                        if pending is not None:
                            payload = {"type": "audio.chunk", "data": base64.b64encode(_wav(pending, sample_rate) if response_format == "wav" else pending).decode(), "format": response_format, "sample_rate": sample_rate, "chunk_index": chunk_count - 1, "final": True}
                            yield (f"data: {json.dumps(payload)}\n\n").encode()
                        yield (f"data: {json.dumps({'type': 'done', 'chunks': chunk_count, 'request_id': request_id})}\n\n").encode()
                        yield b"data: [DONE]\n\n"
                    app.state.metrics["requests_success"] += 1
                    app.state.metrics["latency_ms"].append(
                        (time.perf_counter() - started) * 1000
                    )
                except Exception as exc:
                    app.state.metrics["requests_error"] += 1
                    logger.exception("Breeze streaming synthesis failed")
                    if stream_format == "sse":
                        yield f"data: {json.dumps({'type': 'error', 'error': str(exc), 'message': str(exc)})}\n\n".encode()
                    else:
                        raise
                finally:
                    finish_request()

            media = "text/event-stream" if stream_format == "sse" else "audio/pcm"
            handoff = True
            return StreamingResponse(
                stream_body(),
                media_type=media,
                headers={"Cache-Control": "no-cache", "X-Sample-Rate": str(sample_rate), "X-Audio-Sample-Rate": str(sample_rate), "X-Sample-Format": "s16le", "X-Request-Id": request_id},
                background=BackgroundTask(finish_request),
            )
        active_runtime = getattr(app.state, "eager_runtime", None) if scale == 1.0 else None
        active_runtime = active_runtime or app.state.runtime
        chunks = []
        for segment_index, inputs in enumerate(inputs_by_segment):
            set_all_seeds(seed)
            logger.info(
                "Breeze segment synthesis: request_id=%s segment=%d/%d "
                "clone=%s profile_id=%s cached_ref_codes=%s",
                request_id,
                segment_index + 1,
                len(inputs_by_segment),
                "ref_text" in ref_request,
                ref_request.get("profile_id"),
                "ref_audio_codes" in ref_request,
            )
            for chunk in active_runtime.iter_audio_chunks(
                inputs, request_id=f"{request_id}-{segment_index}", seed=seed
            ):
                item = _pcm16(chunk.audio)
                if item:
                    chunks.append(item)
        pcm = b"".join(chunks)
        if not pcm:
            raise RuntimeError("Breeze produced an empty audio response")
        app.state.metrics["requests_success"] += 1
        app.state.metrics["latency_ms"].append((time.perf_counter() - started) * 1000)
        output = _wav(pcm, sample_rate) if response_format == "wav" else pcm
        return Response(output, media_type="audio/wav" if response_format == "wav" else "audio/pcm", headers={"X-Sample-Rate": str(sample_rate), "X-Audio-Sample-Rate": str(sample_rate), "X-Request-Id": request_id})
    except HTTPException: app.state.metrics["requests_error"] += 1; raise
    except Exception as exc: app.state.metrics["requests_error"] += 1; raise HTTPException(500, f"Synthesis failed: {exc}") from exc
    finally:
        if not locals().get("handoff", False):
            finish_request()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Breeze TTS 2"); parser.add_argument("model", type=Path); parser.add_argument("--weights", type=Path); parser.add_argument("--voice-dir", type=Path, default=ApiSettings.voice_dir); parser.add_argument("--max-ref-audio-mb", type=int, default=25); parser.add_argument("--cors-origins", default=','.join(ApiSettings.cors_origins)); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=7860)
    for flag in ("fast-all", "fast-text-encoder", "fast-backbone-prefill", "fast-backbone-decode", "fast-depth-decoder", "fast-codec"): parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=None if flag == "fast-all" else False)
    args = parser.parse_args(); global _settings
    _settings = ApiSettings(model=args.model, weights=args.weights, voice_dir=args.voice_dir, max_ref_audio_bytes=args.max_ref_audio_mb * 1024 * 1024, cors_origins=tuple(x.strip() for x in args.cors_origins.split(',') if x.strip()), fast_all=args.fast_all, fast_text_encoder=args.fast_text_encoder, fast_backbone_prefill=args.fast_backbone_prefill, fast_backbone_decode=args.fast_backbone_decode, fast_depth_decoder=args.fast_depth_decoder, fast_codec=args.fast_codec)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__": main()
