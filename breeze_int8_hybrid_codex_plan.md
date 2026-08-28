# Breeze TTS 2 Hybrid INT8 — Codex Implementation & Test Plan

**Target repository:** local `breeze-tts` checkout  
**Primary target:** standalone Breeze FastAPI streaming server on CUDA  
**Quantized checkpoint:** `Breeze-TTS-2-int8-hybrid.safetensors` in the repository root (confirm exact filename before editing)  
**Model assets/config:** existing `models/Breeze-TTS-2/` directory  
**Goal:** run the hybrid INT8 checkpoint without ComfyUI, preserving the official Breeze streaming API and the existing fast depth-decoder / codec paths.

---

## 1. Objective

Implement support for the `drbaph/Breeze-TTS-2-comfyui` **hybrid INT8 ConvRot** checkpoint in the existing Breeze TTS 2 repository.

The hybrid checkpoint intentionally uses:

- **INT8 ConvRot:** Breeze speech backbone
- **INT8 ConvRot:** T5Gemma2 text encoder
- **BF16:** depth decoder / per-frame autoregressive hot loop
- **Unmodified/BF16 runtime:** audio tokenizer / codec

The expected advantage is a large VRAM reduction without the major decode-speed regression seen when the depth decoder is also quantized.

Do **not** port the ComfyUI application or ComfyUI memory manager. Reuse/adapt only the minimal quantized linear implementation and checkpoint-loading logic needed for the standalone Breeze runtime.

---

## 2. Current Known-Good Baseline

Before implementing anything, preserve and re-measure the current BF16 baseline.

Current tested environment:

- Linux / CachyOS
- NVIDIA RTX 4070 Ti SUPER 16 GB
- NVIDIA driver `610.57.04`
- PyTorch `2.9.1+cu128`
- Torch CUDA `12.8`
- Python `3.12`
- FlashAttention `2.8.3`
- API streaming output: mono 24 kHz signed 16-bit little-endian PCM

Current memory/performance tuning already applied locally:

- FlashAttention installed and functional.
- API `MAX_SEQ_LEN` reduced from 2048 to **512**.
- Fast warmup profile has been reduced to **CFG4 / branch batch 2 only**.
- Server is run with:

```bash
python -m breeze_infer.api models/Breeze-TTS-2 \
  --host 0.0.0.0 \
  --port 7860 \
  --fast-depth-decoder \
  --fast-codec
```

Known BF16 CFG4 benchmark:

- Peak/steady process VRAM reported by `nvidia-smi`: **8120 MiB**
- RTF: **0.651**
- Speed: **1.54x realtime**

Known fuller fast configuration (`fast-backbone-decode + fast-depth-decoder + fast-codec`) is faster but uses too much VRAM for an 8 GB GPU.

### Critical requirement

**Do not discard or overwrite current uncommitted local changes.** Start by recording:

```bash
git status --short
git diff > /tmp/breeze-pre-int8.patch
```

If convenient, work on a new branch/worktree without losing current changes. Do not reset the repository.

---

## 3. Source Implementations to Reuse

Use the following projects as implementation references.

### Official Breeze runtime

Repository:

`https://github.com/breezeblue-ai/breeze-tts`

Important files:

- `breeze_infer/runtime.py`
- `breeze_infer/api.py`
- `models/breeze.py`
- `models/breeze_config.py`
- `models/fast_streaming.py`
- `models/cudagraph/depth_decoder_graph.py`
- `configs/fast.json`

The normal loader currently uses:

```python
BreezeForConditionalGeneration.from_pretrained(..., dtype=torch.bfloat16)
```

That path cannot directly load the derivative INT8 checkpoint because quantized `nn.Linear` modules must be replaced **before** assigning the INT8 weights.

### ComfyUI Breeze INT8 reference

Repository:

`https://github.com/Saganaki22/ComfyUI-Breeze-TTS-2`

Important files:

- `int8.py`
- `loader.py`
- `native.py`

Key reusable concept from `int8.py`:

- scan `*.comfy_quant` safetensors metadata;
- identify quantized linear prefixes;
- replace the corresponding `nn.Linear` modules with a custom `ConvRotInt8Linear`;
- retain weight tensor as `torch.int8`;
- retain `weight_scale` as FP32;
- execute the layer through `comfy_kitchen.int8_linear(...)` with `convrot=True`.

The reference `ConvRotInt8Linear.forward()` concept is:

```python
return comfy_kitchen.int8_linear(
    x.contiguous(),
    self.weight,
    self.weight_scale,
    self.bias,
    out_dtype=x.dtype,
    convrot=True,
    convrot_groupsize=self.convrot_groupsize,
)
```

Do not silently dequantize the checkpoint back to BF16. The INT8 weights must remain resident as INT8 during inference.

---

## 4. Dependency Strategy

Keep the dependency addition minimal.

Required/expected packages:

```bash
uv pip install comfy-kitchen accelerate safetensors
```

`accelerate` and `safetensors` may already exist; do not unnecessarily replace working versions.

Before modifying Breeze, verify the kernel runtime:

```bash
python - <<'PY'
import torch
import comfy_kitchen

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("backends:", comfy_kitchen.list_backends())
print("cuda available:", torch.cuda.is_available())
PY
```

The implementation must verify that a CUDA-capable `comfy_kitchen` backend is available. Do not accept an accidental slow CPU/eager fallback as a passing performance result.

Add only the minimum new dependency declarations required for standalone INT8 support. Do **not** add ComfyUI as a dependency.

---

## 5. First Task: Inspect the Downloaded Checkpoint

Before coding the loader, inspect the exact local file.

Expected filename:

```text
./Breeze-TTS-2-int8-hybrid.safetensors
```

Confirm:

```bash
ls -lh *Breeze*TTS*int8*hybrid*.safetensors
```

Create a small inspection script, preferably `scripts/inspect_quant_checkpoint.py`, that prints:

- file size;
- total key count;
- number of `*.comfy_quant` metadata entries;
- number of `*.weight_scale` tensors;
- quantization group sizes;
- first 20 quantized module prefixes;
- counts grouped by top-level component, especially:
  - `text_encoder`
  - `backbone_model`
  - `depth_decoder`
  - anything unexpected.

### Mandatory assertion

The **hybrid** checkpoint must not quantize the depth decoder hot loop. Fail loudly if quant metadata is found under a depth-decoder prefix unless the checkpoint naming proves the prefix classification is different.

Also inspect the checkpoint's non-quantized tensor dtypes. Do not assume every non-INT8 tensor is BF16.

---

## 6. Architecture: Keep the Existing Runtime, Add an Alternate Weight Loader

Preferred design:

### New module

Create something like:

```text
breeze_infer/int8_convrot.py
```

Adapt the small, self-contained parts of the ComfyUI node's `int8.py`:

- `QuantLayerInfo`
- `scan_checkpoint_quantization()`
- `ConvRotInt8Linear`
- `replace_quantized_linears()`
- `quantized_parameter_count()`
- a runtime/backend validation helper

Include source attribution and the Apache-2.0 provenance in the file header.

### New quantized loader

Add either:

```text
breeze_infer/quantized_runtime.py
```

or a clearly isolated helper inside `breeze_infer/runtime.py`.

Do not make the normal BF16 loader complicated. Prefer:

```python
load_runtime(..., weights_path: Path | None = None)
```

Behavior:

- `weights_path is None` -> existing official BF16 path unchanged.
- plain float safetensors -> reject unless explicitly supported.
- ConvRot metadata present -> quantized loader path.

### API CLI

Add an optional argument such as:

```bash
--weights /path/to/Breeze-TTS-2-int8-hybrid.safetensors
```

Example desired command:

```bash
python -m breeze_infer.api models/Breeze-TTS-2 \
  --weights ./Breeze-TTS-2-int8-hybrid.safetensors \
  --host 0.0.0.0 \
  --port 7860 \
  --fast-depth-decoder \
  --fast-codec
```

The positional model directory remains the source for:

- `config.json`
- tokenizer files
- generation config
- `audio_tokenizer/`

The separate `--weights` file overrides only the Breeze main-model weights.

This preserves backward compatibility and avoids duplicating all model assets next to the derivative checkpoint.

---

## 7. Quantized Model Construction

The quantized loader must instantiate the **official Breeze model architecture without first allocating BF16 copies of all weights**.

Preferred approach:

1. Load `BreezeConfig` from `models/Breeze-TTS-2`.
2. Instantiate `BreezeForConditionalGeneration` under `accelerate.init_empty_weights()` / meta-device construction if compatible.
3. Scan the hybrid safetensors quant metadata.
4. Replace matching `nn.Linear` modules with `ConvRotInt8Linear` **before materializing tensors**.
5. Materialize checkpoint tensors directly onto the target CUDA device with an explicit dtype policy.
6. Materialize any remaining meta buffers.
7. `model.eval()` and disable gradients.
8. Load the existing Qwen audio tokenizer exactly as the official runtime does.

### Important: no transient full BF16 model

A loader that first loads the complete BF16 checkpoint and then swaps/quantizes layers defeats much of the memory goal and can OOM on the 8 GB target. Avoid it.

### Module-name compatibility check

Before loading weights, compare every quantized prefix to `dict(model.named_modules())`.

Required result:

```text
quantized checkpoint prefixes matched: 100%
```

If names differ, build the smallest explicit mapping required and document it. Do not use fuzzy matching that could quantize the wrong layer.

---

## 8. Safetensors Materialization Rules

Implement an explicit loader rather than relying on an opaque conversion.

For each checkpoint tensor:

### Quantized module weights

For `ConvRotInt8Linear`:

- `.weight` -> retain `torch.int8`
- `.weight_scale` -> retain `torch.float32`
- `.bias` -> preserve checkpoint dtype unless the reference implementation requires another dtype
- `.comfy_quant` -> metadata only; do not try to assign it as a model parameter

### Non-quantized model tensors

Preserve the checkpoint/reference dtype policy. Prefer BF16 for ordinary model weights unless the checkpoint explicitly stores a numerically sensitive head differently.

Verify the reference implementation's treatment of:

- `lm_head.weight`
- `depth_decoder.codebooks_head.weight`

The ComfyUI reference intentionally uses FP32 for some output heads. Do not change the current official runtime's numerical behavior blindly. Benchmark both correctness and memory before adopting any dtype change solely because ComfyUI does it.

### Post-load invariants

Add a diagnostic function that prints/asserts:

- quantized module count;
- INT8 parameter count;
- total INT8 bytes;
- total BF16/FP32 bytes;
- no parameters remain on `meta`;
- all expected runtime parameters are on the target device;
- no quantized module exists below `depth_decoder` for the hybrid checkpoint.

---

## 9. Tokenizer Fix

The current local API still emits the Mistral regex warning.

Fix tokenizer loading in both normal and quantized paths using:

```python
AutoTokenizer.from_pretrained(
    ckpt_dir,
    fix_mistral_regex=True,
)
```

Do this once in the shared loading path if possible.

This is a correctness cleanup, not a performance optimization.

---

## 10. FlashAttention

FlashAttention 2.8.3 is already installed and verified with Torch 2.9.1 / CUDA 12.8.

Preserve it.

Do not require FlashAttention for the ConvRot implementation itself, but the quantized server should continue to use the same attention implementation that is currently working for the text encoder.

During startup, log which attention implementation is active.

No FlashAttention warning should appear in the successful target configuration.

---

## 11. CUDA Graph Compatibility

For the first implementation, **do not enable fast backbone decode**.

Target fast stages remain:

```text
fast_depth_decoder = true
fast_codec = true
fast_backbone_decode = false
fast_backbone_prefill = false
fast_text_encoder = false
```

Why:

- the hybrid checkpoint quantizes the backbone;
- the existing fast backbone graph was expensive in VRAM;
- we already have realtime headroom without it;
- keeping ConvRot INT8 outside the existing backbone CUDA graph minimizes integration risk;
- the depth decoder stays BF16 specifically so its existing fast/CUDA-graph path remains appropriate.

If the baseline hybrid implementation is stable, a later optional phase can investigate graph capture around INT8 backbone operations. That is **not** part of the minimum viable implementation.

---

## 12. Preserve Current Memory Tuning

Keep the current local configuration for the first hybrid tests:

```text
MAX_SEQ_LEN = 512
CFG scales = [4.0]
backbone branch batches = [2]
depth decoder warmup batches = [2]
```

Do not reduce to 256 yet.

The quantized checkpoint should provide enough memory headroom that further sequence-cache compromises are unnecessary.

A later cleanup should make `MAX_SEQ_LEN` configurable by CLI instead of a source edit, e.g.:

```bash
--max-seq-len 512
```

Do this only if it stays a small, low-risk change.

---

## 13. Kernel Smoke Test

Before loading the full model, create `scripts/test_convrot_kernel.py`.

It should:

1. import `comfy_kitchen`;
2. print available backends;
3. create a small BF16 activation tensor on CUDA;
4. create/quantize a compatible test weight using the ConvRot layout or use a tiny synthetic INT8 weight+scale pair;
5. call `comfy_kitchen.int8_linear()`;
6. verify output shape, dtype and all-finite values;
7. exit nonzero if the expected CUDA path is unavailable.

This separates dependency/kernel problems from Breeze loader problems.

---

## 14. Loader Smoke Test

Create a script such as:

```text
scripts/test_hybrid_load.py
```

It should load the model but not generate audio, then print:

```text
weights path
quantized modules
quantized parameters
quantized components
model parameter dtypes
CUDA allocated MiB
CUDA reserved MiB
nvidia-smi process MiB (if easy to obtain)
```

Then verify at least one real `ConvRotInt8Linear` call executes on a tiny model forward/prefill path if practical.

The runtime stats / call counter should prove that the quantized kernel is actually being used.

---

## 15. Functional Inference Tests

Use the exact same benchmark sentence for regression comparison.

### API request

```bash
curl -sS --fail \
  -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F 'text=This is a longer test sentence designed to measure the real time generation speed of Breeze TTS on this computer. We want enough speech to get a meaningful performance measurement.' \
  -F 'instruction=A natural adult male voice speaking clearly and conversationally.' \
  -F 'cfg_scale=4' \
  -F 'seed=42' \
  -o /tmp/breeze-hybrid.pcm
```

Verify:

- HTTP 200;
- output is non-empty;
- response remains 24 kHz mono signed 16-bit little-endian PCM;
- duration is sensible;
- no NaN/Inf kernel failures;
- no dtype mismatch;
- no meta-device errors;
- no attempt to interpret INT8 weights as floating weights.

---

## 16. Streaming Test

Use live playback:

```bash
curl -sS -N --fail \
  -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F 'text=This audio should play continuously while Breeze generates the remainder of the sentence. We are testing the hybrid INT8 streaming implementation.' \
  -F 'instruction=A natural adult male voice speaking clearly and conversationally.' \
  -F 'cfg_scale=4' \
  -F 'seed=42' \
  | ffplay -nodisp -autoexit \
      -f s16le \
      -ar 24000 \
      -ch_layout mono \
      -
```

Acceptance:

- audio starts promptly;
- playback remains smooth;
- no repeated underruns/stuttering;
- request lock is released at completion;
- aborting a client request does not permanently wedge the API lock.

---

## 17. Benchmark Harness

Create a reusable benchmark script rather than relying on shell timing.

Suggested file:

```text
scripts/benchmark_api.py
```

Measure:

- wall time;
- bytes received;
- audio duration (`bytes / 48000` for s16le mono 24 kHz);
- RTF (`wall / audio_duration`);
- realtime multiplier (`audio_duration / wall`);
- time to response headers;
- optional time to first audio bytes;
- CUDA memory allocated/reserved if exposed by server;
- external process VRAM if accessible.

Run the same prompt at least 3 times after warmup and report median/mean.

Do not include initial model compilation/warmup in steady-state RTF unless reported separately.

---

## 18. Optional Debug Memory Endpoint

For testing only, it is acceptable to add a local endpoint such as:

```text
GET /debug/memory
```

Return:

```json
{
  "allocated_mib": 0,
  "reserved_mib": 0,
  "max_allocated_mib": 0,
  "max_reserved_mib": 0,
  "device_free_mib": 0,
  "device_total_mib": 0
}
```

If retained, mark it clearly as a diagnostic endpoint or gate it behind an environment variable / CLI flag.

Remember: `nvidia-smi` process usage includes CUDA allocator reservation and is still the most relevant high-level deployment number, while Torch allocated/reserved metrics help explain the footprint.

---

## 19. Performance / Memory Acceptance Criteria

### Hard functional requirements

All must pass:

- BF16 path still loads and runs when `--weights` is omitted.
- Hybrid checkpoint loads successfully with `--weights`.
- Quantized weights remain INT8 in memory.
- `comfy_kitchen.int8_linear()` is actually called.
- Text encoder + backbone use the hybrid INT8 modules.
- Depth decoder remains non-quantized/BF16.
- Fast depth-decoder path remains functional.
- Fast codec path remains functional.
- CFG4 request produces intelligible speech.
- streaming response remains smooth.
- no recurring CUDA OOM or memory leak over repeated requests.

### 4070 Ti SUPER target

Preferred result:

- **Peak process VRAM <= 6.5 GiB**
- stretch goal: around **5.5–6.0 GiB**
- **RTF <= 0.80** at CFG4
- hard performance ceiling: **RTF < 1.0**
- TTFA should remain subjectively fast; no strict target unless regression is obvious.

The ComfyUI derivative reports approximately 5.53 GiB peak on an RTX 5090 for the hybrid build, but do not assume the standalone Breeze runtime will exactly match it.

### 8 GB RTX 3070 viability criterion

Recommend moving to real 3070 testing if the 4070 Ti SUPER result is approximately:

```text
peak process VRAM <= 6.5 GiB
RTF <= 0.8
```

This should leave meaningful deployment headroom on an 8 GB card.

The 3070 must still be tested directly because compute performance will be lower than the 4070 Ti SUPER.

---

## 20. Quality Regression Testing

Quantized audio does not need to be waveform-identical to BF16.

Generate BF16 and hybrid outputs with the same:

- text;
- instruction;
- CFG scale;
- seed.

Test at least:

1. voice design prompt;
2. voice direction/style prompt;
3. longer sentence;
4. punctuation/numbers;
5. Chinese sample if convenient;
6. voice cloning if a reference file is available.

Compare:

- intelligibility;
- missing/repeated words;
- abnormal pauses;
- duration;
- obvious voice-quality degradation;
- clipping;
- silence;
- NaNs.

If an ASR tool is already available, optionally compare transcripts/WER. Do not install a large ASR stack solely for the first implementation.

---

## 21. Stability Test

After a successful hybrid implementation, issue at least 10 sequential requests against the same loaded server.

Track:

- process VRAM before request 1;
- peak VRAM;
- process VRAM after request 10;
- CUDA allocated/reserved memory;
- RTF trend;
- API lock state;
- exceptions/warnings.

Acceptance:

- no monotonic VRAM growth beyond normal allocator settling;
- no failed second/subsequent request;
- no stale codec request state;
- no degraded RTF after repeated requests.

---

## 22. Error Handling Requirements

Fail clearly for:

- `--weights` file missing;
- checkpoint has no recognized ConvRot metadata;
- unsupported quant format;
- module prefix in checkpoint does not exist in Breeze model;
- quant metadata shape disagrees with target `nn.Linear`;
- group size invalid;
- `comfy-kitchen` unavailable;
- CUDA backend unavailable;
- unsupported device for INT8 runtime;
- meta parameters remain after loading;
- hybrid checkpoint unexpectedly quantizes depth decoder.

Do not silently fall back to loading the INT8 checkpoint as BF16.

---

## 23. Logging

On successful startup with hybrid weights, print a concise banner similar to:

```text
Breeze quantization: INT8 ConvRot hybrid
weights: ./Breeze-TTS-2-int8-hybrid.safetensors
quantized modules: N
quantized params: X.XXB
components: text_encoder, backbone_model
depth decoder: BF16
comfy-kitchen backend: cuda
attention: flash_attention_2
max_seq_len: 512
fast stages: depth_decoder, codec
```

This should make it obvious when a performance test is using the intended runtime.

---

## 24. Tests to Add

Add lightweight automated tests that do not require the full checkpoint where possible.

### Unit tests

Test:

- ConvRot metadata parsing;
- group-size validation;
- module replacement;
- INT8 dtype preserved;
- `weight_scale` FP32 preserved;
- unsupported metadata rejected;
- quantized target shape mismatch rejected;
- BF16 loader unchanged when no quant weights are passed.

Use a tiny synthetic `nn.Module` and synthetic safetensors fixture for unit tests.

### GPU integration tests

Keep full-model tests separate/optional because they require CUDA and the local checkpoint.

Document a command such as:

```bash
pytest -q -m gpu_int8
```

or provide a standalone integration script if repository test conventions make that cleaner.

---

## 25. Recommended Implementation Order

### Phase A — repository safety and inspection

1. Save `git diff`.
2. Confirm hybrid checkpoint path/size.
3. Inspect quant metadata and component distribution.
4. Verify `comfy-kitchen` CUDA backend.

### Phase B — reusable INT8 runtime

5. Add `breeze_infer/int8_convrot.py`.
6. Add synthetic unit tests.
7. Confirm standalone `ConvRotInt8Linear` kernel execution.

### Phase C — model loader

8. Add optional `weights_path` to runtime loader.
9. Instantiate official Breeze model without allocating BF16 weight copies.
10. Replace quantized linears before materialization.
11. Materialize hybrid safetensors with correct dtype rules.
12. Add model integrity checks and startup banner.
13. Fix `fix_mistral_regex=True` in tokenizer loading.

### Phase D — API integration

14. Add `--weights` CLI option.
15. Preserve normal BF16 behavior when absent.
16. Start hybrid API without fast flags first.
17. Generate one successful CFG4 sample.

### Phase E — optimized hybrid path

18. Enable `--fast-depth-decoder --fast-codec`.
19. Keep `MAX_SEQ_LEN=512` and CFG4-only warmup.
20. Verify CUDA graph capture still succeeds for the BF16 depth decoder.
21. Benchmark RTF and VRAM.
22. Test streaming playback.

### Phase F — regression/stability

23. BF16 regression test.
24. Hybrid quality comparison.
25. 10-request stability test.
26. Write final measured result table.

---

## 26. Do Not Do These in the First Pass

Avoid scope creep:

- do not port all of ComfyUI;
- do not use ComfyUI's memory manager/AIMDO;
- do not quantize the depth decoder;
- do not implement a new quantizer—the checkpoint is already quantized;
- do not enable fast backbone decode initially;
- do not lower sequence length below 512 until hybrid memory is measured;
- do not replace the existing API contract;
- do not remove the BF16 loader;
- do not redesign the codec;
- do not change sampling defaults while measuring quantization impact.

---

## 27. Expected Final Command

The final standalone target should look approximately like:

```bash
python -m breeze_infer.api models/Breeze-TTS-2 \
  --weights ./Breeze-TTS-2-int8-hybrid.safetensors \
  --host 0.0.0.0 \
  --port 7860 \
  --fast-depth-decoder \
  --fast-codec
```

And the existing request contract should remain:

```bash
curl -N -X POST http://127.0.0.1:7860/v1/audio/speech \
  -F 'text=Hello from the hybrid INT8 Breeze runtime.' \
  -F 'instruction=A natural adult male voice speaking clearly and conversationally.' \
  -F 'cfg_scale=4' \
  -F 'seed=42'
```

---

## 28. Final Report Required From Codex

When finished, provide a concise handoff report containing:

### Files changed

List each file and why.

### Dependency changes

Exact installed/added package versions.

### Loader validation

Report:

```text
quantized module count:
quantized parameter count:
INT8 components:
depth decoder dtype:
comfy-kitchen backend:
FlashAttention status:
```

### Performance table

Use this structure:

| Runtime | CFG | Fast stages | RTF | Realtime speed | Peak process VRAM |
|---|---:|---|---:|---:|---:|
| BF16 baseline | 4 | depth + codec | 0.651 | 1.54x | 8120 MiB |
| Hybrid INT8 | 4 | depth + codec | measured | measured | measured |

### Quality/stability

State:

- whether speech is intelligible;
- whether streaming is smooth;
- whether quality appears equivalent/acceptable versus BF16;
- whether 10 sequential requests pass;
- whether VRAM stabilizes;
- any remaining warnings.

### Recommendation

Explicitly answer:

> Is the result ready to move to an 8 GB RTX 3070 for real-hardware testing?

Use the criteria in this document rather than intuition alone.

---

## 29. Reference URLs

- Official Breeze TTS 2: https://github.com/breezeblue-ai/breeze-tts
- Hybrid derivative model: https://huggingface.co/drbaph/Breeze-TTS-2-comfyui
- ComfyUI Breeze implementation: https://github.com/Saganaki22/ComfyUI-Breeze-TTS-2
- ConvRot INT8 reference implementation: https://github.com/Saganaki22/ComfyUI-Breeze-TTS-2/blob/main/int8.py
- Comfy Kitchen: https://github.com/Comfy-Org/comfy-kitchen
- audio.cpp (future native comparison): https://github.com/0xShug0/audio.cpp

---

## 30. Decision Context

The end goal is a dedicated, persistent Breeze TTS streaming service that can plausibly run on a **single 8 GB RTX 3070**. The current BF16 Python path is already fast enough but is too close to the 8 GB VRAM ceiling. The hybrid checkpoint is attractive because it reduces the large backbone/text-encoder weight footprint while intentionally preserving the BF16 depth decoder that dominates the per-frame hot loop.

The first priority is therefore **memory reduction without losing realtime streaming**, not maximum benchmark speed.
