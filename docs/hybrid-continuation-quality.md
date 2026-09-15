# Hybrid continuation quality investigation

## Goal

Keep the API's retained continuation semantics and roughly current hybrid VRAM
footprint while minimizing audible glitches and instruction-following regressions.

## Findings to date

- The current continuation runtime terminates each segment at sampled acoustic
  EOS. `estimated_audio_frames` only reserves context capacity; it does not
  truncate generation.
- Sky boundary diagnostics account for every generated decodable frame. Each
  produced 1,920 samples, with no dropped terminal frame or leading-silence
  trimming.
- The boundary-2 hallucination remains when the externally inserted 120 ms PCM
  silence is removed, when codec state is reset, and when the same codes are
  decoded independently. It is therefore in the generated codec frames.
- In the affected Sky run, the first semantic codebook frames of segment 3
  were `225, 225, 1648` for exact FP32 ConvRot scale values, but `1464, 225,
  1648` for both rounded-scale hybrid and original BF16 runs. The audible
  defect occurs in the first two frames around that transition.
- BF16-rounded ConvRot scale *values* restore the prior hybrid delivery and
  substantially reduce the observed boundary artifacts. Sampling heads are
  not the driver: rounding them alone did not restore that trajectory.
- The original BF16 model can still produce occasional small artifacts, so no
  numerical policy can promise zero model-originated failures.

## Implemented policy

Hybrid scales remain FP32 tensors because the ConvRot kernel requires FP32
storage. For the known Breeze hybrid checkpoint, the server defaults to
`bf16_compat`: checkpoint scale values are rounded BF16 -> FP32 before loading.
This reproduces the previous hybrid numerical trajectory without adding model
storage. The 378 scales occupy about 3.94 MiB in either mode.

`exact_fp32` remains available through `--hybrid-scale-mode` for controlled
evaluation. Both sampling heads remain exact FP32 in either policy.

### Implementation validation (2026-09-10)

- The production default was changed to `bf16_compat`; `exact_fp32` is an
  explicit diagnostic override, not an accuracy upgrade.
- A real-checkpoint audit passed: all 378 ConvRot weights are INT8, all 378
  scales are FP32 buffers containing the BF16-compatible values, and both
  sampling heads equal their original FP32 checkpoint tensors exactly.
- The audited live hybrid model allocated 4,177 MiB on the RTX 4070 Ti SUPER.
  The policy only changes existing scale values, so it adds no persistent VRAM.
- A new Sky seed-42 file was generated at
  `outputs/sky-bf16-compat/continuation_sample_01_sky.wav`. It has the same
  13.88 s duration as the prior rounded-scale control. It is intentionally
  retained for listening review; WAV bytes are not a reliable equivalence
  criterion for this CUDA-graph streaming run.
- **Listening result:** the BF16-compatible Sky render is clean, with no
  audible segment-boundary artifacts. This is now the preferred production
  policy. The earlier exact-FP32-scale render was not a fidelity improvement:
  it introduced the boundary hallucination despite using the same retained
  continuation protocol and parameters.

### BF16-compatible retained-continuation sweep (2026-09-10)

- Completed seeds 42 through 61 (20 trajectories) using the Sky reference,
  `ref_edit_tata`, CFG 4, temperature 0.9, top-k 50, top-p 1, repetition
  penalty 1.1, Audio-EOS, and the fast backbone/depth/codec paths.
- Each seed was set once at segment 1 and its RNG progressed through the next
  three retained-continuation segments. The set contains 20 full assemblies
  and 60 two-second live boundary clips under
  `outputs/continuation-sweep/bf16-compat/`.
- Machine checks found zero frame-accounting failures: every captured frame
  was decoded once and yielded its expected 1,920 PCM samples. This rules out
  a dropped generated frame in these runs, but does not replace listening
  classification of model-originated artifacts.
- Listening across the sweep found intermittent edge artifacts despite the
  clean seed-42 BF16-compatible control. The edge concentration is therefore
  an active continuation-boundary hypothesis, not evidence that the BF16
  policy solved the general problem. Exact-FP32 seed 43 was generated for a
  direct matched-seed listening comparison.
- Exact-FP32 seed 43 produces a different trajectory and exhibits artifacts as
  expected. The next isolated variable is the explicit Audio-EOS embedding
  inserted before every appended text segment; an otherwise matched
  BF16-compatible seed-43 run with that embedding disabled is complete. Its
  first segment remains 25 frames (as expected before a continuation boundary);
  later segments change from 38/49/31 frames with Audio-EOS to 37/49/40 frames
  without it. Listening comparison is required before making Audio-EOS a
  production policy change.
- Audio-EOS-off still exhibits artifacts. This supports the conclusion that
  the boundary is not a PCM-gap or EOS-representation bug. The next test is a
  retained StaticCache versus full eager recomputation logit comparison at the
  first and second acoustic frames after an appended segment.
- Initial BF16-compatible seed-43 logit comparison: retained and full eager
  recomputation agree on argmax and all top-5 candidates for both examined
  post-boundary frames, but raw maximum absolute logit differences are 0.296
  and 0.432. Those differences can alter stochastic sampling near a cutoff,
  but this first comparison also changes StaticCache/graph attention versus
  full-attention kernels. An eager-StaticCache control is required before
  interpreting it as a continuation mask/cache defect.
- The eager-StaticCache control produces the same 0.296/0.432 differences and
  identical top-5/argmax agreement. CUDA-graph replay is therefore not the
  cause. The remaining distinction is retained StaticCache/masked attention
  versus full recomputation; this needs a same-attention-backend control before
  claiming a cache or mask error.
- The same-attention-backend control passed exactly: a fresh eager StaticCache
  rebuilt from the original prompt plus the captured segment-1 acoustic frames
  produces bit-identical first and second post-boundary logits to the live
  retained cache. This rules out a stale-KV, cache-position, RoPE-position, or
  causal-mask error in the tested continuation transition. The nonzero full
  recomputation difference is an attention/cache implementation numerical
  distinction, not an incorrect retained sequence.
- A five-seed BF16-compatible `repetition_penalty=1.0` control (seeds 42-46)
  is complete at `outputs/continuation-sweep/bf16-compat-rp1/`. It contains 15
  live contiguous boundary clips with zero frame-accounting failures. It is
  ready for matched listening comparison against the `repetition_penalty=1.1`
  baseline; no production policy change is implied until that comparison.
- **Listening result:** the RP 1.0 control retains the same audible artifacts.
  It changes some sampled trajectories (frame counts differ), but does not
  improve the artifact pattern. The per-segment repetition-history reset is
  therefore not the primary cause; do not change the production penalty on
  this evidence.
- Two five-seed controls are complete and ready for listening: original BF16
  weights with the retained-continuation runtime at
  `outputs/continuation-sweep/original-bf16/`, and one-request monolithic Sky
  targets at `outputs/monolithic-sky/bf16-compat-seeds-42-46/`. Both sets use
  seeds 42-46. The original-weight set has zero frame-accounting failures; the
  monolithic set removes all internal `audio → text → acoustic` transitions.
- **Listening result:** original BF16 retained continuation still produces
  boundary artifacts, though less extreme than the hybrid path. The five
  monolithic generations have no noticeable artifacts. Quantization therefore
  amplifies some failures but is not their root cause, and Breeze can generate
  this material cleanly when all target text is conditioned up front.

## Current continuation-layout diagnosis

- Insufficient retained history is ruled out for the tested transition. The
  live cache contains the complete original prompt and every generated frame;
  reconstructing that history in a fresh StaticCache yields bit-identical
  boundary logits.
- Monolithic conditioning is `reference/instruction/all target text → one
  acoustic span`. Retained continuation instead becomes `reference/instruction/
  text1 → audio1 → Audio-EOS → plain text2 → audio2`. That alternating layout
  is the material out-of-distribution difference.
- CFG semantics weaken at every append. Initial `ref_edit_tata` conditioning
  gives the positive branch `[S0]<ins_bos>instruction<ins_eos>text` and the
  negative branch `[S0]text`. `append_text()` currently encodes only plain
  `[S0]text` and repeats the same embeddings into both branches. The guidance
  difference for later segments therefore survives only in older retained KV,
  rather than being refreshed beside each new target.
- The next controls should distinguish short-utterance behavior from retained
  layout behavior: generate the four sentences as independent normal prompts,
  then test retained CFG 1 and refreshed per-segment CFG. If independent short
  prompts are clean but current retained CFG is not, the continuation append
  layout—not model convergence—is implicated.
- The first two controls are complete for seeds 42-46: retained CFG 1 at
  `outputs/continuation-sweep/bf16-compat-cfg1/`, and independent normal
  `ref_edit_tata` prompts (full reference and instruction for every sentence)
  at `outputs/continuation-sweep/independent-normal-prompts/`. Both have zero
  frame-accounting failures and are ready for matched listening.
- **Listening result:** independent normal prompts exhibit artifacts and
  inconsistent instruction following. They are not a viable chunking strategy.
  Retained continuation is materially more consistent and has the lowest
  artifact rate observed outside monolithic generation; the remaining standout
  defect is small and currently deferred. Do not add fade, crossfade, trimming,
  or other output masking: it would conceal a model-generated signal without
  improving retained conditioning. Revisit only if the defect becomes more
  noticeable in real agent traffic.

### CFG and text-encoder follow-up (2026-09-10)

The continuation CFG hypothesis is confirmed by direct code inspection.
`append_text()` calls `prepare_continuation_text_inputs()`, which renders one
plain `[S0]text` segment, encodes it once, and repeats the resulting embedding
for both CFG rows. For Voice Direction, the initial branches are correctly
different:

```text
conditional:   [S0]<ins_bos>instruction<ins_eos>target
unconditional: [S0]target
```

Every later append is instead:

```text
conditional:   [S0]next target
unconditional: [S0]next target
```

At CFG 4, the new-target guidance direction is consequently only a residual
in the older KV state. This is a concrete semantic mismatch with the normal
Voice Direction prompt, and should be evaluated as the next retained-
continuation architecture change.

The associated text-context observation is also correct. The model's
`convert_input_ids_to_embeds()` deliberately forwards every rendered text
segment as an independent text-encoder batch row. T5Gemma2's text self-
attention is bidirectional (with a bidirectional sliding window on its local
layers), so a monolithic target segment provides context across sentence
boundaries that continuation chunks cannot receive. The clean monolithic
control is therefore meaningful. A real-time service cannot use future agent
text without delaying output, but it may be possible to preserve *past* text
context in a later design; that needs a deliberate conditioning/cache design,
not a simple text prepend.

Do **not** patch this by just placing instruction tokens in the current
batch-two `append_text()` call. The conditional append is longer than the
unconditional one. The current direct-prefill path has a single physical
cache-position span and constructs an all-valid append mask, so naïve padding
would make dummy positions observable or misalign per-row logical positions.
The safe diagnostic implementation is split conditional/unconditional
backbone caches (or a redesigned per-row append mask and position scheme),
then stack their hidden states only where the CFG/depth path needs batch two.
That retains the existing model-weight VRAM footprint; its KV-cache footprint
is expected to remain approximately the existing two CFG rows, while it may
cost some graph-capture/launch efficiency.

Recommended next experiment: implement this split-cache **refreshed-CFG**
diagnostic with `Audio-EOS + instruction + next text` on the conditional row
and `Audio-EOS + next text` on the unconditional row; keep all sampler,
retained acoustic history, RNG, and codec state identical. Compare five
matched seeds against current retained CFG 4 before considering it for the
production path. This tests the concrete CFG mismatch independently of the
unavoidable lack of future text context.

### Refreshed-CFG experiment (2026-09-10)

The diagnostic was implemented without duplicating the two-row CFG KV cache.
`BackboneGraph` now tracks valid physical KV slots, allowing a shorter
unconditional append to be right-aligned beside the longer conditional
instruction-plus-text append. The direct append mask excludes those padding
holes and per-row RoPE positions advance only over actual tokens. Default
continuation does not use this path; it is enabled only by
`scripts/diagnose_continuation_boundaries.py --refreshed-cfg`.

Completed a matched hybrid `bf16_compat`, CFG 4, Audio-EOS, retained-state
sweep for Sky seeds 42-46 at:

```text
outputs/continuation-sweep/refreshed-cfg/seed-42/
.../seed-43/
.../seed-44/
.../seed-45/
.../seed-46/
```

The produced frame counts per four segments were respectively
`[29, 64, 44, 42]`, `[25, 47, 44, 46]`, `[30, 35, 36, 37]`,
`[54, 46, 50, 50]`, and `[45, 38, 35, 40]`. Every captured decodable frame was
decoded once and yielded exactly 1,920 PCM samples. Thus the experiment is
mechanically valid and ready for matched listening against the current CFG-4
retained-continuation sweep. Do not promote it yet: a changed sampling
trajectory is expected; artifact rate and instruction adherence require human
evaluation.

**Listening result:** all five refreshed-CFG samples still exhibit the
continuation-edge artifacts. Refreshing the explicit instruction contrast is
therefore not a sufficient fix and must not replace the current retained
continuation path. These runs retained the established `bf16_compat` hybrid
policy: INT8 ConvRot weights, FP32 scale storage populated with BF16-compatible
scale values, and exact-FP32 `lm_head` and depth codebook head. The failure is
not a regression caused by reverting the dtype policy.

The refreshed-CFG switch is not exposed through the API and is off by default.
A non-refreshed seed-42 regression run after the valid-slot cache support was
added produced byte-identical captured codec frames to the saved current
`bf16_compat` baseline for all four segments (`29`, `48`, `51`, and `41`
frames). The shared cache-mask support therefore does not alter the ordinary
continuation path in this matched control.

### CFG-scale interpretation (2026-09-10)

The clean CFG-1 retained control versus artifact-prone CFG-4 controls does not
by itself demonstrate an arithmetic CFG implementation bug. The CFG formula is
the standard `unconditional + scale * (conditional - unconditional)` and is
applied at both the backbone first-codebook logits and the depth-decoder
logits. Cache reconstruction has also cleared a stale-cache/position error.

Critically, scale 1 takes the runtime's `no_cfg` path: it runs only the
conditional Voice Direction row. Any scale other than 1 runs two rows and
guides their difference. Thus CFG 1 is not merely CFG 4 with a weaker
multiplier. At later continuation segments both rows append plain text, so
their remaining difference is retained conditioning history. Higher scales
amplify that history-derived difference and can plausibly amplify an otherwise
small unstable acoustic preference at a segment edge. The refreshed-CFG result
shows that re-appending the instruction alone does not remove the instability.

Next isolating control: sweep the current retained path at 1.0001, 1.25, 1.5,
2, 3, and 4 on matched seeds. Scale 1.0001 forces the two-row CFG path while
being numerically near conditional-only sampling; it separates a branch-count/
kernel-layout issue from a genuine guidance-scale sensitivity.

### Continuation CFG-ramp experiment (2026-09-11)

The matched seed-43 CFG-scale sweep found that the artifact is barely audible
at the lowest two-row CFG setting and becomes progressively more pronounced as
CFG increases. This supports guidance-scale amplification rather than a simple
binary cache or codec failure.

Implemented a diagnostic-only continuation ramp. It leaves the initial
segment at requested CFG, then for every appended segment applies the same
per-frame scale to both backbone and depth-decoder guidance. The first test is
target CFG 4 with `low_scale=1.0001`, a two-frame hold, then a four-frame
linear ramp to 4. At 1,920 samples/frame and 24 kHz, this affects only the
first 480 ms of each appended utterance; it neither inserts nor alters codec
frames or silence.

The seed-43 render is at
`outputs/continuation-sweep/cfg-ramp-seed-43/hold-2-ramp-4/`. It produced
`[25, 50, 58, 50]` decodable frames and passed exact frame/PCM accounting.
It is ready for listening against fixed-CFG-4 seed 43. Do not promote the
ramp to the API until artifact rate and instruction adherence are reviewed
over matched seeds.

### Combined refreshed-CFG plus ramp control (2026-09-11)

Listening on seed 43 found that both refreshed-CFG alone and the continuation
CFG-ramp alone improve on the fixed-CFG-4 retained control, which was the
worst of these matched samples. Generated the combined control with both
changes enabled: conditional instruction/text refresh plus CFG 1.0001 for two
frames and a four-frame ramp to target CFG 4. It is at
`outputs/continuation-sweep/refreshed-cfg-ramp-seed-43/hold-2-ramp-4/`.
The run produced `[25, 44, 43, 37]` decodable frames and passed exact
frame/PCM accounting. It is ready for listening; do not infer that the two
improvements are additive until this control and additional matched seeds are
reviewed.

### Terminal artifact follow-up (2026-09-11)

Listening to the combined control suggests that onset distortion may be
substantially reduced, while distortion near a segment's end remains. An onset
ramp cannot repair that: the terminal frames are sampled before the following
continuation append exists. Do not trim, fade, or overwrite the tail as a
first response.

Recommended terminal evaluation order:

1. Run the combined refreshed-CFG/ramp layout at target CFG 2 and CFG 3. The
   scale sweep already shows increasing artifacts with higher guidance, so a
   lower steady-state target may preserve the onset improvement and reduce
   tail instability without an endpoint heuristic.
2. Add a diagnostic log of the raw conditional, unconditional, and guided
   backbone EOS rank/probability for the last 8-12 acoustic steps of each
   segment. Align the reported audible defect with codec-frame indices. This
   distinguishes an EOS-adjacent sampling failure from a generic late-speech
   trajectory failure.
3. If the defect reliably begins when EOS becomes competitive, test an
   **EOS-proximity CFG taper**: after each generated frame, use the next raw
   logits to reduce CFG only when guided EOS enters a calibrated top-k/rank or
   probability threshold. Apply that scale to both the next backbone sample
   and depth-decoder sample. This changes no generated audio externally and
   leaves ordinary mid-utterance guidance at the requested target.
4. Only if the model has already sampled a bad tail before any EOS signal is
   detectable should we consider a cache-rebuild/re-sampling experiment from a
few frames before EOS. That is more invasive and needs an explicit quality
win to justify its latency cost.

EOS telemetry was added as an opt-in diagnostic and proved non-invasive: a
logged combined seed-43 run produced byte-identical codec frames to its
unlogged counterpart. It records conditional, unconditional, and guided EOS
rank/probability before each acoustic frame. In the unmodified combined seed
43, EOS became guided top-10 four frames before one segment terminated,
supporting an EOS-proximity control.

Generated the first terminal-taper control at
`outputs/continuation-sweep/refreshed-cfg-ramp-eos-taper-seed-43/rank-10-scale-1_0001/`.
It uses the combined onset treatment and persistently drops to CFG 1.0001 once
guided EOS rank is <=10. It produced `[25, 43, 48, 32]` frames and passed
frame/PCM accounting. In its changed trajectory, terminal taper activated at
frame 28 of segment 3 and frame 30 of segment 4; it did not activate in
segment 2. This may be too early for segment 3, so listening must judge both
tail cleanliness and whether the end becomes unnaturally weak or prolonged.

The rank-10 taper did not activate on the reported segment-2 tail artifact.
Its telemetry showed guided EOS at rank 13 five frames before termination, so
a matched rank-16 control was generated at
`outputs/continuation-sweep/refreshed-cfg-ramp-eos-taper-seed-43/rank-16-scale-1_0001/`.
It activated at frame 38 of segment 2 (six terminal low-CFG frames), frame 40
of segment 3, and frame 34 of segment 4; it produced `[25, 44, 45, 41]`
frames with exact frame/PCM accounting. This is the final inexpensive taper
control before considering cache-rebuilt tail re-sampling.

A matched seed-42 rank-16 control was generated at
`outputs/continuation-sweep/refreshed-cfg-ramp-eos-taper-seed-42/rank-16-scale-1_0001/`.
It produced `[29, 44, 41, 40]` frames with exact frame/PCM accounting. Terminal
taper activation began at frames 40, 39, and 37 of continuation segments 2,
3, and 4 respectively (4, 2, and 3 terminal low-CFG frames). This gives a
second seed for listening before considering production promotion or a more
costly re-sampled-tail design.

### Separating onset and terminal experiments (2026-09-11)

Terminal taper is not consistently predictive: changing the onset hold changes
the full autoregressive trajectory, including EOS rank and tail content. It
therefore confounds onset-hold comparisons and should not be used while
selecting an onset schedule. Generated onset-only, refreshed-CFG seed-42
controls with no terminal taper at:

```text
outputs/continuation-sweep/refreshed-cfg-onset-ramp-seed-42/hold-2-ramp-6/
outputs/continuation-sweep/refreshed-cfg-onset-ramp-seed-42/hold-2-ramp-8/
```

They produced `[29, 33, 34, 48]` and `[29, 47, 42, 41]` frames respectively,
with exact frame/PCM accounting. Evaluate onset quality independently before
continuing terminal work.

If tail artifacts remain material after choosing that schedule, prefer a
buffered cache-rebuild/re-sampled-tail diagnostic over further EOS rank tuning:
hold the final 4-6 uncommitted codec frames, once EOS is sampled rebuild the
verified-equivalent backbone cache before that tail, and generate a replacement
tail at a lower fixed CFG. Record baseline versus candidate total wall time,
cache rebuild time, tail-generation time, buffered-end latency, added/removed
frame count, peak allocated VRAM, and artifact/instruction ratings. It cannot
be used for already-emitted frames, so the production version must buffer that
tail before PCM emission; this gives a bounded 320-480 ms end-of-segment
commit delay rather than affecting TTFA.

### Operational decision (2026-09-14)

For production requests that require retained continuation, use CFG 1 as
operational guidance, not a hard API restriction. It selects the single
conditional branch and has shown substantially lower boundary-artifact risk in
this investigation. Callers that need stronger explicit Voice Direction may
still use higher CFG with the existing quality caveat. Do not expose the
experimental refreshed-CFG, onset-ramp, or EOS-taper controls in the API yet.

Cache-rebuilt buffered tail regeneration remains a possible future fix. It
would require a deliberate quality/latency evaluation with rebuild and tail
generation timings, buffered-end latency, emitted-frame accounting, GPU peak
memory, and blind listening outcomes before production consideration.

### Refreshed-CFG archival decision (2026-09-14)

Refreshed CFG remains semantically closer to the normal Voice Direction CFG
template: later conditional appends contain instruction plus target text while
later unconditional appends contain only target text. Matched listening,
however, found no observable reduction in generation errors compared with the
existing retained continuation path. Keep it diagnostic-only and API-disabled.

Revisit it only as part of a future CFG-greater-than-1 quality study. It has
no effect under the adopted CFG-1 continuation guidance. If revisited, account
for its cost: an additional text-encoder branch pass and a longer append
prefill/retained-context footprint, without an expected per-acoustic-frame
decode benefit.

## Recommended evaluation order

1. Run a 20-50 seed continuation sweep with `bf16_compat` and `exact_fp32`.
   Record first codec frames, segment-boundary clips, artifact labels,
   instruction adherence, duration, and GPU peak allocation.
   Use `scripts/diagnose_continuation_boundaries.py --hybrid-scale-mode ...`
   for one retained Sky trajectory, and `scripts/diagnose_startup_audio.py
   --hybrid-scale-mode ...` for an onset seed sweep.
2. Keep `bf16_compat` only if it wins the sweep, not merely the Sky seed-42
   sample. Preserve the exact mode as a diagnostic control.
3. Test audio.cpp's BF16 GGUF as an implementation baseline, then its Q8_0
   GGUF on the same *single-request* prompts. Do not treat its result as a
   retained-continuation test: its long-text chunking starts a fresh generator
   per text chunk.
4. Do not switch the API backend unless a native implementation can preserve
   retained backbone, RNG, and codec state across continuation segments.
   audio.cpp's codec lookahead may help stream emission artifacts but cannot
   repair already-generated bad codec frames.
5. Independently revise external silence pacing only after the model-frame
   evaluation. Silence is not fed into the model, but it is added to already
   variable model-generated pauses.

## Relevant external observations

The official Breeze implementation loads original weights in BF16. audio.cpp's
Breeze backend also deliberately uses BF16 activations and warns that FP32
activation paths can drift toward repetitions or mispronunciations. Its Q8
package is another lossy quantized autoregressive path, not a guarantee against
sampling divergence. See the upstream sources linked in the README and the
audio.cpp Breeze documentation:

- <https://github.com/breezeblue-ai/breeze-tts/blob/main/breeze_infer/runtime.py>
- <https://github.com/0xShug0/audio.cpp/blob/main/docs/models/breeze_tts.md>
- <https://github.com/0xShug0/audio.cpp/blob/main/src/models/breeze_tts/generator.cpp>
- <https://github.com/0xShug0/audio.cpp/blob/main/src/models/breeze_tts/session.cpp>
