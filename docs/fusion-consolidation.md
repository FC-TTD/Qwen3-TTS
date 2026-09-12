# Self-contained Qwen Fusion

The sole desired inference service is `qwen-tts-fusion-single_fusion`, worker GPU UUID
`GPU-aee06b60-5da4-ae60-2775-88f094ffeab7` (GPU2 at planning time). It exposes
Clone, VoiceDesign and CustomVoice through FastAPI and Gradio, with one shared
GPU queue and at most one loaded model. The old two-service Base stack definition
has been removed; redeploying this source cannot recreate it.

## Runtime contract

- API: `POST /api/tts`; same multipart fields as the prior fusion, including
  reference audio/text, sampling controls, speed, expected duration, silence
  trimming, loudness and checkpoint selection. All three modes now use those
  audio controls and return 48kHz WAV. Failed requested audio processing reports
  an error instead of silently returning unmodified audio.
- Clone is local. There is no `BASE_API_URL`, outbound Base request or hidden
  second inference process. Design remains 1.7B; Clone defaults to 1.7B and
  CustomVoice keeps 0.6B/1.7B plus `/app/lora` checkpoints.
- API and Gradio serialize model loading, inference, switching, and unloading.
  Eight requests may wait behind one active request; overload returns 429 and
  a 300-second queue wait returns 503. These are tunable via `FUSION_MAX_PENDING`
  and `FUSION_QUEUE_TIMEOUT_SECONDS`. Queue status is `/health/backends`.
- CUDA readiness uses ttd-fastapi-utils. Health remains responsive during model
  inference. Idle weights are unloaded after 7200 seconds; `POST /api/unload`
  waits behind any active inference and releases the current model safely.
- Only one model remains loaded. Mixed-mode requests trade cold loading latency
  for GPU memory. No throughput or peak-memory guarantee is claimed until the
  real GPU acceptance run.
- Fine-tuning and automatic ASR stay disabled, as on the prior fusion. The three
  inference buttons all use the shared runtime; the old Design callback bypass
  is corrected.

## Deployment and cutover

1. Record rollback from the prior committed project source and live resolved
   image digests; preserve HF cache, LoRA and the shared voxcpm-named data mount.
   Do not generate desired state by copying inspected containers.
2. Commit the reviewed participating inputs. Use the registered TTD launcher
   with `--exec ./deploy.sh --ci swarm`. The project wrapper creates a detached,
   read-only clone of the recorded commit and verifies it remains unchanged.
   `docker_ci_cd` builds on nest and publishes only an immutable `h-*` image;
   `h-build-required` in the template is intentionally nondeployable and must
   be replaced by the build's resolved identity. Never publish/deploy `latest`.
3. Existing `qwen-tts-fusion` is a Swarm stack without Portainer ownership.
   Create the final managed stack `qwen-tts-fusion-single` on endpoint4 using
   a validation rendering of this source with only `qwen-fusion-candidate`
   labels. This avoids same-name HTTP409 without deleting the legacy service.
   Keep the Base stack (Portainer id202) during fusion preparation.
4. Before route ownership cutover, deploy the new image to an isolated smoke
   service using the same committed service definition with temporary identity
   and no production labels; or use a coordinated fusion maintenance window.
   Verify the new service's three modes and UI without depending on Base.
5. Switch `qwen-api` and the existing fusion aliases to the verified local
   runtime. Caddy source labels must have one final owner. The existing Base
   labels may temporarily produce a mixed clone upstream, so verify effective
   config after removing them; do not claim completion from labels alone.
6. Drain old Base requests, then remove old Base services and Portainer stack202.
   Final desired/management/runtime state: one managed `qwen-tts-fusion-single` stack,
   one replica, no `qwen-tts_base-*`, correct GPU UUID, five preserved hostnames.
7. Run `scripts/smoke_fusion.py --url http://qwen-fusion --output /tmp/qwen-smoke
   --gradio` (one command; line break shown only for readability), verify saved
   WAVs, slow/fast ordering and expected duration. Capture GPU memory before
   loading, at peak and after `/api/unload`, with PID ownership and CUDA trace.
   Also test `qwen-api` via the real consuming application/model gateway.

`QWEN_FUSION_IMAGE=registry.ttd/qwen-tts-fusion/fusion:h-<CI-hash> ./deploy.sh --cd swarm`
requires an explicit immutable image from that same committed snapshot's CI result.
The wrapper rejects missing or mutable image references. The wrapper does not automatically remove the old Base stack or
silently adopt unmanaged services. These are explicit cutover actions after
successful smoke. Existing worker GPU0 Base is retained until this point.

## Validation

`python -m pytest tests/test_fusion.py -q` covers all three modes, API/UI sharing,
reference validation, speed and duration with the real audio pipeline, queue
limits/timeouts, model switching, safe idle/manual unload, load failure recovery,
and responsiveness of the ASGI status route during inference. Model outputs in
these CPU tests are fakes; they do not replace real speech/GPU acceptance.

The root legacy integration tests call an already running HTTP service and are
not run against production while preparing this change. `scripts/smoke_fusion.py`
is the explicit real-service acceptance entrypoint.
