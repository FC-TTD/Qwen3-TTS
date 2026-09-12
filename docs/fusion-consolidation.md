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

The final managed stack name is `qwen-tts-fusion-single`. The public/internal hostnames remain unchanged. Keeping a new internal stack name allows the verified candidate to become the final instance without rebuilding it again. Portainer2.33.3 cannot adopt the old unmanaged `qwen-tts-fusion` namespace by a same-name POST; it returns409 before deployment.

Build with the committed read-only source wrapper through the registered TTD launcher (`--exec ./deploy.sh --ci swarm`). Publish only the resulting content-addressed image, then use its registry digest. The template's `h-build-required` must never be deployed.

The project-owned `scripts/rollout_single_fusion.py` has three explicit phases:

1. `candidate --state <private-dir> --commit <build-commit> --image <registry-digest>` creates a Portainer-managed stack from the committed service definition. It initially declares only the temporary internal `qwen-fusion-candidate` validation hostname. Old fusion and Base remain untouched; all rollback specs, management files and environments are stored privately.
2. Run `scripts/smoke_fusion.py --url http://qwen-fusion-candidate --output <candidate-evidence> --gradio --phase candidate --expected-commit <build-commit> --expected-image <registry-digest>`. It exercises API/UI Clone, Design and CustomVoice, speed/expected duration, invalid inputs and unload. Evidence includes the actual backend deployment identity and execution time.
3. `promote --state <private-dir> --evidence <candidate-evidence>` publishes the five final aliases, removes discovery labels from the old unmanaged fusion via a version-checked Docker service update, and updates Base202 labels through Portainer. Runtime templates and old model processes remain available. Verify the effective Caddy upstream sets contain only the new fusion, not a mixture which happens to answer one health request.
4. Run the same complete smoke against `http://qwen-fusion`, with `--phase production` and a new evidence directory. Its start time must follow promotion and its commit/image must match the live deployment.
5. `retire --state <private-dir> --evidence <production-evidence>` drains old HTTP connections, rechecks each management/spec/namespace immediately before deletion, deletes the old external fusion stack and Base202 through Portainer, and verifies that only one managed fusion remains with all five aliases.

The new backend reports deployment identity in `/health/backends`. Before the candidate exists, rollback is the unchanged old deployment. After label cutover, restore saved old discovery labels and Base's complete file plus environment through their respective management paths if validation fails. Do not delete the old models until production smoke passes. After retirement, restore the old fixed image and reviewed source stack through Portainer if needed; preserved caches, LoRA and reference data remain shared. Never infer a successful adoption from409 or blindly retry a partially successful create.

The five aliases are `qwen-api`, `qwen-fusion`, `huolieniao`, `qianwen-api-design`, and `qianwen-api-custom`. No standalone Base is part of the final desired state. Image build metadata and the operation script commit are recorded separately when only rollout tooling changes after an immutable image build.

## Validation

`python -m pytest tests/test_fusion.py -q` covers all three modes, API/UI sharing,
reference validation, speed and duration with the real audio pipeline, queue
limits/timeouts, model switching, safe idle/manual unload, load failure recovery,
and responsiveness of the ASGI status route during inference. Model outputs in
these CPU tests are fakes; they do not replace real speech/GPU acceptance.

The root legacy integration tests call an already running HTTP service and are
not run against production while preparing this change. `scripts/smoke_fusion.py`
is the explicit real-service acceptance entrypoint.
