# OpenAI reconstruction VLM

Reconstruction selects `vlm_backend: openai` in `scripts/cfg/real2sim_cfg.yaml`.
Text/vision calls use `openai.vision_model=gpt-5.6-luna` and
`openai.reasoning_effort=high`. Image editing uses the separate
`openai.image_model=gpt-image-2` setting, selected after a successful paid preflight. Image generation availability must be
checked separately; a text response does not prove that image editing works.

Hydra overrides:

```bash
vlm_backend=openai openai.vision_model=gpt-5.6-luna openai.reasoning_effort=high
openai.image_detail=high openai.timeout_ms=180000 openai.max_output_tokens=16384
openai.image_model=gpt-image-2
```

For a non-reasoning model, set `openai.reasoning_effort=null`. Sampling/seed
arguments inherited from Gemini are omitted for reasoning models. Gemini model
fields remain as fallback values; switch back with `vlm_backend=gemini`.
Stage 5/6 image models set to `flux` remain local. Explicit `gpt` image selection
also works. `OPENAI_VISION_MODEL` and `OPENAI_IMAGE_MODEL` override model IDs.

## Credentials and the PC relay

Direct API use needs `OPENAI_API_KEY` in the environment or ignored
`api_keys.txt`. `OPENAI_BASE_URL` optionally selects an OpenAI-compatible provider.
A ChatGPT subscription and a separately billed API credential are different.
The key in this PC's Codex configuration currently uses its configured provider;
provider model availability, billing and quotas apply.

On the PC, the user services `pc-openai-relay.service` and
`ebcloud-vpn-tunnel.service` read `/home/xiang/.codex/auth.json` and
`config.toml` locally. The reverse SSH tunnel exposes the relay only at EBCloud's
`127.0.0.1:18973`. EBCloud does not hold the actual provider key.

On EBCloud use `/usr/local/bin/pc-openai COMMAND ...`. It sets:

```bash
OPENAI_API_KEY=pc-relay                       # non-secret placeholder
OPENAI_BASE_URL=http://127.0.0.1:18973/v1
NO_PROXY=127.0.0.1,localhost
```

The PC must remain online with its VPN proxy running. Data files, weights,
caches and generated outputs remain on EBCloud. Request images pass through
the relay in memory to reach the provider; no dataset is synchronized to the PC.
For direct provider access through the PC HTTP proxy instead, use
`HTTP_PROXY=http://127.0.0.1:18971` and `HTTPS_PROXY` with the same value.

## Preflight

The following opt-in command makes a billable vision request using one frame.
Add `--image-edit` to make one additional image-edit request. Each call has one
attempt and a bounded timeout. No API key or request headers are logged.

```bash
. /root/data/SOA/simfoundry-env.sh
cd /root/data/SOA/simfoundry
/usr/local/bin/pc-openai /root/data/SOA/envs/simfoundry/bin/python \
  scripts/openai_preflight.py --frame /absolute/path/frame.png \
  --allow-paid-openai-test
```

Sanitized manifests and responses are written to
`/root/data/SOA/runs/openai_preflight/`. Unit tests use fake clients and require
neither credentials nor network. `CACHE_MODE=1` stores provider-separated
responses; `TEST_MODE=1` replays matching responses without constructing a client.
Incomplete or empty Responses results are rejected and not cached.
Image edits accept validated base64 image data or an HTTPS image URL returned
by a compatible provider. URL downloads use the existing SDK without forwarding
API authorization or organization/project headers to the image host. Missing or
corrupt image payloads are rejected before caching.

Run reconstruction through `pc-openai` and pass the same Hydra overrides. Check
the command selection with the runner's `--dry-run` first. Do not start a full
run until vision/image availability and a GPU stage smoke test have passed.
HTTP 401/403/429 fail immediately; transient errors have at most three attempts.
API region support, quotas, and model availability depend on the provider.
Outputs can differ from Gemini; inspect reconstructed alignment before research use.

`/usr/local/bin/run-simfoundry-openai` sources the EBCloud environment, selects
the relay, data root and installed mesh environment, then forwards all arguments
to the normal reconstruction runner. It also routes HTTP(S) downloads through
the PC VPN at `127.0.0.1:18971` (`PC_VPN_PROXY` overrides it), bypasses the
proxy for the local relay, and allows 120 seconds for checkpoint download reads.
For example, inspect the command plan with:

```bash
run-simfoundry-openai --scene-name fruits_preparation_smoke --dry-run
```

Provide `--video-fpath` for a new source video and omit `--dry-run` to execute.
All normal runner options and Hydra overrides after `--` remain available.

The prepared fruit scene runs sequentially on the 24 GB RTX 4090 D:

```bash
OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES \
run-simfoundry-openai --scene-name fruits_preparation_smoke \
  --no-stream --skip-successful --cache-mode \
  --model-cache-dir /root/data/SOA/cache/simfoundry/model_calls \
  -- low_vram=true s7_mesh.low_vram=true visualize=false
```

The same command resumes completed stages and cached API calls. Keep the PC
relay and VPN tunnel running throughout. The text and Prior Depth Anything
checkpoints download into the EBCloud Hugging Face cache on first use.
The background-removal model is cached in `/root/data/SOA/models/u2net/`.
Headless Isaac Sim also requires `libglu1-mesa`; this package and its
`libopengl0` dependency are now installed on EBCloud. The first successful
RTX startup compiles shaders and can take several minutes.

## Load the generated scene in Isaac Sim

The completed stage-14 scene can be loaded without running reconstruction
again. On EBCloud:

```bash
# Headless smoke run: load the seven generated objects and Franka, then step physics.
/root/data/SOA/run_reconstructed_scene_isaac.sh --headless --steps 20

# Save a short viewer-frame video when a viewer camera is available.
/root/data/SOA/run_reconstructed_scene_isaac.sh \
  --headless --steps 60 \
  --video /root/data/SOA/data/fruits_preparation_smoke/s14_og/isaac_preview.mp4

# Record a full camera orbit around the generated objects without stepping physics.
/root/data/SOA/run_reconstructed_scene_isaac.sh \
  --headless --steps 0 --frames 72 --fps 12 --hide-robot \
  --orbit-video /root/data/SOA/data/fruits_preparation_smoke/s14_og/isaac_orbit.mp4

# Keep the Isaac/OmniGibson scene alive for interactive use; Ctrl-C exits.
/root/data/SOA/run_reconstructed_scene_isaac.sh --interactive
```

The launcher reads
`/root/data/SOA/data/fruits_preparation_smoke/s14_og/reconstructed_og_scene.json`.
Use `--scene-json` to load another saved scene. `--random-actions` is optional;
the default uses zero robot actions while physics steps.
The orbit centers on the generated objects' combined bounds. `--orbit-radius`
and `--orbit-height` control camera placement in metres (defaults: 1.45 and
1.15). `--frames` divided by `--fps` gives the clip duration.
`--hide-robot` temporarily hides the robot while recording so it cannot
occlude the generated objects. Its visibility is restored after capture.

Stage 9's optional `deps/articulate-anything` generation and the separate
augmentation pipeline still have their own Gemini configuration. This adapter
does not change vendored dependencies. The default reconstruction omits the
optional articulation generation unless `--detect-articulation` is selected.

## Current validation

On 2026-10-10, the PC relay returned a real `gpt-5.6-luna` text response and
the EBCloud SDK preflight described a prepared fruit-scene frame successfully
(HTTP 200, high reasoning). The `gpt-image-1` edit request returned HTTP 404.
That configured endpoint has not passed the image-edit capability check.
The provider advertised `gpt-image-2`; a separate one-attempt edit succeeded
(HTTP 200, 1024x1024), so the reconstruction config now selects that model.
The old GPT image class still defaults to `gpt-image-1` for compatibility, and
that model can be selected on providers that support it. For GPT Image 2,
`input_fidelity` is omitted as required by its API.
The preflight manifests are
`/root/data/SOA/runs/openai_preflight/20261010T093138Z/manifest.json` (vision)
and `/root/data/SOA/runs/openai_preflight/20261010T093556Z/manifest.json` (image).
Full reconstruction of `fruits_preparation_smoke` started on 2026-10-10.
Depth reconstruction, GPT canonical-frame selection, world alignment, seven-object
decomposition, image upsampling, mesh generation, pose matching, compilation,
simulation asset preparation and PyBullet stabilization passed. All seven shape
and texture mesh pairs have finite vertices and nonzero extents. All URDF mesh
references exist; stabilized positions are finite and rotations normalized.
USD conversion and final OmniGibson scene creation also passed after installing
the missing GLU dependency. All 13 required stages have success markers, and the
last resumed runner exited 0 at 2026-10-10 19:24:41 Asia/Shanghai.
The saved scene contains seven DatasetObjects and one FrankaPanda robot. All
seven USD assets open with geometry and resolved texture references; their
asset-pipeline material passes report `bound`. The final settled poses are
finite with normalized rotations. The 1280x720 preview is nonblank (RGB standard
deviation 26.18, 20,366 distinct colors).
Artifacts remain on EBCloud:

- Scene: `/root/data/SOA/data/fruits_preparation_smoke/s14_og/reconstructed_og_scene.json`
- Preview: `/root/data/SOA/data/fruits_preparation_smoke/s14_og/reconstructed_scene.png`
- Validated run manifest: `/root/data/SOA/runs/openai_reconstruction/fruits_preparation_smoke/manifest.json`
- Runner inventory: `/root/data/SOA/data/fruits_preparation_smoke/pipeline_manifest.json`

The full log is
`/root/data/SOA/runs/openai_reconstruction/fruits_preparation_smoke/pipeline.log`.
The mesh process exited with code 139 after writing all seven completed jobs
and its success marker. Its artifacts were validated, then reconstruction
resumed with `--skip-successful`. This native cleanup failure remains a known
limitation for an unattended first run.
The first USD import also crashed while `libGLU.so.1` was missing. Installing
`libglu1-mesa` and `libopengl0` resolved the startup/material failure; subsequent
imports completed and shut down successfully.

This proves reconstruction execution through stage 14 and API transport; the
render has not received a visual accuracy review. Review alignment, inferred
metric scale, masses, friction and object relationships before research use.
For example, the current inferred apple and banana masses are 1.47 kg and 2 kg;
the validation manifest records all masses. Teleoperation, policy training and
evaluation were not run.
The GPU depth smoke passed on the RTX 4090 D after a cached retry: finite
126x224 depth output, 8.16 GB peak allocated VRAM, 15.35 seconds including
model load. The first cold-load attempt hit the five-minute timeout.
Its manifest is `/root/data/SOA/runs/openai_migration/gpu_depth_smoke/manifest.json`.

Validation: 88 focused tests passed and one existing test was skipped across
the OpenAI adapter, Gemini result handling, remote caches/retries, configured
model names, frame selection and front canonicalization. Compilation and
`git diff --check` passed. The reconstruction runner and the installed launcher
completed CPU-only dry-runs.
After the image-payload validation and URL-download fix, 21 focused adapter/cache
tests passed. An SDK transport check also confirmed that image downloads omit
authorization and organization/project headers. These checks use fake clients
and do not make additional paid calls.

Changed source/config files:

- `simfoundry/models/vlm.py` adds OpenAI vision, provider resolution, bounded
  sanitized errors and configurable image editing.
- `simfoundry/pipeline/frame_selection.py` and `front_canonicalization.py`
  use the provider resolver.
- `scripts/cfg/real2sim_cfg.yaml` and `auto_bg.yaml` select the tested models.
- Reconstruction stages 3, 5, 6, 8, 8b, 9 and 11, plus optional background
  quadmask proposal, route their VLM calls through the configuration.
- `scripts/openai_preflight.py` and `tests/test_openai_vlm.py` provide live
  opt-in checks and credential-free unit tests.
- `docs/OPENAI_VLM.md` records usage and validation.

PC relay/tunnel sources and EBCloud launcher helpers are under
`/media/xiang/Xiang/sim_foundry/ops/ebcloud_pc_vpn/` on the PC.
Original EBCloud source backups and the tracked migration patch are under
`/root/data/SOA/runs/openai_migration/`. Existing installation/model-loader
changes were preserved.
