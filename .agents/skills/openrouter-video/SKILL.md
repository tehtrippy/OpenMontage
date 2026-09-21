---
name: openrouter-video
description: |
  Generate video through OpenRouter's unified async video API. Use when: (1) One API key
  should cover several upstream video models (Veo, Seedance, Wan, Hailuo, Grok), (2) A cheap
  first-pass test generation is wanted before committing to a premium provider, (3) Switching
  model should be a one-string change rather than a new integration. Backed by the
  `openrouter_video` tool.
metadata:
  openclaw:
    requires:
      env_any:
        - OPENROUTER_API_KEY
---

# OpenRouter Video Generation

One key, one schema, many upstream models. Submit a job, poll it, download the MP4.

## Authentication

```
Authorization: Bearer $OPENROUTER_API_KEY
```

A single `OPENROUTER_API_KEY` covers every model below. No per-provider keys.

## The three calls

| Step | Call |
|---|---|
| Submit | `POST https://openrouter.ai/api/v1/videos` → `202 {id, polling_url, status:"pending"}` |
| Poll | `GET https://openrouter.ai/api/v1/videos/{id}` → `status: pending \| in_progress \| completed \| failed` |
| Download | `GET` the `unsigned_urls[0]`, or `/api/v1/videos/{id}/content?index=0` |

`GET /api/v1/videos/models` lists what is currently available.

**The download URLs are unsigned.** They still require the `Authorization` header — a plain
unauthenticated fetch returns an error. This is the one place OpenRouter differs from the
providers that hand back a public CDN link.

## Request body

Required: `model`, `prompt`.

Optional: `duration` (integer seconds) · `resolution` (**tier enum only**: `360p`, `480p`,
`720p`, `768p`, `1080p`, `1K`, `2K`, `4K`) ·
`aspect_ratio` · `size` · `frame_images` (first/last frame conditioning) ·
`input_references` (style/identity guidance) · `generate_audio` · `seed` ·
`callback_url` (HTTPS webhook) · `provider` (provider-specific passthrough).

**`resolution` is a tier, never a pixel size.** `"1080x1920"` comes back
`400 ZodError` naming the eight tiers above — measured 2026-09-21, when it was the
adapter's own default and no default-parameter call could generate. The vertical
delivery frame comes from `1080p` + `aspect_ratio: "9:16"`, which produces exactly
1080x1920. `openrouter_video` normalises a `WIDTHxHEIGHT` to the tier of its short
side (`1080x1920` and `1920x1080` both → `1080p`, `1440x2560` → `2K`) and refuses a
size that maps to no tier *before* submitting, so callers may still pass either form.

**There is no `negative_prompt` field.** Fold negatives into the prompt text, or pass them
through the `provider` object if the upstream model accepts them. A prompt written for Veo's
native API with a separate negative block has to be rewritten before it is sent here.

## Models and rates

Per-second "from" prices — the base tier, meaning lowest resolution with audio off. Higher
resolution or audio-on costs more, and the page does not publish the full matrix. Treat these
as a floor.

**Measured exception:** a real 4s / 720p / 9:16 / audio-off job on `google/veo-3.1-lite` billed **$0.12, i.e. $0.03/s** — below its listed "from" rate. The same job at **1080p billed $0.20, i.e. $0.05/s**, so resolution moves the charge and `estimate_cost` (per-model only) under-quotes the 1080p default. Treat the published table as indicative only and read `usage.cost`.

| Model id | From | Durations | Max resolution |
|---|---|---|---|
| `bytedance/seedance-2.0-mini` | $0.03363/s | — | 720p |
| `bytedance/seedance-2.0-fast` | $0.04035/s | 4–15s, any integer | 720p |
| `alibaba/wan-3.0` | $0.0425/s | — | — |
| `google/veo-3.1-lite` | **$0.03/s measured** (listed $0.05/s) | 4, 6, 8s | 1080p |
| `minimax/hailuo-3-max` | $0.05/s | — | — |
| `x-ai/grok-imagine-video` | $0.05/s | — | — |
| `bytedance/seedance-2.0` | $0.06726/s | — | — |
| `x-ai/grok-imagine-video-1.5` | $0.08/s | — | — |
| `google/veo-3.1-fast` | $0.10/s | 4, 6, 8s | 4K (2160×3840) |
| `bytedance/seedance-2.5` | $0.1028/s | — | — |

Blank cells are not published on the model page — check `GET /api/v1/videos/models` rather
than assuming.

## Frame conditioning (cross-clip continuity)

The accepted wire shape, established against the live endpoint — its ZodError
named all three fields:

```json
{"type": "image_url", "image_url": {"url": "<https or data uri>"}, "frame_type": "first_frame"}
```

The intuitive form is **rejected with a 400**:

```json
{"type": "first_frame", "image_url": "<string>"}   // WRONG
```

`type` must be the literal `"image_url"`, `image_url` must be an **object**, and
the first/last selector lives in `frame_type` (`first_frame` | `last_frame`).
Base64 data URIs are accepted. Failed submits are not charged.

`openrouter_video` normalises bare strings, local paths and the legacy form into
this shape, so callers can pass `first_frame: "<path>"` or `last_frame: "<path>"`
and forget the detail. Both ergonomic inputs land in the same `frame_images`
list with the same guards; the only difference on the wire is `frame_type`.
Per-model support is published as `supported_frame_images` (see below): most of
the Veo, Seedance, Kling and Hailuo families advertise both ends, while
`alibaba/wan-3.0`, `runway/gen-4.5` and the Grok models advertise `first_frame`
only.
Chaining act N's final frame into act N+1 is the mechanism that holds object
identity across clips — see `skills/creative/causal-video-production.md`.

## Frame conditioning semantics (measured, not assumed)

`supported_frame_images` lists which frame slots a model exposes. It does **not**
say whether they work independently. Measured on `google/veo-3.1-lite`,
2026-09-21:

| Sent | Result |
|---|---|
| `first_frame` alone | works - ordinary image-to-video conditioning |
| `first_frame` + `last_frame` | an **interpolation** job between two known states |
| `last_frame` alone | **rejected**: "Frame interpolation requires both an input image and a last frame." (job created, then failed; **not billed**) |

So for this model:

- **`first_frame` alone is the supported conditioning mode**, and the one the
  causal chain uses.
- **`last_frame` is not a standalone identity or reference mechanism.** It is
  one endpoint of an interpolation, and it must never be described as an
  object-identity primitive.
- `openrouter_video` refuses `last_frame` without `first_frame` locally, before
  submitting, for models whose semantics have been observed
  (`OpenRouterVideo.frame_semantics(model) == "interpolation"`). Models whose
  semantics have **not** been observed are passed through untouched - an
  untested model does not inherit another model rule.

Read capabilities with `OpenRouterVideo.fetch_model_capabilities(model)`
(metadata only, never billed). Nothing calls it automatically; payload building
stays offline.

## Identity references (`input_references`)

Same normalised shape, minus `frame_type` -- a reference is not a position in
time:

```json
{"type": "image_url", "image_url": {"url": "<https or data uri>"}}
```

Strings, local paths, https URLs and data URIs are all accepted and rewritten.
Local paths get the **same guards as frame conditioning**: confined to the
working tree, validated by image magic bytes rather than extension, size-capped,
then base64-inlined. They used to be forwarded verbatim, which both failed (the
provider cannot read your disk) and bypassed those guards.

**Use it for what frame chaining cannot carry.** The chain transports only what
is visible in the handoff frame. On 2026-09-21 a muddler head was submerged in
pulp at every handoff and three chained acts produced three different tools --
toothed crown, flat puck, perforated barrel -- while glass, board, hands and
camera held perfectly.

**Check the model before relying on it.** The unified request schema accepts
`input_references` for every model; the upstream model may ignore it. There is a
free metadata endpoint:

```
GET /api/v1/videos/models
```

Each entry carries `supported_resolutions`, `supported_aspect_ratios`,
`supported_sizes`, `supported_durations`, `supported_frame_images`,
`generate_audio`, `seed`, `pricing_skus` and `allowed_passthrough_parameters`.
**No model in the catalog advertises a reference-image capability** -- the only
image-input field exposed is `supported_frame_images`. For
`google/veo-3.1-lite` that is `["first_frame", "last_frame"]`. Treat reference
support as unverified for every model until a real generation demonstrates it.

Two more things that endpoint settles, free:

- **Pricing is published per SKU.** `google/veo-3.1-lite`:
  `duration_seconds_without_audio: 0.05`, `..._720p: 0.03`,
  `duration_seconds_with_audio: 0.08`, `..._with_audio_720p: 0.05`. This matches
  the measured charges exactly ($0.20 for 4s at 1080p, $0.12 at 720p) and is a
  better source than the adapter's per-model rate table.
- **`negativePrompt` is an allowed passthrough** for `google/veo-3.1-lite`
  (`allowed_passthrough_parameters`: personGeneration, aspectRatio,
  negativePrompt, conditioningScale, enhancePrompt). The adapter currently folds
  negatives into the prompt text because the *unified* body has no such field --
  the `provider` passthrough may be the better route for this model.

## Cost reporting

The poll response carries `usage.cost` — the **actual** charge, not an estimate. Prefer it over
any local price table. `openrouter_video` reports it as `ToolResult.cost_usd` and falls back to
the static table only when the field is absent.

## Choosing a model

- **Cheapest usable test pass** → `bytedance/seedance-2.0-mini`, then `seedance-2.0-fast`.
- **Long single takes** → `bytedance/seedance-2.0-fast` is the only one documented at 4–15s in
  arbitrary integer steps. Everything in the Veo family is 4/6/8 only. A 15s piece can be one
  generation instead of four, which removes cross-clip continuity drift entirely — at the cost
  of handing shot rhythm to the model.
- **Physical realism, product and liquid work** → the Veo family.
- **Native resolution above 1080p** → `google/veo-3.1-fast` reaches 2160×3840.

## Prompting

OpenRouter does not transform the prompt; it forwards to the upstream model. Use the
model-family guide — `ai-video-gen` for the fleet-wide vocabulary, the Veo or Seedance Layer 3
skills for family-specific structure — and write to the 5-aspect skeleton in
`skills/creative/video-gen-prompting.md`. The only OpenRouter-specific rule is the missing
`negative_prompt` field.

## Failure modes

| Symptom | Cause |
|---|---|
| 401 on download but 200 on poll | Auth header omitted from the content request |
| Job stuck `in_progress` | Normal — video jobs run minutes; poll every ~10s |
| `status: failed` with no output | Read `error` on the poll body; often a content filter or an unsupported parameter for that upstream model |
| Parameter silently ignored | The unified schema accepts it but the upstream model does not support it — check the model page |
