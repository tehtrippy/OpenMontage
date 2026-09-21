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

Optional: `duration` (integer seconds) · `resolution` (`720p`, `1080p`, `4K`, or `WIDTHxHEIGHT`) ·
`aspect_ratio` · `size` · `frame_images` (first/last frame conditioning) ·
`input_references` (style/identity guidance) · `generate_audio` · `seed` ·
`callback_url` (HTTPS webhook) · `provider` (provider-specific passthrough).

**There is no `negative_prompt` field.** Fold negatives into the prompt text, or pass them
through the `provider` object if the upstream model accepts them. A prompt written for Veo's
native API with a separate negative block has to be rewritten before it is sent here.

## Models and rates

Per-second "from" prices — the base tier, meaning lowest resolution with audio off. Higher
resolution or audio-on costs more, and the page does not publish the full matrix. Treat these
as a floor.

**Measured exception:** a real 4s / 720p / 9:16 / audio-off job on `google/veo-3.1-lite` billed **$0.12, i.e. $0.03/s** — below its listed "from" rate. Treat the published table as indicative only and read `usage.cost`.

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
this shape, so callers can pass `first_frame: "<path>"` and forget the detail.
Chaining act N's final frame into act N+1 is the mechanism that holds object
identity across clips — see `skills/creative/causal-video-production.md`.

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
