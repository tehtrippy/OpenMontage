"""OpenRouter video generation — one key, many upstream video models.

OpenRouter fronts Veo, Seedance, Wan, Hailuo and Grok behind a single
asynchronous API: submit a job, poll it, download the MP4. Switching model is a
one-string change, which is the whole reason this adapter exists — it replaces
several provider-specific keys with OPENROUTER_API_KEY.

Discovery is automatic: this class declares capability="video_generation", so
`video_selector` picks it up from the registry with no changes to the selector.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

_API_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "google/veo-3.1-lite"

# Vertical social delivery target. 2K is opt-in, never the default — the H3 A/B
# experiment showed 2K triples the bill for pixels a 1080p timeline throws away.
_DEFAULT_RESOLUTION = "1080x1920"
_DEFAULT_ASPECT_RATIO = "9:16"
_DEFAULT_ACT_SECONDS = 5

# Per-second fallback rates, used ONLY when a real charge is unavailable — i.e.
# by estimate_cost() before a job runs, and by the scoring engine. The
# authoritative charge is `usage.cost` on the poll response, which execute()
# always prefers.
#
# Unmarked entries are OpenRouter's published "from" prices (base tier: lowest
# resolution, audio off) and are a floor, not a quote. Entries marked VERIFIED
# were measured against a real completed job at the stated configuration.
_RATE_USD_PER_SECOND: dict[str, float] = {
    # VERIFIED 2026-09-21: 4s @ 720p, 9:16, audio off billed $0.12 => $0.03/s.
    # The collection page lists this model "from $0.05/s"; that rate did not
    # match the actual charge for this configuration. Higher resolutions or
    # audio-on may still bill above this.
    "google/veo-3.1-lite": 0.03,
    "google/veo-3.1-fast": 0.10,
    "bytedance/seedance-2.0-mini": 0.03363,
    "bytedance/seedance-2.0-fast": 0.04035,
    "bytedance/seedance-2.0": 0.06726,
    "bytedance/seedance-2.5": 0.1028,
    "alibaba/wan-3.0": 0.0425,
    "minimax/hailuo-3-max": 0.05,
    "x-ai/grok-imagine-video": 0.05,
    "x-ai/grok-imagine-video-1.5": 0.08,
}
_FALLBACK_RATE_USD_PER_SECOND = 0.10

# OpenRouter's request body has no negative_prompt field. Rather than drop the
# negatives — or gamble on the undocumented `provider` passthrough — they are
# appended to the prompt text behind this marker. The marker is also what makes
# the append idempotent: a prompt that already carries its negatives is left
# alone, so re-assembling the same act twice cannot double them up.
_NEGATIVE_MARKER = "Negative constraints:"

_TERMINAL_OK = "completed"
_TERMINAL_FAIL = "failed"

_POLL_INTERVAL_SECONDS = 10
_POLL_TIMEOUT_SECONDS = 600

# Frame conditioning reads a LOCAL file and ships its bytes to a third party.
# Left unconstrained that is a read-anything-and-exfiltrate primitive, and the
# path is agent-authored, so a manipulated turn could point it at a key or a
# .env. Two guards: the file must live under the working tree, and it must
# actually be an image by magic bytes — an extension check would not stop a
# secret renamed to .jpg.
_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"BM", b"RIFF")
_MAX_FRAME_BYTES = 12 * 1024 * 1024


class OpenRouterVideo(BaseTool):
    name = "openrouter_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "openrouter"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:OPENROUTER_API_KEY"]
    install_instructions = (
        "Set OPENROUTER_API_KEY to an OpenRouter API key.\n"
        "  Get one at https://openrouter.ai/keys\n"
        "  One key covers every model in the video collection "
        "(Veo, Seedance, Wan, Hailuo, Grok)."
    )
    agent_skills = ["openrouter-video", "ai-video-gen"]

    capabilities = ["text_to_video", "image_to_video", "reference_to_video"]
    supports = {
        "text_to_video": True,
        "image_to_video": True,
        "first_last_frame": True,
        "reference_image": True,
        "multiple_reference_images": True,
        "native_audio": True,
        "seed": True,
        "vertical_9_16": True,
        # OpenRouter's documented request body has no negative_prompt field.
        # Left False on purpose: this flag means "the API takes a native
        # negative_prompt", and it does not. The tool still honours a
        # negative_prompt input by folding it into the prompt text — see
        # negative_prompt_via_prompt_text below.
        "negative_prompt": False,
        "negative_prompt_via_prompt_text": True,
    }

    best_for = [
        "reaching several video models through a single API key",
        "cheap first-pass test generations",
        "switching model without switching provider integration",
        "cost-sensitive short-form vertical clips",
    ]
    not_good_for = [
        "prompts that depend on a dedicated negative_prompt field",
        "provider-specific features not exposed by OpenRouter's unified schema",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Text description of the video"},
            "model": {
                "type": "string",
                "default": _DEFAULT_MODEL,
                "description": (
                    "OpenRouter video model id, e.g. google/veo-3.1-lite, "
                    "bytedance/seedance-2.0-fast, alibaba/wan-3.0"
                ),
            },
            "duration": {"type": "integer", "default": _DEFAULT_ACT_SECONDS,
                         "description": "Clip length in seconds. 5s is one causal act."},
            "resolution": {
                "type": "string",
                "default": _DEFAULT_RESOLUTION,
                "description": (
                    "Default 1080x1920 — the vertical social delivery target. Accepts a tier "
                    "(720p, 1080p, 2K, 4K) or WIDTHxHEIGHT. 2K is an optional high-quality mode, "
                    "not a default: it costs more and is wasted on a 1080p timeline. Some models "
                    "support only one tier and reject the rest at validation."
                ),
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"],
                "default": _DEFAULT_ASPECT_RATIO,
            },
            "generate_audio": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Defaults to false: OpenMontage composes its own audio, and audio-on "
                    "tiers cost more. Set true only when model-generated audio is wanted."
                ),
            },
            "negative_prompt": {
                "type": "string",
                "description": (
                    "What to avoid. OpenRouter has no negative_prompt field, so this is appended "
                    "to the prompt text as 'Negative constraints: ...'. It is never sent as its "
                    "own request field."
                ),
            },
            "seed": {"type": "integer"},
            "frame_images": {
                "type": "array",
                "description": (
                    "Frame conditioning. Entries are normalised to the wire shape the API "
                    "actually accepts: {type: 'image_url', image_url: {url}, frame_type: "
                    "'first_frame'|'last_frame'}. A bare string, a local path, or the legacy "
                    "{type: 'first_frame', image_url: '<str>'} form are all accepted as input "
                    "and rewritten. Prefer the `first_frame` input for chaining."
                ),
                "items": {"type": "object"},
            },
            "input_references": {
                "type": "array",
                "description": "Style/identity guidance images",
                "items": {"type": "object"},
            },
            "output_path": {"type": "string", "default": "openrouter_output.mp4"},
            "first_frame": {
                "type": "string",
                "description": (
                    "Convenience form of frame_images: a local image path, an https URL or a "
                    "data: URI to condition this clip's opening frame on. This is the "
                    "cross-clip continuity mechanism — pass the previous act's final frame."
                ),
            },
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "job_id": {"type": "string"},
            "output": {"type": "string"},
            "reported_cost_usd": {"type": "number"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=200, network_required=True)
    retry_policy = RetryPolicy()
    idempotency_key_fields = ["prompt", "model", "duration", "seed"]
    side_effects = ["writes the generated MP4 to output_path", "bills the OpenRouter account"]
    fallback_tools = []
    user_visible_verification = [
        "Play the MP4 and confirm the motion matches the prompt",
        "Check reported_cost_usd against the estimate",
    ]

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _api_key() -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")

    @staticmethod
    def _headers(key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    @staticmethod
    def _rate(model: str) -> float:
        return _RATE_USD_PER_SECOND.get(model, _FALLBACK_RATE_USD_PER_SECOND)

    @staticmethod
    def _assemble_prompt(prompt: str, negative_prompt: str | None) -> str:
        """Fold the negatives into the prompt text.

        The positive prompt is never altered — the negatives are appended after
        it. Returns the prompt unchanged when there are no negatives, or when
        they are already present (so repeated assembly is a no-op).
        """
        positive = str(prompt).strip()
        negative = str(negative_prompt or "").strip()
        if not negative:
            return positive
        if _NEGATIVE_MARKER in positive or negative in positive:
            return positive
        return f"{positive} {_NEGATIVE_MARKER} {negative}"


    # ---- frame conditioning -------------------------------------------

    @staticmethod
    def _to_data_uri(src: str) -> str:
        """Accept an http(s) URL, a data: URI, or a local image path.

        A local path is read and base64-inlined into an outbound request body,
        so it is validated first: inside the working tree, real image bytes,
        under the size cap. See _IMAGE_MAGIC for why the extension is not
        trusted.
        """
        if src.startswith(("http://", "https://", "data:")):
            return src

        path = Path(src).expanduser().resolve()
        root = Path.cwd().resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError(
                f"frame image must live under the working directory ({root}); refusing to read {path}"
            ) from None
        if not path.is_file():
            raise FileNotFoundError(f"frame image not found: {src}")

        size = path.stat().st_size
        if size > _MAX_FRAME_BYTES:
            raise ValueError(f"frame image too large: {size} bytes > {_MAX_FRAME_BYTES}")

        data = path.read_bytes()
        if not data.startswith(_IMAGE_MAGIC):
            raise ValueError(
                f"frame image is not a recognised image format: {path.name}. "
                "Only real image bytes are sent to the provider."
            )

        import base64
        import mimetypes
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")

    @classmethod
    def _normalize_frame_image(cls, entry: Any, default_frame_type: str = "first_frame") -> dict:
        """Rewrite any accepted input form into the shape the API validates.

        The wire shape was established against the live endpoint, whose ZodError
        named all three required fields:

            {"type": "image_url",
             "image_url": {"url": "<https or data uri>"},
             "frame_type": "first_frame" | "last_frame"}

        The obvious-looking {"type": "first_frame", "image_url": "<str>"} is
        REJECTED with a 400 — `type` must be the literal "image_url", `image_url`
        must be an object, and the first/last selector lives in `frame_type`.
        Normalising here means no caller has to remember that.
        """
        frame_type = default_frame_type
        url: Any = None

        if isinstance(entry, str):
            url = entry
        elif isinstance(entry, dict):
            if "frame_type" in entry:
                # Explicit and wrong must fail loudly. Falling back to
                # first_frame would silently condition the wrong end of a clip.
                frame_type = entry["frame_type"]
            elif entry.get("type") in ("first_frame", "last_frame"):
                frame_type = entry["type"]  # legacy slot
            raw = entry.get("image_url", entry.get("url"))
            url = raw.get("url") if isinstance(raw, dict) else raw
        else:
            raise ValueError(f"unsupported frame image entry: {entry!r}")

        if not url or not isinstance(url, str):
            raise ValueError(f"frame image entry has no usable url: {entry!r}")
        if frame_type not in ("first_frame", "last_frame"):
            raise ValueError(f"frame_type must be first_frame or last_frame, got {frame_type!r}")

        return {"type": "image_url",
                "image_url": {"url": cls._to_data_uri(url)},
                "frame_type": frame_type}

    def _build_payload(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Assemble the POST /videos body. Optional keys are omitted, not nulled.

        `negative_prompt` is deliberately absent from the body: OpenRouter has no
        such field, so it is folded into `prompt` by _assemble_prompt instead.
        """
        payload: dict[str, Any] = {
            "model": str(inputs.get("model") or _DEFAULT_MODEL),
            "prompt": self._assemble_prompt(inputs["prompt"], inputs.get("negative_prompt")),
            "duration": int(inputs.get("duration", _DEFAULT_ACT_SECONDS)),
            "resolution": str(inputs.get("resolution", _DEFAULT_RESOLUTION)),
            "aspect_ratio": str(inputs.get("aspect_ratio", _DEFAULT_ASPECT_RATIO)),
            "generate_audio": bool(inputs.get("generate_audio", False)),
        }
        for optional in ("seed", "input_references"):
            if inputs.get(optional) is not None:
                payload[optional] = inputs[optional]

        # `first_frame` is the ergonomic chaining input; frame_images is the
        # explicit one. Both end up in the same normalised list.
        frames: list[Any] = []
        if inputs.get("first_frame"):
            frames.append({"frame_type": "first_frame", "image_url": inputs["first_frame"]})
        if inputs.get("frame_images"):
            frames.extend(inputs["frame_images"])
        if frames:
            payload["frame_images"] = [self._normalize_frame_image(f) for f in frames]
        return payload

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if self._api_key() else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        duration = int(inputs.get("duration", _DEFAULT_ACT_SECONDS) or _DEFAULT_ACT_SECONDS)
        return round(self._rate(str(inputs.get("model") or _DEFAULT_MODEL)) * duration, 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 120.0

    # ---- execution -----------------------------------------------------

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        key = self._api_key()
        if not key:
            return ToolResult(success=False, error="OPENROUTER_API_KEY not set. " + self.install_instructions)

        import requests

        start = time.time()
        try:
            payload = self._build_payload(inputs)
        except (FileNotFoundError, ValueError) as exc:
            # Fail securely like every other path in execute(): a structured
            # failure, not a traceback out of the tool.
            return ToolResult(success=False, error=f"invalid generation inputs: {exc}")
        output_path = Path(str(inputs.get("output_path", "openrouter_output.mp4")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = self._headers(key)

        try:
            submit = requests.post(f"{_API_BASE}/videos", headers=headers, json=payload, timeout=60)
            if submit.status_code >= 400:
                return ToolResult(success=False, error=f"OpenRouter submit failed ({submit.status_code}): {submit.text}")
            job = submit.json()
        except Exception as exc:
            return ToolResult(success=False, error=f"OpenRouter submit failed: {exc}")

        job_id = job.get("id")
        if not job_id:
            return ToolResult(success=False, error=f"OpenRouter response had no job id: {job}")
        poll_url = job.get("polling_url") or f"{_API_BASE}/videos/{job_id}"

        status = job.get("status", "pending")
        state: dict[str, Any] = job
        deadline = time.time() + _POLL_TIMEOUT_SECONDS
        while status not in (_TERMINAL_OK, _TERMINAL_FAIL):
            if time.time() > deadline:
                return ToolResult(success=False, error=f"OpenRouter job {job_id} timed out after {_POLL_TIMEOUT_SECONDS}s")
            time.sleep(_POLL_INTERVAL_SECONDS)
            try:
                poll = requests.get(poll_url, headers=headers, timeout=60)
                if poll.status_code >= 400:
                    return ToolResult(success=False, error=f"OpenRouter poll failed ({poll.status_code}): {poll.text}")
                state = poll.json()
            except Exception as exc:
                return ToolResult(success=False, error=f"OpenRouter poll failed: {exc}")
            status = state.get("status", status)

        if status == _TERMINAL_FAIL:
            return ToolResult(success=False, error=f"OpenRouter job {job_id} failed: {state.get('error') or state}")

        urls = state.get("unsigned_urls") or []
        download_url = urls[0] if urls else f"{_API_BASE}/videos/{job_id}/content?index=0"
        try:
            # Content URLs are unsigned — the auth header is required here too.
            media = requests.get(download_url, headers={"Authorization": f"Bearer {key}"}, timeout=300)
            if media.status_code >= 400:
                return ToolResult(success=False, error=f"OpenRouter download failed ({media.status_code}): {media.text}")
            output_path.write_bytes(media.content)
        except Exception as exc:
            return ToolResult(success=False, error=f"OpenRouter download failed: {exc}")

        reported = (state.get("usage") or {}).get("cost")
        return ToolResult(
            success=True,
            data={
                "provider": "openrouter",
                "model": payload["model"],
                "job_id": job_id,
                "prompt": payload["prompt"],
                "output": str(output_path),
                "duration_seconds": payload["duration"],
                "resolution": payload["resolution"],
                "aspect_ratio": payload["aspect_ratio"],
                "format": "mp4",
                "reported_cost_usd": reported,
            },
            artifacts=[str(output_path)],
            cost_usd=float(reported) if reported is not None else self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=payload["model"],
        )
