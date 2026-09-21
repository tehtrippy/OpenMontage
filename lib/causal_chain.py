"""Frame-chained generation of a causal act sequence.

Generate act 1, take its final frame, feed that frame as act 2's first frame,
repeat. The chain is what holds object identity across cuts: the A/B experiment
on 2026-09-21 showed causal prompts alone fix *within-clip* causality but leave
the working glass, board crop and lighting drifting between clips, while the
same prompts with chaining held all of them steady across five acts.

This is orchestration, not a provider. It lives in lib/ rather than tools/ on
purpose — a BaseTool with capability="video_generation" would be picked up by
`video_selector` as if it were another model to route to, which it is not.

Nothing here calls an API unless :func:`generate_chain` is invoked with
``dry_run=False``. The default is a dry run that plans the chain and returns the
payloads it *would* send.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Vertical social delivery target. Mirrors the adapter's defaults on purpose —
# see tools/video/openrouter_video.py. 2K is opt-in, never implicit.
#
# "1080p" is the provider's tier enum, and at 9:16 it IS 1080x1920 — the
# delivery frame stays the same, only the wire format changed. The old
# "1080x1920" here was rejected by every submit with a 400 ZodError.
DEFAULT_RESOLUTION = "1080p"
DEFAULT_ASPECT_RATIO = "9:16"
DEFAULT_ACT_SECONDS = 5
DEFAULT_MODEL = "google/veo-3.1-lite"

# Act ids are interpolated into output paths. Agent-authored ids must not be
# able to escape the run directory via "../".
_ACT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class CausalAct:
    """One causally complete action.

    ``prompt`` must describe PREVIOUS_STATE -> ACTION -> RESULTING_STATE with the
    action visually observable. A state-only description ("blueberry mash") is
    what produced the teleporting-ingredient failures; it does not belong here.
    """

    id: str
    prompt: str
    negative_prompt: str = ""
    duration_seconds: int = DEFAULT_ACT_SECONDS
    carries_identity_from: Optional[str] = None  # previous act id, for the record
    # Identity anchors for entities the handoff frame cannot carry. Chaining
    # transports only what is VISIBLE in the previous act's final frame: on
    # 2026-09-21 the muddler head was submerged in pulp at both handoffs, so
    # three acts invented three different tools (toothed crown -> flat puck ->
    # perforated barrel) while glass, board, hands and camera held perfectly.
    # Paths/URLs here are sent as input_references. Tuple, not list, because
    # CausalAct is frozen and a mutable default would be shared across acts.
    reference_images: tuple[str, ...] = ()
    # Optional closing-frame pin. The chain sets first_frame automatically; this
    # one is always explicit, because the frame an act should END on is a
    # production decision, not something the chain can derive. None means the
    # act is free at its close, which is the existing behaviour for every
    # caller written before this field existed.
    last_frame: Optional[str] = None


@dataclass
class ChainStep:
    act_id: str
    output_path: str
    payload: dict[str, Any]
    chained: bool
    first_frame_source: Optional[str] = None
    job_id: Optional[str] = None
    cost_usd: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class ChainResult:
    steps: list[ChainStep] = field(default_factory=list)
    total_cost_usd: float = 0.0
    dry_run: bool = True

    @property
    def ok(self) -> bool:
        return all(s.success for s in self.steps) if self.steps else False


def extract_final_frame(video_path: str | Path, out_jpg: str | Path) -> Path:
    """Grab the TRUE last decoded frame of a clip.

    Decodes straight through and lets ``-update 1`` overwrite the same output
    file on every frame, so what survives is the final frame by construction —
    no seeking, no duration probe, no arithmetic that can land short.

    It must stay that way. The previous implementation seeked with
    ``-sseof -0.1`` and took one frame, which on a 24fps clip lands on frame
    n-1 or n-2, not n. On the 2026-09-21 A1 diagnostic that one-frame miss was
    the difference between a gloved hand 11px above the glass rim and the same
    hand 379px clear of it; it was read as a generation defect and rerolled at
    $0.20 before the extractor was found to be the cause. ``frame_sampler`` is
    still the wrong tool here for the original reason — it addresses frames by
    absolute timestamp, so it needs a duration probe and still lands short when
    container and stream durations disagree (every H3 clip: 5.167s container
    for a 5s request).

    Note ``-frames:v`` must NOT be passed: combined with ``-update`` it would
    stop after the first frame and pin exactly the wrong end of the clip.
    """
    out = Path(out_jpg)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Full decode, no seek. Acts are 4-8s so this costs milliseconds; if clips
    # ever get long enough for it to matter, seek to duration-1s first and keep
    # -update on the remainder.
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video_path),
         "-an", "-update", "1", "-q:v", "3", str(out)],
        check=True,
    )
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"final-frame extraction produced nothing for {video_path}")
    return out


def build_chain_payloads(
    acts: list[CausalAct],
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    generate_audio: bool = False,
    chain: bool = True,
) -> list[ChainStep]:
    """Plan the chain without generating anything.

    The first act is never chained — there is no previous frame. Every later act
    carries ``first_frame`` pointing at the frame that *will* be extracted from
    its predecessor, so the plan is inspectable before a credit is spent.
    """
    for act in acts:
        if not _ACT_ID_RE.match(act.id):
            raise ValueError(
                f"act id {act.id!r} must match {_ACT_ID_RE.pattern} — ids become file paths"
            )

    out_dir = Path(output_dir)
    frames_dir = out_dir / "frames"
    steps: list[ChainStep] = []

    for i, act in enumerate(acts):
        video_out = out_dir / f"{act.id}.mp4"
        payload: dict[str, Any] = {
            "prompt": act.prompt,
            "model": model,
            "duration": act.duration_seconds,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "generate_audio": generate_audio,
            "output_path": str(video_out),
        }
        if act.negative_prompt:
            payload["negative_prompt"] = act.negative_prompt
        if act.reference_images:
            payload["input_references"] = list(act.reference_images)
        if act.last_frame:
            # last_frame turns the act into an interpolation between two known
            # states on every model tested so far, so it needs a first frame.
            # Refuse at plan time rather than mid-run: the chain supplies one to
            # every act but the first, and the first has nothing to chain from.
            if not (chain and i > 0):
                raise ValueError(
                    f"act {act.id!r} sets last_frame but has no first_frame "
                    "(it is the chain seed, or chaining is off). last_frame is an "
                    "interpolation endpoint, not a standalone identity pin - supply both "
                    "frames or neither."
                )
            payload["last_frame"] = act.last_frame

        first_frame_source = None
        if chain and i > 0:
            first_frame_source = str(frames_dir / f"{acts[i - 1].id}_last.jpg")
            payload["first_frame"] = first_frame_source

        steps.append(ChainStep(
            act_id=act.id,
            output_path=str(video_out),
            payload=payload,
            chained=bool(first_frame_source),
            first_frame_source=first_frame_source,
        ))
    return steps


def identity_mechanisms(model: str, model_capabilities: Optional[dict[str, Any]] = None
                        ) -> dict[str, bool]:
    """Which identity mechanisms are usable for this model, so a planner can choose.

    Model-agnostic by construction: nothing here knows a model name. Pass
    ``model_capabilities`` from the provider listing (for OpenRouter,
    ``OpenRouterVideo.fetch_model_capabilities(model)`` - metadata only, never
    billed) and the answer follows the listing. With no listing, only what the
    adapter reports about itself is used, and anything unadvertised is False,
    because "not advertised" must not read as "available".

    ``input_references`` is False unless a capability listing positively
    declares reference support. As of 2026-09-21 the OpenRouter video listing
    has no reference field at all, so this is False for every model there -
    which is the point: prompt constraints and handoff visibility carry identity
    until a provider says otherwise.
    """
    caps = model_capabilities or {}
    frames = tuple(caps.get("supported_frame_images") or ())
    reference_keys = ("supported_input_references", "supported_reference_images",
                      "input_references", "reference_images")
    supports_references = any(bool(caps.get(k)) for k in reference_keys)

    if not caps:
        # No listing: fall back to the adapter contract, which describes the
        # request schema, not any one model.
        from tools.video.openrouter_video import OpenRouterVideo

        supported = OpenRouterVideo.supports
        return {
            "prompt_constraints": True,
            "first_frame": bool(supported.get("image_to_video")),
            "last_frame": False,          # unknown per model; never assumed
            "input_references": False,    # never assumed - must be declared
            "interpolation_pair": False,
        }

    return {
        "prompt_constraints": True,
        "first_frame": "first_frame" in frames,
        "last_frame": "last_frame" in frames,
        "input_references": supports_references,
        # Both endpoints available means an interpolation job is expressible.
        # It is a distinct operation, not an identity mechanism.
        "interpolation_pair": "first_frame" in frames and "last_frame" in frames,
    }


def attach_entity_references(
    act: CausalAct,
    entity_bible: dict[str, Any],
    entity_ids: Iterable[str],
) -> CausalAct:
    """Return a copy of act carrying the bible's reference images for those entities.

    The entity bible already declares reference_images per entity
    (schemas/artifacts/entity_bible.schema.json) and nothing consumed it. This
    is that wiring: cite the entities an act must keep identical, and their
    anchors ride along with the prompt.

    Existing references are kept and bible ones appended, de-duplicated with
    order preserved, so calling it twice is a no-op rather than a doubling.
    An unknown entity_id raises — silently generating without the anchor is the
    failure this exists to prevent.
    """
    by_id = {e["entity_id"]: e for e in entity_bible.get("entities", [])}
    refs: list[str] = list(act.reference_images)
    for entity_id in entity_ids:
        entity = by_id.get(entity_id)
        if entity is None:
            raise KeyError(
                f"entity {entity_id!r} is not in the entity bible; "
                f"known ids: {sorted(by_id)}"
            )
        for ref in entity.get("reference_images", []):
            if ref not in refs:
                refs.append(ref)
    return replace(act, reference_images=tuple(refs))


def estimate_chain_cost(acts: list[CausalAct], model: str = DEFAULT_MODEL) -> float:
    """Total the chain against the adapter's rate table. Known before spending."""
    from tools.video.openrouter_video import OpenRouterVideo

    tool = OpenRouterVideo()
    return round(sum(
        tool.estimate_cost({"prompt": a.prompt, "model": model,
                            "duration": a.duration_seconds})
        for a in acts
    ), 4)


def generate_chain(
    acts: list[CausalAct],
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    resolution: str = DEFAULT_RESOLUTION,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    generate_audio: bool = False,
    chain: bool = True,
    dry_run: bool = True,
    executor: Optional[Callable[[dict[str, Any]], Any]] = None,
    stop_on_failure: bool = True,
) -> ChainResult:
    """Run the chain.

    ``dry_run=True`` (the default) plans and returns without calling anything —
    a chain that spends money should have to be asked for explicitly. Pass an
    ``executor`` to inject a fake in tests; otherwise the openrouter_video tool
    is used.

    ``stop_on_failure`` matters more here than in a normal batch: once an act
    fails there is no final frame to chain from, so continuing would silently
    produce an UNchained clip and quietly destroy the thing being tested.
    """
    steps = build_chain_payloads(
        acts, output_dir, model=model, resolution=resolution,
        aspect_ratio=aspect_ratio, generate_audio=generate_audio, chain=chain,
    )
    result = ChainResult(steps=steps, dry_run=dry_run)
    if dry_run:
        return result

    if executor is None:
        from tools.tool_registry import registry
        registry.ensure_discovered()
        tool = registry.get("openrouter_video")
        executor = tool.execute

    frames_dir = Path(output_dir) / "frames"
    for i, step in enumerate(steps):
        r = executor(step.payload)
        step.success = bool(getattr(r, "success", False))
        step.error = getattr(r, "error", None)
        data = getattr(r, "data", None) or {}
        step.job_id = data.get("job_id")
        step.cost_usd = data.get("reported_cost_usd")
        if step.cost_usd:
            result.total_cost_usd = round(result.total_cost_usd + step.cost_usd, 4)

        if not step.success:
            if stop_on_failure:
                break
            continue
        if chain and i + 1 < len(steps):
            try:
                extract_final_frame(step.output_path, frames_dir / f"{step.act_id}_last.jpg")
            except Exception as exc:
                # The clip is generated and already billed. Letting ffmpeg's
                # failure unwind would throw away every ChainStep and the
                # running cost total with it. Record and stop instead.
                step.error = f"final-frame extraction failed: {exc}"
                break

    return result
