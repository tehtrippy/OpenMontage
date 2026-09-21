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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Vertical social delivery target. Mirrors the adapter's defaults on purpose —
# see tools/video/openrouter_video.py. 2K is opt-in, never implicit.
DEFAULT_RESOLUTION = "1080x1920"
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
