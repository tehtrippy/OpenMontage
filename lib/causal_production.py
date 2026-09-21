"""End-to-end runner for a causal production: plan -> approval -> chain -> QA -> edit.

The pieces this connects all existed already and none of them are changed here:
:mod:`lib.causal_chain` generates a frame-chained act sequence and extracts the
true final frame between acts, :mod:`lib.continuity_qa` stages boundary evidence,
:func:`lib.checkpoint.init_project` creates the workspace the board reads, and
ffmpeg cuts the result. What was missing was the seam between them: every run so
far threaded frames, QA and the edit by hand, which is exactly the kind of manual
step that silently drifts between runs.

Two rules shape the design:

* **Nothing spends money without an explicit approval.** :func:`dry_run` is
  free and total, :func:`execute` refuses unless ``approved=True``.
* **The plan is data.** An agent writes a :class:`ProductionPlan` from the
  user's idea; this module never invents creative content.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from lib.causal_chain import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MODEL,
    CausalAct,
    ChainResult,
    build_chain_payloads,
    generate_chain,
)
from lib.continuity_qa import build_contact_sheet, collect_evidence

DEFAULT_RESOLUTION = "1080p"  # provider tier; at 9:16 this is 1080x1920
DEFAULT_FPS = 24


@dataclass(frozen=True)
class Shot:
    """One act plus what the edit does with it.

    ``cut_seconds`` trims the TAIL only, and only in the edit - generation still
    produces the whole causal action. Compressing an action at generation time is
    what produced state-only clips in the first place; cut rhythm is an edit
    property.
    """

    act: CausalAct
    beat: str
    cut_seconds: Optional[float] = None

    @property
    def edit_seconds(self) -> float:
        return self.cut_seconds if self.cut_seconds is not None else float(self.act.duration_seconds)


@dataclass(frozen=True)
class ProductionPlan:
    project_id: str
    title: str
    concept: str
    shots: tuple[Shot, ...]
    model: str = DEFAULT_MODEL
    resolution: str = DEFAULT_RESOLUTION
    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    fps: int = DEFAULT_FPS
    generate_audio: bool = False
    pipeline_type: str = "cinematic"
    risks: tuple[str, ...] = ()

    @property
    def project_dir(self) -> Path:
        return Path("projects") / self.project_id

    @property
    def acts_dir(self) -> Path:
        return self.project_dir / "assets" / "video" / "acts"

    @property
    def output_path(self) -> Path:
        return self.project_dir / "renders" / "final.mp4"

    @property
    def acts(self) -> list[CausalAct]:
        return [s.act for s in self.shots]

    @property
    def generated_seconds(self) -> int:
        return sum(s.act.duration_seconds for s in self.shots)

    @property
    def edit_seconds(self) -> float:
        return round(sum(s.edit_seconds for s in self.shots), 3)


# ----------------------------------------------------------------- planning


def plan_payloads(plan: ProductionPlan) -> list:
    """The exact generation calls, inspectable before a credit is spent."""
    return build_chain_payloads(
        plan.acts, plan.acts_dir, model=plan.model, resolution=plan.resolution,
        aspect_ratio=plan.aspect_ratio, generate_audio=plan.generate_audio,
    )


def estimate_cost(plan: ProductionPlan) -> float:
    """Price the run at the SKU that applies - resolution and audio included."""
    from tools.video.openrouter_video import OpenRouterVideo

    tool = OpenRouterVideo()
    return round(sum(
        tool.estimate_cost({
            "prompt": s.act.prompt, "model": plan.model, "duration": s.act.duration_seconds,
            "resolution": plan.resolution, "generate_audio": plan.generate_audio,
        })
        for s in plan.shots
    ), 4)


def approval_report(plan: ProductionPlan) -> str:
    """The human gate: everything needed to say yes or no, in one page."""
    from tools.video.openrouter_video import OpenRouterVideo

    tool = OpenRouterVideo()
    steps = plan_payloads(plan)
    lines = [
        f"# {plan.title}",
        "",
        plan.concept,
        "",
        "## Shot plan",
        "",
        "| # | Beat | Generated | In edit | Chained from |",
        "|---|---|---|---|---|",
    ]
    for shot, step in zip(plan.shots, steps):
        chained = shot.act.carries_identity_from if step.chained else "- (chain seed)"
        lines.append(
            f"| {shot.act.id} | {shot.beat} | {shot.act.duration_seconds}s | "
            f"{shot.edit_seconds}s | {chained} |"
        )
    lines += [
        "",
        f"- **Model:** `{plan.model}`",
        f"- **Resolution:** {plan.resolution} at {plan.aspect_ratio} "
        f"({'1080x1920' if plan.resolution == '1080p' else plan.resolution}), "
        f"{plan.fps}fps, audio {'on' if plan.generate_audio else 'off'}",
        f"- **Generations:** {len(plan.shots)} "
        f"({plan.generated_seconds}s generated, {plan.edit_seconds}s in the edit)",
        "- **Chaining:** each act after the first opens on the TRUE final frame of the previous "
        "act (full decode, no seek). No last_frame, no input_references.",
        f"- **Estimated cost:** ${estimate_cost(plan):.2f}",
        f"- **Expected output:** `{plan.output_path.as_posix()}`",
        "",
        "## Exact prompts",
        "",
    ]
    for shot in plan.shots:
        lines += [f"### {shot.act.id} - {shot.beat} ({shot.act.duration_seconds}s)", "",
                  shot.act.prompt, ""]
        if shot.act.negative_prompt:
            lines += [f"*Negatives:* {shot.act.negative_prompt}", ""]
    if plan.risks:
        lines += ["## Known risks", ""] + [f"- {r}" for r in plan.risks] + [""]
    lines += ["## Per-shot cost", "",
              "| Shot | Duration | Cost |", "|---|---|---|"]
    for shot in plan.shots:
        cost = tool.estimate_cost({
            "prompt": shot.act.prompt, "model": plan.model,
            "duration": shot.act.duration_seconds, "resolution": plan.resolution,
            "generate_audio": plan.generate_audio})
        lines.append(f"| {shot.act.id} | {shot.act.duration_seconds}s | ${cost:.2f} |")
    lines += [f"| **Total** | **{plan.generated_seconds}s** | **${estimate_cost(plan):.2f}** |", ""]
    return "\n".join(lines)


def dry_run(plan: ProductionPlan) -> dict[str, Any]:
    """Everything the run would do, without doing any of it.

    Frame payloads are reported by PATH rather than as the base64 the adapter
    will inline, so the plan stays readable; the adapter does the conversion at
    submit time behind its working-tree and magic-byte guards.
    """
    steps = plan_payloads(plan)
    calls = []
    for shot, step in zip(plan.shots, steps):
        calls.append({
            "act": shot.act.id,
            "beat": shot.beat,
            "tool": "openrouter_video",
            "chained": step.chained,
            "first_frame": step.first_frame_source,
            "payload": dict(step.payload),
        })
    return {
        "kind": "causal_production_dry_run",
        "project_id": plan.project_id,
        "title": plan.title,
        "workspace": plan.project_dir.as_posix(),
        "generation_calls": calls,
        "handoffs": [
            {"after": a.id, "extract": (plan.acts_dir / "frames" / f"{a.id}_last.jpg").as_posix(),
             "extractor": "lib.causal_chain.extract_final_frame (full decode + -update 1)",
             "feeds": b.id}
            for a, b in zip(plan.acts, plan.acts[1:])
        ],
        "qa": {
            "probe": [f"{a.id}: resolution / duration / fps / audio streams" for a in plan.acts],
            "boundaries": [f"{a.id} -> {b.id}" for a, b in zip(plan.acts, plan.acts[1:])],
            "contact_sheet": (plan.project_dir / "renders" / "qa" / "boundaries.jpg").as_posix(),
            "semantic": "NEEDS_AGENT_REVIEW - no automated identity score",
        },
        "edit": {
            "segments": [{"act": s.act.id, "in": 0.0, "out": s.edit_seconds,
                          "frames": int(round(s.edit_seconds * plan.fps))} for s in plan.shots],
            "transitions": "hard cuts only",
            "command": _cut_command(
                [(str(plan.acts_dir / f"{s.act.id}.mp4"), s.edit_seconds) for s in plan.shots],
                plan.output_path, plan.fps, plan.generate_audio),
            "audio": "carried through the cut" if plan.generate_audio else "picture only (-an)",
        },
        "output": plan.output_path.as_posix(),
        "expected_duration_seconds": plan.edit_seconds,
        "generations": len(plan.shots),
        "estimated_cost_usd": estimate_cost(plan),
        "spends_credits": False,
    }


# -------------------------------------------------------------------- edit


def _cut_command(segments: list[tuple[str, float]], out_path: Path, fps: int,
                 audio: bool = False) -> list[str]:
    """Frame-exact trim + hard-cut concat in ONE encode.

    One pass rather than trim-then-concat: every extra encode is another
    generation loss on footage that already went through conditioned generation.
    Stream-copy is not an option - these clips are a single GOP with ~70
    B-frames, so a copy trim cuts mid-GOP.

    ``audio=True`` carries the clips' own soundtrack through the cut. It stays
    off by default because a picture-only source has no audio stream to map and
    concat would fail on the missing input; the caller knows which it generated
    (``ProductionPlan.generate_audio``). Where a production DOES generate audio,
    dropping it here would deliver a silent MP4 assembled from clips that each
    had sound - the failure this flag exists to prevent, and one that costs a
    full run to discover.
    """
    inputs: list[str] = []
    trims: list[str] = []
    labels = ""
    for i, (path, seconds) in enumerate(segments):
        inputs += ["-i", str(path)]
        frames = int(round(seconds * fps))
        trims.append(
            f"[{i}:v]trim=start_frame=0:end_frame={frames},"
            f"setpts=PTS-STARTPTS[s{i}];")
        labels += f"[s{i}]"
        if audio:
            # Cut audio to the wall-clock length the FRAME trim produces, not to
            # `seconds`: rounding to a whole frame is what the picture actually
            # gets, and any other number drifts the streams apart segment by
            # segment. concat takes the pairs interleaved - [v0][a0][v1][a1].
            trims.append(
                f"[{i}:a]atrim=end={frames / fps:.6f},asetpts=PTS-STARTPTS[a{i}];")
            labels += f"[a{i}]"
    graph = "".join(trims) + (
        f"{labels}concat=n={len(segments)}:v=1:a={'1[v][a]' if audio else '0[v]'}")
    maps = ["-map", "[v]", "-map", "[a]"] if audio else ["-map", "[v]"]
    audio_codec = ["-c:a", "aac", "-b:a", "192k"] if audio else ["-an"]
    return ["ffmpeg", "-loglevel", "error", "-y", *inputs,
            "-filter_complex", graph, *maps, "-r", str(fps),
            "-c:v", "libx264", "-crf", "16", "-preset", "slow",
            "-pix_fmt", "yuv420p", *audio_codec, "-movflags", "+faststart", str(out_path)]


def render_cut(segments: list[tuple[str, float]], out_path: str | Path,
               fps: int = DEFAULT_FPS, audio: bool = False) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(_cut_command(segments, out, fps, audio), check=True)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"edit produced nothing at {out}")
    return out


# ----------------------------------------------------------------- execute


@dataclass
class ProductionResult:
    plan_id: str
    chain: Optional[ChainResult] = None
    output: Optional[str] = None
    qa: dict[str, Any] = field(default_factory=dict)
    report: Optional[str] = None
    cost_usd: float = 0.0
    ok: bool = False


def execute(plan: ProductionPlan, *, approved: bool = False,
            executor: Any = None) -> ProductionResult:
    """Run the approved plan end to end. Refuses without an explicit approval.

    A failed act stops the chain (there is no final frame to carry forward) and
    the edit is skipped rather than assembled from a hole.
    """
    if not approved:
        raise PermissionError(
            f"{plan.project_id}: generation costs money and has not been approved. "
            "Present approval_report(plan) to a human, then call execute(plan, approved=True)."
        )

    from lib.checkpoint import init_project

    # pipeline_dir keeps the workspace where the plan says it is. Without it
    # init_project writes to the package-level PROJECTS_DIR, so a run under a
    # different working directory - a test, a sandbox - would quietly scatter a
    # project.json into the repo while every other path stayed local.
    init_project(plan.project_id, title=plan.title, pipeline_type=plan.pipeline_type,
                 pipeline_dir=plan.project_dir.parent)
    result = ProductionResult(plan_id=plan.project_id)

    chain = generate_chain(
        plan.acts, plan.acts_dir, model=plan.model, resolution=plan.resolution,
        aspect_ratio=plan.aspect_ratio, generate_audio=plan.generate_audio,
        chain=True, dry_run=False, executor=executor,
    )
    result.chain = chain
    result.cost_usd = chain.total_cost_usd
    if not chain.ok:
        result.report = _write_report(plan, result, failed=True)
        return result

    qa_dir = plan.project_dir / "renders" / "qa"
    report = collect_evidence(
        [(s.act.id, str(plan.acts_dir / f"{s.act.id}.mp4")) for s in plan.shots],
        qa_dir, expected_duration=float(plan.shots[0].act.duration_seconds),
        expect_audio=plan.generate_audio,
    )
    build_contact_sheet(report, qa_dir / "boundaries.jpg")
    result.qa = report.to_dict()

    result.output = str(render_cut(
        [(str(plan.acts_dir / f"{s.act.id}.mp4"), s.edit_seconds) for s in plan.shots],
        plan.output_path, plan.fps, plan.generate_audio))
    result.ok = True
    result.report = _write_report(plan, result)
    return result


def _write_report(plan: ProductionPlan, result: ProductionResult, failed: bool = False) -> str:
    """Production report: what was made, what it cost, what QA could and could not say."""
    body = {
        "kind": "production_report",
        "project_id": plan.project_id,
        "title": plan.title,
        "status": "failed" if failed else "completed",
        "model": plan.model,
        "resolution": plan.resolution,
        "aspect_ratio": plan.aspect_ratio,
        "generations": len(plan.shots),
        "generated_seconds": plan.generated_seconds,
        "edit_seconds": plan.edit_seconds,
        "estimated_cost_usd": estimate_cost(plan),
        "billed_cost_usd": result.cost_usd,
        "output": result.output,
        "acts": [
            {"act": s.act_id, "clip": s.output_path, "chained": s.chained,
             "first_frame": s.first_frame_source, "job_id": s.job_id,
             "cost_usd": s.cost_usd, "success": s.success, "error": s.error}
            for s in (result.chain.steps if result.chain else [])
        ],
        "qa": result.qa,
        "qa_caveat": "Continuity QA stages evidence and measures what a machine can measure. "
                     "Semantic identity is NEEDS_AGENT_REVIEW; boundary frames match by "
                     "construction, so mid-clip drift needs its own look.",
    }
    path = plan.project_dir / "artifacts" / "production_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return str(path)
