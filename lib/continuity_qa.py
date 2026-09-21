"""Continuity evidence for a generated causal chain.

This does NOT decide whether continuity holds. No local tool can judge whether
two glasses are "the same glass" — that is a semantic question and the honest
answer is that a multimodal reviewer has to look. What this module does is make
looking cheap and systematic: it gathers the frames and the measurable signals
at the boundaries that matter, and hands back a report with every check marked
either MEASURED (a number a machine produced) or NEEDS_AGENT_REVIEW (a question
only a viewer can answer).

Designing it the other way round — a scorer that returns "continuity: 0.87" —
would launder a guess into a number, which is worse than no number.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Semantic dimensions. The module cannot score these; it stages the evidence and
# names the question so an agent review is systematic rather than impressionistic.
SEMANTIC_CHECKS = (
    "character_identity",
    "clothing_continuity",
    "object_identity",
    "prop_continuity",
    "container_continuity",
    "ingredient_continuity",
    "tool_continuity",
    "environment_continuity",
    "lighting_continuity",
    "camera_continuity",
    "state_progression",
    "action_visibility",
    "unexplained_state_change",
    "teleportation_or_morph_risk",
)


@dataclass
class ClipFacts:
    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    fps: Optional[str] = None
    audio_streams: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BoundaryEvidence:
    """The seam between act N and act N+1 — where continuity actually breaks."""

    from_act: str
    to_act: str
    last_frame_of_previous: str
    first_frame_of_next: str
    measured: dict[str, Any] = field(default_factory=dict)
    agent_review: dict[str, str] = field(default_factory=dict)


@dataclass
class ContinuityReport:
    clips: list[ClipFacts] = field(default_factory=list)
    boundaries: list[BoundaryEvidence] = field(default_factory=list)
    measured_findings: list[str] = field(default_factory=list)
    review_required: list[str] = field(default_factory=list)
    contact_sheet: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "continuity_qa_report",
            "verdict": "EVIDENCE_ONLY — semantic continuity requires agent review",
            "clips": [c.__dict__ for c in self.clips],
            "boundaries": [
                {"from_act": b.from_act, "to_act": b.to_act,
                 "last_frame_of_previous": b.last_frame_of_previous,
                 "first_frame_of_next": b.first_frame_of_next,
                 "measured": b.measured, "agent_review": b.agent_review}
                for b in self.boundaries
            ],
            "measured_findings": self.measured_findings,
            "review_required": self.review_required,
            "contact_sheet": self.contact_sheet,
        }


def _ffprobe(path: str | Path, args: list[str]) -> str:
    return subprocess.run(["ffprobe", "-v", "error", *args, str(path)],
                          capture_output=True, text=True).stdout.strip()


def probe_clip(path: str | Path) -> ClipFacts:
    p = Path(path)
    facts = ClipFacts(path=str(p))
    if not p.exists():
        facts.error = "missing"
        return facts
    try:
        facts.duration_seconds = float(_ffprobe(p, ["-show_entries", "format=duration", "-of", "csv=p=0"]))
        wh = _ffprobe(p, ["-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0"])
        w, h = wh.replace("\n", ",").split(",")[:2]
        facts.width, facts.height = int(w), int(h)
        facts.fps = _ffprobe(p, ["-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0"]).splitlines()[0]
        facts.audio_streams = len([x for x in _ffprobe(
            p, ["-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"]).splitlines() if x.strip()])
    except Exception as exc:  # a malformed clip is a finding, not a crash
        facts.error = f"{type(exc).__name__}: {exc}"
    return facts


def _extract(video: str | Path, out: Path, *, at_end: bool, width: int = 220) -> Path:
    """Stage one boundary frame as evidence.

    The end frame is taken the same way :func:`lib.causal_chain.extract_final_frame`
    takes it — decode through with ``-update 1``, no seek — because this pane is
    the evidence for what the chain handed forward. Sampling it 0.1s earlier than
    the frame actually used made the evidence disagree with the run, which is
    worse than having no evidence at all.

    The start frame stays at ``-ss 0.1``: the first decoded frame of a generated
    clip is sometimes a lead-in that misrepresents the opening state.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = ["-vf", f"scale={width}:-1"]
    cmd = (["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-an", "-update", "1", *scale]
           if at_end else
           ["ffmpeg", "-loglevel", "error", "-y", "-ss", "0.1", "-i", str(video),
            "-frames:v", "1", *scale])
    subprocess.run([*cmd, str(out)], check=True)
    return out


def collect_evidence(
    act_clips: list[tuple[str, str]],
    work_dir: str | Path,
    *,
    expected_width: int = 1080,
    expected_height: int = 1920,
    expected_duration: float = 5.0,
    duration_tolerance: float = 0.35,
    expect_audio: bool = False,
) -> ContinuityReport:
    """Gather what a machine can measure, and stage what only a viewer can judge.

    ``act_clips`` is ordered ``[(act_id, mp4_path), ...]``.

    ``expect_audio`` says which way the audio check points. Off (the default) an
    audio stream is the finding, because the production asked for picture only.
    On, a MISSING stream is the finding: a production whose sound is part of the
    deliverable would otherwise pass QA while being silent, while the one-way
    check flagged every correct clip instead.
    """
    work = Path(work_dir)
    report = ContinuityReport()

    for act_id, path in act_clips:
        facts = probe_clip(path)
        report.clips.append(facts)
        if facts.error:
            report.measured_findings.append(f"{act_id}: unreadable ({facts.error})")
            continue
        if (facts.width, facts.height) != (expected_width, expected_height):
            report.measured_findings.append(
                f"{act_id}: {facts.width}x{facts.height}, expected {expected_width}x{expected_height}")
        if facts.duration_seconds and abs(facts.duration_seconds - expected_duration) > duration_tolerance:
            report.measured_findings.append(
                f"{act_id}: duration {facts.duration_seconds:.3f}s, expected ~{expected_duration}s")
        if expect_audio and not facts.audio_streams:
            report.measured_findings.append(
                f"{act_id}: no audio stream; generated audio was requested")
        elif facts.audio_streams and not expect_audio:
            report.measured_findings.append(
                f"{act_id}: {facts.audio_streams} audio stream(s) present; picture-only was requested")

    # Boundaries are where identity drift shows. Pair last-of-N with first-of-N+1.
    for i in range(len(act_clips) - 1):
        (a_id, a_path), (b_id, b_path) = act_clips[i], act_clips[i + 1]
        if not (Path(a_path).exists() and Path(b_path).exists()):
            continue
        try:
            last = _extract(a_path, work / "boundaries" / f"{a_id}_last.jpg", at_end=True)
            first = _extract(b_path, work / "boundaries" / f"{b_id}_first.jpg", at_end=False)
        except Exception as exc:
            # Same rule as probe_clip: a bad clip is a finding, not a crash.
            report.measured_findings.append(
                f"{a_id} -> {b_id}: boundary frame extraction failed ({exc})")
            continue
        report.boundaries.append(BoundaryEvidence(
            from_act=a_id, to_act=b_id,
            last_frame_of_previous=str(last), first_frame_of_next=str(first),
            measured={"frames_staged": True},
            agent_review={c: "NEEDS_AGENT_REVIEW" for c in SEMANTIC_CHECKS},
        ))
        report.review_required.append(
            f"{a_id} -> {b_id}: compare {last.name} against {first.name} on all "
            f"{len(SEMANTIC_CHECKS)} semantic dimensions")

    return report


def _image_size(path: str | Path) -> Optional[tuple[int, int]]:
    """Pixel size of one pane, used as the box every other pane is fitted into."""
    wh = _ffprobe(path, ["-select_streams", "v:0", "-show_entries",
                         "stream=width,height", "-of", "csv=p=0"])
    parts = [f for line in wh.splitlines() for f in line.split(",") if f.strip()]
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def build_contact_sheet(report: ContinuityReport, out_path: str | Path, columns: int = 2) -> Optional[Path]:
    """Tile every boundary pair into one image so the agent reads it in one look.

    Side-by-side is the point: identity drift is obvious in a pair and easy to
    miss when the frames are viewed apart.

    Each pane must therefore actually BE in the sheet. `tile` is a single-input
    filter: fed N separate `-i` inputs it only ever consumed input 0, so every
    sheet was pane 1 repeated with the rest black — and it exited 0, so nothing
    flagged it. Found 2026-09-21 on the A1->A2 boundary, where the evidence for
    a $0.20 chained act was a black rectangle. The panes are concatenated into
    one stream first, so `tile` sees the N frames it is being asked to lay out.

    concat requires identical geometry, which the old tile-pads-it behaviour did
    not — mismatched aspect ratios are exactly what this module exists to flag,
    so each pane is letterboxed into the first pane's box rather than rejected.
    """
    if not report.boundaries:
        return None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    panes: list[str] = []
    for b in report.boundaries:
        panes += [b.last_frame_of_previous, b.first_frame_of_next]
    inputs: list[str] = []
    for pane in panes:
        inputs += ["-i", pane]

    box = _image_size(panes[0])
    if box is None:
        report.measured_findings.append(
            f"contact sheet could not be built: pane {panes[0]} is unreadable")
        return None
    # scale() rounds each side to an even number, which can overshoot an odd box
    # by a pixel and make pad() fail ("Padded dimensions cannot be smaller than
    # input dimensions") — _extract's scale=220:-1 produces exactly such a box.
    # An even box cannot be overshot by that rounding.
    w, h = (d + d % 2 for d in box)
    rows = -(-len(panes) // columns)  # ceil: every pane gets a cell

    normalise = "".join(
        f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[p{i}];"
        for i in range(len(panes))
    )
    joined = "".join(f"[p{i}]" for i in range(len(panes)))
    graph = (f"{normalise}{joined}concat=n={len(panes)}:v=1:a=0[panes];"
             f"[panes]tile={columns}x{rows}")

    proc = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs,
                           "-filter_complex", graph, "-frames:v", "1", str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        report.measured_findings.append(
            "contact sheet could not be built (frames likely differ in dimensions): "
            f"{(proc.stderr or '').strip()[:200]}")
        return None
    report.contact_sheet = str(out)
    return out
