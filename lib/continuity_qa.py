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
    out.parent.mkdir(parents=True, exist_ok=True)
    pre = ["-sseof", "-0.1"] if at_end else ["-ss", "0.1"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *pre, "-i", str(video),
                    "-frames:v", "1", "-vf", f"scale={width}:-1", str(out)], check=True)
    return out


def collect_evidence(
    act_clips: list[tuple[str, str]],
    work_dir: str | Path,
    *,
    expected_width: int = 1080,
    expected_height: int = 1920,
    expected_duration: float = 5.0,
    duration_tolerance: float = 0.35,
) -> ContinuityReport:
    """Gather what a machine can measure, and stage what only a viewer can judge.

    ``act_clips`` is ordered ``[(act_id, mp4_path), ...]``.
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
        if facts.audio_streams:
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


def build_contact_sheet(report: ContinuityReport, out_path: str | Path, columns: int = 2) -> Optional[Path]:
    """Tile every boundary pair into one image so the agent reads it in one look.

    Side-by-side is the point: identity drift is obvious in a pair and easy to
    miss when the frames are viewed apart.
    """
    if not report.boundaries:
        return None
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    for b in report.boundaries:
        inputs += ["-i", b.last_frame_of_previous, "-i", b.first_frame_of_next]
    rows = len(report.boundaries)
    # ffmpeg's tile filter needs uniform input dimensions, and mismatched clip
    # aspect ratios are exactly what this module exists to flag — so the sheet
    # must not be the thing that crashes when it finds one.
    proc = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs,
                           "-filter_complex", f"tile={columns}x{rows}", str(out)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        report.measured_findings.append(
            "contact sheet could not be built (frames likely differ in dimensions): "
            f"{(proc.stderr or '').strip()[:200]}")
        return None
    report.contact_sheet = str(out)
    return out
