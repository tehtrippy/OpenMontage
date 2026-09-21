"""Continuity QA evidence layer.

The contract worth pinning is as much about what this module REFUSES to do as
what it does: it must never emit a semantic continuity score. "Are these the
same glass?" is a question for a viewer, and a number there would launder a
guess into data. Every semantic dimension stays NEEDS_AGENT_REVIEW.

Clips are generated with ffmpeg's lavfi testsrc — no network, no paid API,
no fixtures checked in.

Run: pytest tests/tools/test_continuity_qa.py -v
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from lib.continuity_qa import (
    SEMANTIC_CHECKS,
    build_contact_sheet,
    collect_evidence,
    probe_clip,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _make_clip(path, seconds=1, width=108, height=192, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"color=c={color}:s={width}x{height}:d={seconds}:r=10",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


class TestProbeClip:
    def test_reads_real_media_facts(self, tmp_path):
        facts = probe_clip(_make_clip(tmp_path / "a.mp4"))
        assert (facts.width, facts.height) == (108, 192)
        assert facts.duration_seconds == pytest.approx(1.0, abs=0.2)
        assert facts.audio_streams == 0
        assert facts.error is None

    def test_missing_file_is_a_finding_not_a_crash(self, tmp_path):
        assert probe_clip(tmp_path / "nope.mp4").error == "missing"


class TestMeasuredFindings:
    def test_flags_wrong_resolution(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mp4", width=108, height=192)
        report = collect_evidence([("A1", str(clip))], tmp_path / "qa",
                                  expected_width=1080, expected_height=1920,
                                  expected_duration=1.0)
        assert any("expected 1080x1920" in f for f in report.measured_findings)

    def test_flags_wrong_duration(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mp4", seconds=1)
        report = collect_evidence([("A1", str(clip))], tmp_path / "qa",
                                  expected_width=108, expected_height=192,
                                  expected_duration=5.0)
        assert any("duration" in f for f in report.measured_findings)

    def test_clean_clip_produces_no_measured_findings(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mp4", seconds=1)
        report = collect_evidence([("A1", str(clip))], tmp_path / "qa",
                                  expected_width=108, expected_height=192,
                                  expected_duration=1.0)
        assert report.measured_findings == []

    def test_unreadable_clip_is_reported_not_raised(self, tmp_path):
        report = collect_evidence([("A1", str(tmp_path / "ghost.mp4"))], tmp_path / "qa")
        assert any("unreadable" in f for f in report.measured_findings)


class TestBoundaryEvidence:
    def test_one_boundary_per_adjacent_pair(self, tmp_path):
        clips = [(f"A{i}", str(_make_clip(tmp_path / f"a{i}.mp4"))) for i in range(1, 4)]
        report = collect_evidence(clips, tmp_path / "qa")
        assert len(report.boundaries) == 2
        assert [(b.from_act, b.to_act) for b in report.boundaries] == [("A1", "A2"), ("A2", "A3")]

    def test_stages_last_of_previous_against_first_of_next(self, tmp_path):
        from pathlib import Path
        clips = [(f"A{i}", str(_make_clip(tmp_path / f"a{i}.mp4"))) for i in (1, 2)]
        b = collect_evidence(clips, tmp_path / "qa").boundaries[0]
        assert Path(b.last_frame_of_previous).exists()
        assert Path(b.first_frame_of_next).exists()
        assert b.last_frame_of_previous.endswith("A1_last.jpg")
        assert b.first_frame_of_next.endswith("A2_first.jpg")

    def test_single_clip_has_no_boundaries(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mp4")
        assert collect_evidence([("A1", str(clip))], tmp_path / "qa").boundaries == []


class TestRefusesToScoreSemantics:
    """The load-bearing constraint. If this ever fails, someone has added a
    scorer and the report has started laundering guesses as measurements."""

    def test_every_semantic_dimension_defers_to_a_human(self, tmp_path):
        clips = [(f"A{i}", str(_make_clip(tmp_path / f"a{i}.mp4"))) for i in (1, 2)]
        review = collect_evidence(clips, tmp_path / "qa").boundaries[0].agent_review
        assert set(review) == set(SEMANTIC_CHECKS)
        assert set(review.values()) == {"NEEDS_AGENT_REVIEW"}

    def test_report_declares_itself_evidence_only(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mp4")
        d = collect_evidence([("A1", str(clip))], tmp_path / "qa").to_dict()
        assert d["verdict"].startswith("EVIDENCE_ONLY")
        assert "score" not in d

    def test_covers_the_named_continuity_dimensions(self):
        for required in ("object_identity", "ingredient_continuity", "container_continuity",
                         "tool_continuity", "lighting_continuity", "camera_continuity",
                         "state_progression", "action_visibility", "teleportation_or_morph_risk"):
            assert required in SEMANTIC_CHECKS


class TestContactSheet:
    def test_tiles_every_boundary_pair(self, tmp_path):
        clips = [(f"A{i}", str(_make_clip(tmp_path / f"a{i}.mp4"))) for i in (1, 2, 3)]
        report = collect_evidence(clips, tmp_path / "qa")
        out = build_contact_sheet(report, tmp_path / "qa" / "sheet.jpg")
        assert out is not None and out.exists() and out.stat().st_size > 0
        assert report.contact_sheet == str(out)

    def test_no_boundaries_yields_no_sheet(self, tmp_path):
        clip = _make_clip(tmp_path / "a.mp4")
        report = collect_evidence([("A1", str(clip))], tmp_path / "qa")
        assert build_contact_sheet(report, tmp_path / "qa" / "sheet.jpg") is None


class TestDegradesInsteadOfCrashing:
    """Regressions from the pre-push review of 920701c."""

    def test_unextractable_boundary_becomes_a_finding(self, tmp_path, monkeypatch):
        clips = [(f"A{i}", str(_make_clip(tmp_path / f"a{i}.mp4"))) for i in (1, 2)]
        monkeypatch.setattr("lib.continuity_qa._extract",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg failed")))
        report = collect_evidence(clips, tmp_path / "qa")
        assert report.boundaries == []
        assert any("boundary frame extraction failed" in f for f in report.measured_findings)

    def test_mismatched_dimensions_still_produce_a_sheet(self, tmp_path):
        """ffmpeg's tile filter pads mismatched inputs rather than failing, so
        differing aspect ratios do NOT break the sheet. Pinning the real
        behaviour: an earlier version of this test assumed a crash that does
        not happen."""
        clips = [("A1", str(_make_clip(tmp_path / "a1.mp4", width=108, height=192))),
                 ("A2", str(_make_clip(tmp_path / "a2.mp4", width=192, height=108)))]
        report = collect_evidence(clips, tmp_path / "qa",
                                  expected_width=108, expected_height=192, expected_duration=1.0)
        out = build_contact_sheet(report, tmp_path / "qa" / "sheet.jpg")
        assert out is not None and out.exists()

    def test_ffmpeg_failure_degrades_to_a_finding(self, tmp_path, monkeypatch):
        """Whatever makes ffmpeg fail, the sheet must report rather than raise."""
        clips = [(f"A{i}", str(_make_clip(tmp_path / f"a{i}.mp4"))) for i in (1, 2)]
        report = collect_evidence(clips, tmp_path / "qa")
        real = subprocess.run

        def failing(cmd, *a, **k):
            if cmd and cmd[0] == "ffmpeg" and "-filter_complex" in cmd:
                class R:
                    returncode = 1
                    stderr = "simulated tile failure"
                return R()
            return real(cmd, *a, **k)

        monkeypatch.setattr("lib.continuity_qa.subprocess.run", failing)
        assert build_contact_sheet(report, tmp_path / "qa" / "sheet.jpg") is None
        assert any("contact sheet could not be built" in f for f in report.measured_findings)
