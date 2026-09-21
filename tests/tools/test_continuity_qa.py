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


def _solid(path, color, width=100, height=150):
    """A single-colour pane, so a cell in the sheet can be identified by its mean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"color=c={color}:s={width}x{height}:d=1", "-frames:v", "1", str(path)],
        check=True,
    )
    return str(path)


def _report_of_panes(pane_paths):
    """A report whose boundaries point straight at the given panes, in order."""
    from lib.continuity_qa import BoundaryEvidence, ContinuityReport

    report = ContinuityReport()
    for i in range(0, len(pane_paths), 2):
        report.boundaries.append(BoundaryEvidence(
            from_act=f"A{i // 2 + 1}", to_act=f"A{i // 2 + 2}",
            last_frame_of_previous=pane_paths[i], first_frame_of_next=pane_paths[i + 1]))
    return report


def _cells(sheet_path, columns, rows):
    """Mean RGB of each grid cell, row-major — the order build_contact_sheet lays out."""
    import numpy as np
    from PIL import Image

    img = np.asarray(Image.open(sheet_path).convert("RGB"), dtype=float)
    h, w, _ = img.shape
    ch, cw = h // rows, w // columns
    out = []
    for r in range(rows):
        for c in range(columns):
            # Centre crop: the pad/letterbox border is not what identifies a pane.
            cell = img[r * ch + ch // 4: r * ch + 3 * ch // 4,
                       c * cw + cw // 4: c * cw + 3 * cw // 4]
            out.append(tuple(cell.reshape(-1, 3).mean(axis=0)))
    return out


class TestContactSheetContainsEveryPane:
    """`tile` is a single-input filter.

    Fed N separate `-i` inputs it consumed only input 0, so every sheet was pane
    1 repeated and the rest black — while exiting 0, so nothing flagged it. The
    old test only asserted the file existed, which a black sheet satisfies.
    """

    COLOURS = ["red", "lime", "blue", "yellow", "magenta", "cyan"]
    RGB = {"red": (255, 0, 0), "lime": (0, 255, 0), "blue": (0, 0, 255),
           "yellow": (255, 255, 0), "magenta": (255, 0, 255), "cyan": (0, 255, 255)}

    def _sheet(self, tmp_path, n_panes, columns=2):
        panes = [_solid(tmp_path / f"p{i}.png", self.COLOURS[i]) for i in range(n_panes)]
        report = _report_of_panes(panes)
        out = build_contact_sheet(report, tmp_path / "sheet.png", columns=columns)
        assert out is not None and out.exists(), report.measured_findings
        rows = -(-n_panes // columns)
        return _cells(out, columns, rows), panes

    def test_each_pane_appears_once_in_order(self, tmp_path):
        cells, _ = self._sheet(tmp_path, 6)
        for cell, colour in zip(cells, self.COLOURS):
            expected = self.RGB[colour]
            assert all(abs(got - want) < 40 for got, want in zip(cell, expected)),                 f"expected {colour} {expected}, got {cell}"

    def test_no_pane_is_repeated(self, tmp_path):
        cells, _ = self._sheet(tmp_path, 6)
        rounded = {tuple(round(v / 32) for v in cell) for cell in cells}
        assert len(rounded) == 6, f"only {len(rounded)} distinct panes in a 6-pane sheet"

    def test_no_cell_is_black_from_the_tiling(self, tmp_path):
        """Every input was bright, so a dark cell means a cell got no input."""
        cells, _ = self._sheet(tmp_path, 6)
        for i, cell in enumerate(cells):
            assert max(cell) > 60, f"cell {i} is black: {cell}"

    def test_single_boundary_two_panes_still_works(self, tmp_path):
        cells, _ = self._sheet(tmp_path, 2)
        assert len(cells) == 2
        assert all(abs(g - w) < 40 for g, w in zip(cells[0], self.RGB["red"]))
        assert all(abs(g - w) < 40 for g, w in zip(cells[1], self.RGB["lime"]))
        assert cells[0] != cells[1]

    def test_real_boundary_panes_are_not_identical(self, tmp_path):
        """End to end through collect_evidence: a red clip and a blue clip."""
        clips = [("A1", str(_make_clip(tmp_path / "a1.mp4", color="red"))),
                 ("A2", str(_make_clip(tmp_path / "a2.mp4", color="blue")))]
        report = collect_evidence(clips, tmp_path / "qa")
        out = build_contact_sheet(report, tmp_path / "qa" / "sheet.png")
        assert out is not None
        left, right = _cells(out, 2, 1)
        assert left[0] > left[2] and right[2] > right[0], (left, right)


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


class TestEndFrameMatchesTheChainHandoff:
    """The evidence pane must show the frame the chain actually handed forward.

    Sampling the end frame 0.1s early made the QA evidence disagree with the
    run itself — on the 2026-09-21 A1 diagnostic that gap read as a generation
    defect and cost a $0.20 reroll.
    """

    def _ramp_clip(self, path, n_frames=24, fps=24):
        from PIL import Image
        src = path.parent / "src"
        src.mkdir(parents=True, exist_ok=True)
        values = [10 * (i + 1) for i in range(n_frames)]
        for i, v in enumerate(values):
            Image.new("RGB", (160, 288), (v, v, v)).save(src / f"{i:04d}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-i", str(src / "%04d.png"), "-c:v", "libx264", "-qp", "0",
                        "-pix_fmt", "yuv444p", str(path)], check=True)
        return values

    def test_end_frame_is_the_true_final_frame(self, tmp_path):
        import numpy as np
        from PIL import Image
        from lib.continuity_qa import _extract

        clip = tmp_path / "ramp.mp4"
        values = self._ramp_clip(clip)
        out = _extract(clip, tmp_path / "end.jpg", at_end=True, width=160)
        grey = float(np.asarray(Image.open(out).convert("L"), dtype=float).mean())
        assert grey == pytest.approx(values[-1], abs=4)

    def test_end_frame_matches_what_the_chain_extractor_returns(self, tmp_path):
        """One clip, both extractors, same frame. They must not drift apart."""
        import numpy as np
        from PIL import Image
        from lib.causal_chain import extract_final_frame
        from lib.continuity_qa import _extract

        clip = tmp_path / "ramp.mp4"
        self._ramp_clip(clip)
        qa = _extract(clip, tmp_path / "qa_end.jpg", at_end=True, width=160)
        chain = extract_final_frame(clip, tmp_path / "chain_end.jpg")

        def grey(p):
            return float(np.asarray(Image.open(p).convert("L"), dtype=float).mean())

        assert grey(qa) == pytest.approx(grey(chain), abs=4)

    def test_end_frame_command_does_not_seek(self, monkeypatch, tmp_path):
        from pathlib import Path
        from lib.continuity_qa import _extract
        seen = {}

        def fake_run(cmd, *a, **k):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"\xff\xd8\xff" + b"x" * 64)
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr("lib.continuity_qa.subprocess.run", fake_run)
        _extract(tmp_path / "in.mp4", tmp_path / "out.jpg", at_end=True)
        assert "-sseof" not in seen["cmd"]
        assert "-update" in seen["cmd"]
        assert "-frames:v" not in seen["cmd"]

    def test_start_frame_still_seeks_past_the_lead_in(self, monkeypatch, tmp_path):
        """Backward compatibility: only the end-frame path changed."""
        from pathlib import Path
        from lib.continuity_qa import _extract
        seen = {}

        def fake_run(cmd, *a, **k):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"\xff\xd8\xff" + b"x" * 64)
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr("lib.continuity_qa.subprocess.run", fake_run)
        _extract(tmp_path / "in.mp4", tmp_path / "out.jpg", at_end=False)
        assert "-ss" in seen["cmd"]
        assert "0.1" in seen["cmd"]
        assert "-frames:v" in seen["cmd"]


class TestAudioExpectation:
    """Which way the audio check points is a property of the production.

    Picture-only work wants a stream to be a finding. Work whose sound IS the
    deliverable wants the opposite - and the one-way check flagged every correct
    clip while letting a silent one through.
    """

    @staticmethod
    def _clip_with_tone(path, seconds=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"color=c=red:s=108x192:d={seconds}:r=10",
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
            check=True)
        return path

    def test_missing_audio_is_a_finding_when_audio_was_generated(self, tmp_path):
        silent = _make_clip(tmp_path / "a.mp4")
        report = collect_evidence([("A1", str(silent))], tmp_path / "qa",
                                  expected_width=108, expected_height=192,
                                  expected_duration=1.0, expect_audio=True)
        assert any("no audio stream" in f for f in report.measured_findings)

    def test_present_audio_is_clean_when_audio_was_generated(self, tmp_path):
        clip = self._clip_with_tone(tmp_path / "a.mp4")
        report = collect_evidence([("A1", str(clip))], tmp_path / "qa",
                                  expected_width=108, expected_height=192,
                                  expected_duration=1.0, expect_audio=True)
        assert report.measured_findings == []

    def test_picture_only_behaviour_is_unchanged(self, tmp_path):
        clip = self._clip_with_tone(tmp_path / "a.mp4")
        report = collect_evidence([("A1", str(clip))], tmp_path / "qa",
                                  expected_width=108, expected_height=192,
                                  expected_duration=1.0)
        assert any("picture-only was requested" in f for f in report.measured_findings)

    def test_silent_clip_is_clean_when_picture_only(self, tmp_path):
        silent = _make_clip(tmp_path / "a.mp4")
        report = collect_evidence([("A1", str(silent))], tmp_path / "qa",
                                  expected_width=108, expected_height=192,
                                  expected_duration=1.0)
        assert report.measured_findings == []
