"""The production runner: plan -> approval -> chain -> QA -> edit, with no API.

The generation call is injected, so the whole path is exercised end to end for
free: ffmpeg makes the "generated" clips, the real chain extracts the real final
frames between acts, the real QA probes them and the real edit assembles them.

Run: pytest tests/lib/test_causal_production.py -v
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lib.causal_chain import CausalAct
from lib.causal_production import (
    ProductionPlan,
    Shot,
    _cut_command,
    approval_report,
    dry_run,
    estimate_cost,
    execute,
    plan_payloads,
    render_cut,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _plan(**overrides) -> ProductionPlan:
    shots = (
        Shot(CausalAct(id="S1", prompt="hands set the rusted gear down", duration_seconds=4),
             beat="HOOK", cut_seconds=3.5),
        Shot(CausalAct(id="S2", prompt="the wire brush bites the rim",
                       negative_prompt="no clean metal", duration_seconds=6,
                       carries_identity_from="S1"), beat="ESCALATION"),
        Shot(CausalAct(id="S3", prompt="the rust line retreats", duration_seconds=6,
                       carries_identity_from="S2"), beat="TRANSFORMATION", cut_seconds=5.5),
        Shot(CausalAct(id="S4", prompt="the cloth brings up a mirror", duration_seconds=8,
                       carries_identity_from="S3"), beat="PAYOFF"),
    )
    kwargs = dict(project_id="test-rust-mirror", title="RUST / MIRROR",
                  concept="One gear, four actions.", shots=shots)
    kwargs.update(overrides)
    return ProductionPlan(**kwargs)


class TestApprovalGate:
    """Money is never spent by default - the gate is a hard refusal, not a warning."""

    def test_execute_refuses_without_approval(self):
        with pytest.raises(PermissionError, match="has not been approved"):
            execute(_plan())

    def test_refusal_calls_no_generator(self):
        calls = []
        with pytest.raises(PermissionError):
            execute(_plan(), executor=lambda p: calls.append(p))
        assert calls == []

    def test_refusal_names_the_way_forward(self):
        with pytest.raises(PermissionError, match="approval_report"):
            execute(_plan())


class TestDryRun:
    def test_spends_nothing_and_says_so(self):
        assert dry_run(_plan())["spends_credits"] is False

    def test_one_generation_call_per_shot(self):
        calls = dry_run(_plan())["generation_calls"]
        assert [c["act"] for c in calls] == ["S1", "S2", "S3", "S4"]
        assert [c["beat"] for c in calls] == ["HOOK", "ESCALATION", "TRANSFORMATION", "PAYOFF"]

    def test_first_act_is_the_chain_seed_and_the_rest_are_chained(self):
        calls = dry_run(_plan())["generation_calls"]
        assert calls[0]["chained"] is False and calls[0]["first_frame"] is None
        for call, prev in zip(calls[1:], ["S1", "S2", "S3"]):
            assert call["chained"] is True
            assert call["first_frame"].endswith(f"{prev}_last.jpg")

    def test_payload_carries_the_production_settings(self):
        payload = dry_run(_plan())["generation_calls"][0]["payload"]
        assert payload["model"] == "google/veo-3.1-lite"
        assert payload["resolution"] == "1080p"
        assert payload["aspect_ratio"] == "9:16"
        assert payload["generate_audio"] is False
        assert payload["duration"] == 4

    def test_no_last_frame_and_no_references_anywhere(self):
        """Both were ruled out by measurement; the runner must not sneak them back."""
        for call in dry_run(_plan())["generation_calls"]:
            assert "last_frame" not in call["payload"]
            assert "input_references" not in call["payload"]

    def test_handoffs_name_the_true_final_frame_extractor(self):
        handoffs = dry_run(_plan())["handoffs"]
        assert [(h["after"], h["feeds"]) for h in handoffs] == [
            ("S1", "S2"), ("S2", "S3"), ("S3", "S4")]
        assert all("extract_final_frame" in h["extractor"] for h in handoffs)

    def test_qa_covers_every_clip_and_boundary_without_scoring(self):
        qa = dry_run(_plan())["qa"]
        assert len(qa["probe"]) == 4
        assert qa["boundaries"] == ["S1 -> S2", "S2 -> S3", "S3 -> S4"]
        assert "NEEDS_AGENT_REVIEW" in qa["semantic"]

    def test_edit_plan_is_frame_exact(self):
        edit = dry_run(_plan())["edit"]
        assert [s["frames"] for s in edit["segments"]] == [84, 144, 132, 192]
        assert edit["transitions"] == "hard cuts only"

    def test_expected_duration_and_output(self):
        d = dry_run(_plan())
        assert d["expected_duration_seconds"] == 23.0
        assert d["output"] == "projects/test-rust-mirror/renders/final.mp4"

    def test_cost_matches_the_published_skus(self):
        # 4s + 6s + 6s + 8s at the 1080p audio-off SKU of $0.05/s
        assert dry_run(_plan())["estimated_cost_usd"] == pytest.approx(1.20)
        assert estimate_cost(_plan(resolution="720p")) == pytest.approx(0.72)


class TestApprovalReport:
    def test_contains_what_a_human_needs_to_decide(self):
        report = approval_report(_plan())
        for required in ("RUST / MIRROR", "google/veo-3.1-lite", "1080p", "9:16",
                         "$1.20", "renders/final.mp4", "HOOK", "PAYOFF"):
            assert required in report

    def test_contains_every_prompt_verbatim(self):
        plan = _plan()
        report = approval_report(plan)
        for shot in plan.shots:
            assert shot.act.prompt in report
        assert "no clean metal" in report  # negatives too

    def test_reports_risks_when_the_plan_declares_them(self):
        assert "identity drift" in approval_report(_plan(risks=("identity drift",)))


class TestEditCommand:
    def test_single_encode_hard_cuts_no_audio(self):
        cmd = _cut_command([("a.mp4", 3.5), ("b.mp4", 6.0)], Path("out.mp4"), 24)
        assert cmd.count("-filter_complex") == 1
        assert "concat=n=2:v=1:a=0" in cmd[cmd.index("-filter_complex") + 1]
        assert "-an" in cmd and "-crf" in cmd
        assert cmd[cmd.index("-crf") + 1] == "16"

    def test_trims_are_frame_exact(self):
        cmd = _cut_command([("a.mp4", 3.5)], Path("out.mp4"), 24)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "trim=start_frame=0:end_frame=84" in graph

    def test_renders_a_real_cut(self, tmp_path):
        clips = []
        for i, colour in enumerate(("red", "green")):
            p = tmp_path / f"c{i}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", f"color=c={colour}:s=64x112:d=1:r=10", "-pix_fmt", "yuv420p",
                            str(p)], check=True)
            clips.append((str(p), 1.0))
        out = render_cut(clips, tmp_path / "final.mp4", fps=10)
        assert out.exists() and out.stat().st_size > 0
        frames = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True).stdout.strip()
        assert int(frames) == 20


class TestAudioIsCarriedThroughTheEdit:
    """A production that generates audio must not deliver a silent MP4.

    The runner used to render every cut with `-an`, so ECHO-style work would
    have paid for four clips with native audio and assembled them into silence.
    That is only catchable at the end of a run, which is the expensive place.
    """

    @staticmethod
    def _clip_with_tone(path: Path, colour: str, hz: int) -> tuple[str, float]:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"color=c={colour}:s=64x112:d=1:r=10",
             "-f", "lavfi", "-i", f"sine=frequency={hz}:duration=1",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
            check=True)
        return str(path), 1.0

    def test_audio_off_is_unchanged(self):
        """The Sensoform path: the same command as before, argument for argument."""
        assert (_cut_command([("a.mp4", 3.5)], Path("out.mp4"), 24)
                == _cut_command([("a.mp4", 3.5)], Path("out.mp4"), 24, audio=False))

    def test_audio_on_maps_and_encodes_a_soundtrack(self):
        cmd = _cut_command([("a.mp4", 3.5), ("b.mp4", 6.0)], Path("out.mp4"), 24, audio=True)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "concat=n=2:v=1:a=1[v][a]" in graph
        assert "[s0][a0][s1][a1]concat" in graph  # concat wants the pairs interleaved
        assert "-an" not in cmd
        assert cmd.count("-map") == 2 and "[a]" in cmd
        assert "aac" in cmd

    def test_audio_trim_matches_the_frame_trim(self):
        """84 frames at 24fps is 3.5s of picture; the sound is cut to the same."""
        cmd = _cut_command([("a.mp4", 3.5)], Path("o.mp4"), 24, audio=True)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "trim=start_frame=0:end_frame=84" in graph
        assert "atrim=end=3.500000" in graph

    def test_final_mp4_really_has_sound(self, tmp_path):
        clips = [self._clip_with_tone(tmp_path / "c0.mp4", "red", 440),
                 self._clip_with_tone(tmp_path / "c1.mp4", "green", 880)]
        out = render_cut(clips, tmp_path / "final.mp4", fps=10, audio=True)
        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True).stdout.split()
        assert len(streams) == 1, "the cut must carry exactly one audio stream"
        duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=duration", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True).stdout.strip())
        assert duration == pytest.approx(2.0, abs=0.2), "sound must span both segments"

    def test_silent_sources_still_render_picture_only(self, tmp_path):
        clips = []
        for i, colour in enumerate(("red", "green")):
            p = tmp_path / f"s{i}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", f"color=c={colour}:s=64x112:d=1:r=10", "-pix_fmt",
                            "yuv420p", str(p)], check=True)
            clips.append((str(p), 1.0))
        out = render_cut(clips, tmp_path / "silent.mp4", fps=10)
        assert subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True).stdout.strip() == ""


class TestEndToEndWithoutAnAPI:
    """The whole runner, injected generator, real ffmpeg, zero credits."""

    def _fake_executor(self, calls):
        def run(payload):
            out = Path(payload["output_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            colour = ("red", "green", "blue", "yellow")[len(calls)]
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                            "-i", f"color=c={colour}:s=64x112:d=1:r=10", "-pix_fmt", "yuv420p",
                            str(out)], check=True)
            calls.append(payload)

            class R:
                success = True
                error = None
                data = {"job_id": f"job{len(calls)}", "reported_cost_usd": 0.2}
            return R()
        return run

    @pytest.fixture
    def plan(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        shots = tuple(
            Shot(CausalAct(id=f"S{i}", prompt=f"action {i}", duration_seconds=1,
                           carries_identity_from=(f"S{i - 1}" if i > 1 else None)),
                 beat=b)
            for i, b in enumerate(("HOOK", "ESCALATION", "TRANSFORMATION", "PAYOFF"), start=1)
        )
        return ProductionPlan(project_id="e2e", title="E2E", concept="c", shots=shots, fps=10)

    def test_runs_plan_to_final_mp4(self, plan):
        calls = []
        result = execute(plan, approved=True, executor=self._fake_executor(calls))
        assert result.ok is True
        assert len(calls) == 4
        assert Path(result.output).exists()
        assert result.cost_usd == pytest.approx(0.8)

    def test_frames_are_threaded_between_acts_without_the_user(self, plan):
        calls = []
        execute(plan, approved=True, executor=self._fake_executor(calls))
        assert "first_frame" not in calls[0]
        for call, prev in zip(calls[1:], ["S1", "S2", "S3"]):
            frame = Path(call["first_frame"])
            assert frame.name == f"{prev}_last.jpg"
            assert frame.exists(), "the runner must extract the frame, not just name it"

    def test_writes_a_production_report(self, plan):
        import json
        result = execute(plan, approved=True, executor=self._fake_executor([]))
        body = json.loads(Path(result.report).read_text(encoding="utf-8"))
        assert body["status"] == "completed"
        assert body["generations"] == 4
        assert body["billed_cost_usd"] == pytest.approx(0.8)
        assert len(body["acts"]) == 4
        assert "NEEDS_AGENT_REVIEW" in body["qa_caveat"]

    def test_qa_evidence_is_collected_for_every_boundary(self, plan):
        result = execute(plan, approved=True, executor=self._fake_executor([]))
        assert [(b["from_act"], b["to_act"]) for b in result.qa["boundaries"]] == [
            ("S1", "S2"), ("S2", "S3"), ("S3", "S4")]
        assert result.qa["verdict"].startswith("EVIDENCE_ONLY")

    def test_a_failed_act_stops_the_run_before_the_edit(self, plan):
        import json

        def failing(payload):
            class R:
                success = False
                error = "provider said no"
                data = {}
            return R()

        result = execute(plan, approved=True, executor=failing)
        assert result.ok is False
        assert result.output is None
        assert not plan.output_path.exists()
        assert json.loads(Path(result.report).read_text(encoding="utf-8"))["status"] == "failed"

    def test_workspace_is_initialised_for_the_board(self, plan):
        execute(plan, approved=True, executor=self._fake_executor([]))
        assert (plan.project_dir / "project.json").exists()
        assert (plan.project_dir / "artifacts").is_dir()


class TestPlanShape:
    def test_cut_seconds_only_trims_the_edit_never_the_generation(self):
        plan = _plan()
        assert plan.generated_seconds == 24
        assert plan.edit_seconds == 23.0
        assert [p.payload["duration"] for p in plan_payloads(plan)] == [4, 6, 6, 8]

    def test_a_shot_without_a_cut_uses_its_whole_clip(self):
        shot = Shot(CausalAct(id="S1", prompt="x", duration_seconds=6), beat="HOOK")
        assert shot.edit_seconds == 6.0
