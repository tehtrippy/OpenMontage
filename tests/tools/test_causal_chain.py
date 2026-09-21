"""Causal chain orchestration + the frame-conditioning payload contract.

The wire shape here was established the expensive way: arm B of the H3 A/B
experiment failed 400 on a malformed frame_images entry. These tests pin the
shape the API actually validates so that never costs anything again.

No network. No paid call. The chain executor is injected.

Run: pytest tests/tools/test_causal_chain.py -v
"""

from __future__ import annotations

import pytest

from lib.causal_chain import (
    DEFAULT_ACT_SECONDS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    CausalAct,
    build_chain_payloads,
    estimate_chain_cost,
    generate_chain,
)
from tools.video.openrouter_video import OpenRouterVideo

ACTS = [
    CausalAct(id="A1", prompt="A gloved hand places an empty glass, then fills it.", negative_prompt="no text"),
    CausalAct(id="A2", prompt="The same hand crushes the berries already in that same glass.",
              carries_identity_from="A1"),
    CausalAct(id="A3", prompt="The same hand tips raspberries into that same glass.",
              carries_identity_from="A2"),
]


class FakeResult:
    def __init__(self, success=True, job_id="job", cost=0.12, error=None):
        self.success = success
        self.error = error
        self.data = {"job_id": job_id, "reported_cost_usd": cost} if success else {}


class TestGenerationDefaults:
    def test_adapter_default_target_is_1080x1920(self):
        p = OpenRouterVideo()._build_payload({"prompt": "x"})
        assert p["resolution"] == "1080x1920"
        assert p["aspect_ratio"] == "9:16"

    def test_adapter_default_act_is_five_seconds(self):
        assert OpenRouterVideo()._build_payload({"prompt": "x"})["duration"] == 5

    def test_audio_off_by_default(self):
        assert OpenRouterVideo()._build_payload({"prompt": "x"})["generate_audio"] is False

    def test_two_k_is_optional_not_default(self):
        p = OpenRouterVideo()._build_payload({"prompt": "x", "resolution": "2K"})
        assert p["resolution"] == "2K"
        assert OpenRouterVideo()._build_payload({"prompt": "x"})["resolution"] != "2K"

    def test_chain_defaults_match_the_adapter(self):
        assert (DEFAULT_RESOLUTION, DEFAULT_ASPECT_RATIO, DEFAULT_ACT_SECONDS) == ("1080x1920", "9:16", 5)


class TestFrameImagePayloadShape:
    """The shape the live API validated. Legacy forms must be rewritten, not sent."""

    def test_correct_wire_shape_is_produced(self):
        p = OpenRouterVideo()._build_payload({"prompt": "x", "first_frame": "https://e/f.jpg"})
        assert p["frame_images"] == [{
            "type": "image_url",
            "image_url": {"url": "https://e/f.jpg"},
            "frame_type": "first_frame",
        }]

    def test_legacy_shape_is_rewritten_not_emitted(self):
        """{'type': 'first_frame', 'image_url': '<str>'} is a 400. It must not survive."""
        p = OpenRouterVideo()._build_payload(
            {"prompt": "x", "frame_images": [{"type": "first_frame", "image_url": "https://e/f.jpg"}]})
        entry = p["frame_images"][0]
        assert entry["type"] == "image_url"            # not "first_frame"
        assert isinstance(entry["image_url"], dict)    # not a bare string
        assert entry["frame_type"] == "first_frame"

    def test_bare_string_entry_is_accepted(self):
        p = OpenRouterVideo()._build_payload({"prompt": "x", "frame_images": ["https://e/f.jpg"]})
        assert p["frame_images"][0]["image_url"] == {"url": "https://e/f.jpg"}

    def test_last_frame_type_is_preserved(self):
        p = OpenRouterVideo()._build_payload(
            {"prompt": "x", "frame_images": [{"frame_type": "last_frame", "image_url": "https://e/f.jpg"}]})
        assert p["frame_images"][0]["frame_type"] == "last_frame"

    def test_local_path_becomes_a_data_uri(self):
        """Must be inside the working tree — see the exfiltration guard."""
        import os
        from pathlib import Path as _P
        img = _P(os.getcwd()) / "_test_frame.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)
        try:
            p = OpenRouterVideo()._build_payload({"prompt": "x", "first_frame": str(img)})
            assert p["frame_images"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            img.unlink()

    def test_missing_local_file_inside_the_tree_raises(self):
        import os
        from pathlib import Path as _P
        missing = _P(os.getcwd()) / "_test_absent_frame.jpg"
        assert not missing.exists()
        with pytest.raises(FileNotFoundError):
            OpenRouterVideo()._build_payload({"prompt": "x", "first_frame": str(missing)})

    def test_path_outside_the_tree_is_refused_before_it_is_read(self):
        """Containment is checked before existence, so a probe for a file
        outside the tree cannot even confirm whether it exists."""
        with pytest.raises(ValueError, match="under the working directory"):
            OpenRouterVideo()._build_payload({"prompt": "x", "first_frame": "/etc/hosts"})

    def test_bad_frame_type_raises(self):
        with pytest.raises(ValueError):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "frame_images": [{"frame_type": "middle_frame", "image_url": "https://e/f.jpg"}]})

    def test_no_frame_images_key_when_unchained(self):
        assert "frame_images" not in OpenRouterVideo()._build_payload({"prompt": "x"})


class TestChainPlanning:
    def test_first_act_is_never_chained(self):
        steps = build_chain_payloads(ACTS, "/tmp/out")
        assert steps[0].chained is False
        assert "first_frame" not in steps[0].payload

    def test_each_later_act_chains_from_its_predecessor(self):
        steps = build_chain_payloads(ACTS, "/tmp/out")
        for prev, step in zip(ACTS, steps[1:]):
            assert step.chained is True
            assert step.payload["first_frame"].endswith(f"{prev.id}_last.jpg")

    def test_chain_can_be_disabled_for_an_ab_control(self):
        steps = build_chain_payloads(ACTS, "/tmp/out", chain=False)
        assert not any(s.chained for s in steps)

    def test_plan_carries_the_defaults(self):
        p = build_chain_payloads(ACTS, "/tmp/out")[0].payload
        assert p["resolution"] == "1080x1920" and p["aspect_ratio"] == "9:16"
        assert p["duration"] == 5 and p["generate_audio"] is False

    def test_cost_is_knowable_before_spending(self):
        assert estimate_chain_cost(ACTS, model="google/veo-3.1-lite") == pytest.approx(0.45)


class TestChainExecution:
    def test_dry_run_is_the_default_and_calls_nothing(self):
        calls = []
        res = generate_chain(ACTS, "/tmp/out", executor=lambda p: calls.append(p))
        assert res.dry_run is True
        assert calls == []
        assert len(res.steps) == 3

    def test_previous_final_frame_is_passed_to_the_next_act(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr("lib.causal_chain.extract_final_frame",
                            lambda v, o: __import__("pathlib").Path(o))

        def executor(payload):
            seen.append(payload)
            return FakeResult()

        res = generate_chain(ACTS, tmp_path, dry_run=False, executor=executor)
        assert len(seen) == 3
        assert "first_frame" not in seen[0]
        assert seen[1]["first_frame"].endswith("A1_last.jpg")
        assert seen[2]["first_frame"].endswith("A2_last.jpg")
        assert res.total_cost_usd == pytest.approx(0.36)
        assert res.ok

    def test_a_failed_act_stops_the_chain(self, tmp_path, monkeypatch):
        """No final frame means no chain. Continuing would silently emit an
        unchained clip and destroy the property being tested."""
        monkeypatch.setattr("lib.causal_chain.extract_final_frame",
                            lambda v, o: __import__("pathlib").Path(o))
        calls = {"n": 0}

        def executor(payload):
            calls["n"] += 1
            return FakeResult(success=False, error="boom") if calls["n"] == 2 else FakeResult()

        res = generate_chain(ACTS, tmp_path, dry_run=False, executor=executor)
        assert calls["n"] == 2          # third act never attempted
        assert res.ok is False


class TestBackwardCompatibility:
    def test_existing_callers_without_frames_are_unaffected(self):
        p = OpenRouterVideo()._build_payload({"prompt": "hello", "model": "google/veo-3.1-lite",
                                              "duration": 4, "resolution": "720p", "aspect_ratio": "16:9"})
        assert p == {"model": "google/veo-3.1-lite", "prompt": "hello", "duration": 4,
                     "resolution": "720p", "aspect_ratio": "16:9", "generate_audio": False}

    def test_negative_prompt_folding_still_applies(self):
        p = OpenRouterVideo()._build_payload({"prompt": "a glass", "negative_prompt": "no text"})
        assert p["prompt"] == "a glass Negative constraints: no text"
        assert "negative_prompt" not in p


class TestHardening:
    """Regressions from the pre-push review of 920701c."""

    def test_act_id_cannot_escape_the_output_directory(self):
        from lib.causal_chain import CausalAct, build_chain_payloads
        with pytest.raises(ValueError, match="must match"):
            build_chain_payloads([CausalAct(id="../../etc/x", prompt="p")], "/tmp/out")

    def test_extraction_failure_keeps_the_already_billed_result(self, tmp_path, monkeypatch):
        """The clip is generated and paid for. An ffmpeg failure must not throw
        the ChainResult (and the running cost) away."""
        def boom(_v, _o):
            raise RuntimeError("ffmpeg exploded")

        monkeypatch.setattr("lib.causal_chain.extract_final_frame", boom)
        res = generate_chain(ACTS, tmp_path, dry_run=False, executor=lambda p: FakeResult())
        assert res.total_cost_usd == pytest.approx(0.12)   # act 1 still accounted for
        assert "final-frame extraction failed" in res.steps[0].error
        assert res.steps[1].success is None                # act 2 never attempted

    def test_frame_image_outside_the_working_tree_is_refused(self, tmp_path):
        outside = tmp_path / "secret.jpg"
        outside.write_bytes(b"\xff\xd8\xff" + b"x" * 32)
        with pytest.raises(ValueError, match="under the working directory"):
            OpenRouterVideo()._build_payload({"prompt": "x", "first_frame": str(outside)})

    def test_non_image_bytes_are_refused_even_with_an_image_extension(self):
        """A secret renamed to .jpg must not be base64'd out to a third party."""
        import os
        from pathlib import Path as _P
        fake = _P(os.getcwd()) / "_review_not_an_image.jpg"
        fake.write_text("OPENROUTER_API_KEY=sk-or-totally-not-an-image")
        try:
            with pytest.raises(ValueError, match="not a recognised image format"):
                OpenRouterVideo()._build_payload({"prompt": "x", "first_frame": str(fake)})
        finally:
            fake.unlink()

    def test_bad_frame_input_returns_a_tool_result_not_a_traceback(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        r = OpenRouterVideo().execute({"prompt": "x", "first_frame": "/nope/missing.jpg"})
        assert r.success is False
        assert "invalid generation inputs" in r.error
