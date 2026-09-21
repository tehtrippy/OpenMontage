"""Causal chain orchestration + the frame-conditioning payload contract.

The wire shape here was established the expensive way: arm B of the H3 A/B
experiment failed 400 on a malformed frame_images entry. These tests pin the
shape the API actually validates so that never costs anything again.

No network. No paid call. The chain executor is injected.

Run: pytest tests/tools/test_causal_chain.py -v
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lib.causal_chain import (
    attach_entity_references,
    identity_mechanisms,
    DEFAULT_ACT_SECONDS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    CausalAct,
    build_chain_payloads,
    estimate_chain_cost,
    extract_final_frame,
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
        """1080p + 9:16 IS 1080x1920. The provider takes the tier, not the size —
        see TestResolutionIsSentAsAProviderTier in test_openrouter_video.py."""
        p = OpenRouterVideo()._build_payload({"prompt": "x"})
        assert p["resolution"] == "1080p"
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
        assert (DEFAULT_RESOLUTION, DEFAULT_ASPECT_RATIO, DEFAULT_ACT_SECONDS) == ("1080p", "9:16", 5)

    def test_chain_default_resolution_is_a_provider_tier(self):
        """The chain plans the payload the adapter sends; a pixel size 400s."""
        assert DEFAULT_RESOLUTION in {"360p", "480p", "720p", "768p", "1080p", "1K", "2K", "4K"}
        assert OpenRouterVideo()._build_payload(
            {"prompt": "x", "resolution": DEFAULT_RESOLUTION})["resolution"] == DEFAULT_RESOLUTION


class TestIdentityMechanismSelection:
    """Capability-driven, model-agnostic: nothing here knows a model name.

    The rule that matters is the pessimistic one - an unadvertised capability
    reads as unavailable. Assuming reference support is how three chained acts
    ended up with three different tools while everyone believed identity was
    handled.
    """

    VEO_LISTING = {
        "id": "google/veo-3.1-lite",
        "supported_resolutions": ["720p", "1080p"],
        "supported_frame_images": ["first_frame", "last_frame"],
        "generate_audio": True,
        "seed": True,
    }
    FIRST_ONLY_LISTING = {"id": "some/model", "supported_frame_images": ["first_frame"]}
    NO_FRAMES_LISTING = {"id": "other/model", "supported_frame_images": None}

    def test_listing_drives_the_answer(self):
        m = identity_mechanisms("google/veo-3.1-lite", self.VEO_LISTING)
        assert m["first_frame"] is True
        assert m["last_frame"] is True
        assert m["interpolation_pair"] is True
        assert m["prompt_constraints"] is True

    def test_reference_support_is_false_unless_declared(self):
        """The live listing has no reference field at all, for any model."""
        assert identity_mechanisms("google/veo-3.1-lite", self.VEO_LISTING)["input_references"] is False

    def test_reference_support_is_honoured_when_a_provider_declares_it(self):
        listing = {**self.VEO_LISTING, "supported_reference_images": ["identity"]}
        assert identity_mechanisms("any/model", listing)["input_references"] is True

    def test_first_frame_only_model_has_no_interpolation_pair(self):
        m = identity_mechanisms("some/model", self.FIRST_ONLY_LISTING)
        assert m["first_frame"] is True
        assert m["last_frame"] is False
        assert m["interpolation_pair"] is False

    def test_model_without_frame_support(self):
        m = identity_mechanisms("other/model", self.NO_FRAMES_LISTING)
        assert m["first_frame"] is False and m["last_frame"] is False
        assert m["prompt_constraints"] is True

    def test_without_a_listing_nothing_unadvertised_is_assumed(self):
        m = identity_mechanisms("google/veo-3.1-lite")
        assert m["last_frame"] is False
        assert m["input_references"] is False
        assert m["interpolation_pair"] is False
        assert m["prompt_constraints"] is True

    def test_prompt_constraints_are_always_available(self):
        for caps in (None, self.VEO_LISTING, self.FIRST_ONLY_LISTING, self.NO_FRAMES_LISTING):
            assert identity_mechanisms("any/model", caps)["prompt_constraints"] is True


class TestEntityIdentityReferences:
    """Chaining carries only what the handoff frame SHOWS.

    2026-09-21: across A2/A3/A4 the muddler became a toothed crown, then a flat
    puck, then a perforated barrel, while glass, board, hands, camera and
    lighting held perfectly. Its head was submerged in pulp in both handoff
    frames, so the geometry was never in the pixels being handed forward. Fine
    geometry that is occluded at the handoff needs an explicit anchor.
    """

    BIBLE = {
        "version": "1.0",
        "entities": [
            {"entity_id": "muddler", "type": "tool",
             "visual_description": "stainless muddler, toothed crown head",
             "reference_images": ["refs/muddler.jpg"]},
            {"entity_id": "working_glass", "type": "container",
             "visual_description": "faceted pint glass, black silicone base",
             "reference_images": ["refs/glass.jpg"]},
            {"entity_id": "board", "type": "work_surface",
             "visual_description": "walnut end-grain board"},
        ],
    }

    def test_default_act_has_no_references(self):
        assert CausalAct(id="A1", prompt="x").reference_images == ()

    def test_empty_references_emit_no_input_references(self):
        p = build_chain_payloads([CausalAct(id="A1", prompt="x")], "/tmp/out")[0].payload
        assert "input_references" not in p

    def test_references_are_propagated_into_the_payload(self):
        act = CausalAct(id="A1", prompt="x", reference_images=("refs/muddler.jpg",))
        p = build_chain_payloads([act], "/tmp/out")[0].payload
        assert p["input_references"] == ["refs/muddler.jpg"]

    def test_references_ride_along_with_chaining(self):
        """Anchors and first_frame are complementary, not alternatives."""
        acts = [CausalAct(id="A1", prompt="a"),
                CausalAct(id="A2", prompt="b", reference_images=("refs/muddler.jpg",))]
        steps = build_chain_payloads(acts, "/tmp/out")
        assert steps[1].chained is True
        assert steps[1].payload["first_frame"].endswith("A1_last.jpg")
        assert steps[1].payload["input_references"] == ["refs/muddler.jpg"]

    def test_entity_bible_references_attach_to_an_act(self):
        act = attach_entity_references(CausalAct(id="A2", prompt="x"), self.BIBLE, ["muddler"])
        assert act.reference_images == ("refs/muddler.jpg",)
        assert build_chain_payloads([act], "/tmp/out")[0].payload["input_references"] ==             ["refs/muddler.jpg"]

    def test_several_entities_accumulate_in_order(self):
        act = attach_entity_references(CausalAct(id="A2", prompt="x"), self.BIBLE,
                                       ["muddler", "working_glass"])
        assert act.reference_images == ("refs/muddler.jpg", "refs/glass.jpg")

    def test_attaching_twice_does_not_duplicate(self):
        act = CausalAct(id="A2", prompt="x")
        once = attach_entity_references(act, self.BIBLE, ["muddler"])
        twice = attach_entity_references(once, self.BIBLE, ["muddler"])
        assert twice.reference_images == once.reference_images == ("refs/muddler.jpg",)

    def test_entity_without_reference_images_contributes_nothing(self):
        act = attach_entity_references(CausalAct(id="A2", prompt="x"), self.BIBLE, ["board"])
        assert act.reference_images == ()

    def test_unknown_entity_id_raises_rather_than_generating_unanchored(self):
        with pytest.raises(KeyError, match="muddlerr"):
            attach_entity_references(CausalAct(id="A2", prompt="x"), self.BIBLE, ["muddlerr"])

    def test_attach_does_not_mutate_the_original_act(self):
        act = CausalAct(id="A2", prompt="x")
        attach_entity_references(act, self.BIBLE, ["muddler"])
        assert act.reference_images == ()


class TestLastFrameOnActs:
    """Closing-frame pins are opt-in and never disturb the chain.

    first_frame is derived by the chain; last_frame is always an explicit
    production decision, so it is set on the act and nowhere else.
    """

    def test_default_act_has_no_last_frame(self):
        assert CausalAct(id="A1", prompt="x").last_frame is None

    def test_absent_last_frame_emits_nothing(self):
        p = build_chain_payloads([CausalAct(id="A1", prompt="x")], "/tmp/out")[0].payload
        assert "last_frame" not in p

    def test_last_frame_without_a_first_frame_is_refused_at_plan_time(self):
        """Confirmed provider behaviour: last_frame is an interpolation endpoint.

        The chain seed has nothing to chain from, so an act that pins only its
        close would be a provider-invalid job. Caught before the run, not during.
        """
        act = CausalAct(id="A1", prompt="x", last_frame="frames/target.jpg")
        with pytest.raises(ValueError, match="interpolation endpoint"):
            build_chain_payloads([act], "/tmp/out")

    def test_last_frame_is_refused_when_chaining_is_off(self):
        acts = [CausalAct(id="A1", prompt="a"),
                CausalAct(id="A2", prompt="b", last_frame="frames/target.jpg")]
        with pytest.raises(ValueError, match="interpolation endpoint"):
            build_chain_payloads(acts, "/tmp/out", chain=False)

    def test_last_frame_coexists_with_chained_first_frame(self):
        acts = [CausalAct(id="A1", prompt="a"),
                CausalAct(id="A2", prompt="b", last_frame="frames/target.jpg")]
        p = build_chain_payloads(acts, "/tmp/out")[1].payload
        assert p["first_frame"].endswith("A1_last.jpg")
        assert p["last_frame"] == "frames/target.jpg"

    def test_existing_chain_payloads_are_unchanged(self):
        """Backward compatibility: an act written before these fields existed."""
        p = build_chain_payloads([CausalAct(id="A1", prompt="x", negative_prompt="no text")],
                                 "/tmp/out")[0].payload
        assert set(p) == {"prompt", "model", "duration", "resolution", "aspect_ratio",
                          "generate_audio", "output_path", "negative_prompt"}

    def test_positional_construction_still_works(self):
        """New fields are appended with defaults, so existing positional calls hold."""
        act = CausalAct("A1", "prompt text", "negatives", 6, "A0")
        assert (act.duration_seconds, act.carries_identity_from) == (6, "A0")
        assert act.reference_images == () and act.last_frame is None


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
        """Explicit frame_images entries keep their type.

        Paired with a first_frame because veo treats last_frame as an
        interpolation endpoint and the adapter now refuses it alone.
        """
        p = OpenRouterVideo()._build_payload({
            "prompt": "x",
            "frame_images": [{"frame_type": "first_frame", "image_url": "https://e/a.jpg"},
                             {"frame_type": "last_frame", "image_url": "https://e/f.jpg"}]})
        assert [e["frame_type"] for e in p["frame_images"]] == ["first_frame", "last_frame"]

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
        assert p["resolution"] == "1080p" and p["aspect_ratio"] == "9:16"
        assert p["duration"] == 5 and p["generate_audio"] is False

    def test_cost_is_knowable_before_spending(self):
        """3 acts x 5s at the 1080p audio-off SKU ($0.05/s)."""
        assert estimate_chain_cost(ACTS, model="google/veo-3.1-lite") == pytest.approx(0.75)


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


# --------------------------------------------------------------------------
# Final-frame extraction
# --------------------------------------------------------------------------

def _ramp_clip(path, n_frames: int = 24, fps: int = 24) -> list[int]:
    """Encode a clip whose every frame is a flat grey of a distinct value.

    Frame i is grey 10*(i+1), so identifying which frame came back is a single
    mean(). Near-lossless (qp 0, yuv444p) so the value survives the round trip.
    Returns the per-frame grey values, last element = the true final frame.
    """
    import subprocess as sp
    from PIL import Image

    src = Path(path).parent / "src"
    src.mkdir(parents=True, exist_ok=True)
    values = [10 * (i + 1) for i in range(n_frames)]
    for i, v in enumerate(values):
        Image.new("RGB", (160, 288), (v, v, v)).save(src / f"{i:04d}.png")
    sp.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps),
            "-i", str(src / "%04d.png"), "-c:v", "libx264", "-qp", "0",
            "-pix_fmt", "yuv444p", str(path)], check=True)
    return values


def _mean_grey(img_path) -> float:
    import numpy as np
    from PIL import Image
    return float(np.asarray(Image.open(img_path).convert("L"), dtype=float).mean())


ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


@ffmpeg_required
class TestFinalFrameExtraction:
    """The A1 diagnostic on 2026-09-21 handed A2 a frame 0.04s early.

    On that clip the difference was the gloved hand sitting 11px above the rim
    instead of 379px clear of it — read as a generation defect and rerolled at
    $0.20 before the extraction was found to be the actual cause. These pin the
    frame the chain hands off.
    """

    def test_extracts_the_true_final_frame(self, tmp_path):
        clip = tmp_path / "ramp.mp4"
        values = _ramp_clip(clip)
        out = extract_final_frame(clip, tmp_path / "frames" / "last.jpg")
        assert out.exists()
        assert _mean_grey(out) == pytest.approx(values[-1], abs=4)

    def test_does_not_return_the_frame_sseof_would_have_given(self, tmp_path):
        """Behavioural guard: the old command must disagree with the new one on
        this fixture, otherwise the fixture proves nothing."""
        import subprocess as sp
        clip = tmp_path / "ramp.mp4"
        values = _ramp_clip(clip)

        old = tmp_path / "old_sseof.jpg"
        sp.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.1",
                "-i", str(clip), "-frames:v", "1", "-q:v", "3", str(old)], check=True)
        old_grey = _mean_grey(old)

        new_grey = _mean_grey(extract_final_frame(clip, tmp_path / "new.jpg"))
        assert old_grey < values[-1] - 5, "fixture does not discriminate; old path already landed last"
        assert new_grey == pytest.approx(values[-1], abs=4)
        assert abs(new_grey - old_grey) > 5

    def test_extraction_command_carries_no_seek_flag(self, monkeypatch, tmp_path):
        """Structural guard so the old behaviour cannot be reintroduced by an
        edit that still happens to pass the behavioural test on a short clip."""
        seen = {}

        def fake_run(cmd, *a, **k):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"\xff\xd8\xff" + b"x" * 64)
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr("lib.causal_chain.subprocess.run", fake_run)
        extract_final_frame(tmp_path / "in.mp4", tmp_path / "out.jpg")
        cmd = seen["cmd"]
        assert "-sseof" not in cmd
        assert "-ss" not in cmd
        assert "-update" in cmd, "must decode through to the last frame"
        assert "-frames:v" not in cmd, "-frames:v 1 with -update would pin the FIRST frame"

    def test_empty_output_is_still_an_error(self, monkeypatch, tmp_path):
        def fake_run(cmd, *a, **k):
            Path(cmd[-1]).write_bytes(b"")
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr("lib.causal_chain.subprocess.run", fake_run)
        with pytest.raises(RuntimeError, match="final-frame extraction produced nothing"):
            extract_final_frame(tmp_path / "in.mp4", tmp_path / "out.jpg")

    def test_chain_hands_the_true_final_frame_to_the_next_act(self, tmp_path):
        """End to end through generate_chain with the real extractor: act 2's
        first_frame must be act 1's last frame, not a frame near its end."""
        values = None

        def executor(payload):
            nonlocal values
            values = _ramp_clip(Path(payload["output_path"]))
            return FakeResult()

        res = generate_chain(ACTS[:2], tmp_path, dry_run=False, executor=executor)
        assert res.ok
        handoff = tmp_path / "frames" / "A1_last.jpg"
        assert handoff.exists()
        assert _mean_grey(handoff) == pytest.approx(values[-1], abs=4)
