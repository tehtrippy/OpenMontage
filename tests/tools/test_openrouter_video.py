"""Behavioral tests for the OpenRouter video adapter with a faked `requests` module.

Covers the submit -> poll -> download cycle, the authenticated download (OpenRouter's
content URLs are unsigned and still need the bearer header), cost estimation, and the
error paths. No network access and no API key required.

Run: pytest tests/tools/test_openrouter_video.py -v
"""

from __future__ import annotations

import sys
import types

import pytest

from tools.video.openrouter_video import OpenRouterVideo


class FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200, text=""):
        self._payload = payload if payload is not None else {}
        self.content = content
        self.status_code = status_code
        self.text = text or str(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_requests(monkeypatch):
    """Install a fake `requests` module and record every call made through it."""
    calls = {"post": [], "get": []}
    queues = {"post": [], "get": []}

    def fake_post(url, **kwargs):
        calls["post"].append({"url": url, **kwargs})
        if not queues["post"]:
            raise AssertionError(f"Unexpected POST to {url}")
        return queues["post"].pop(0)

    def fake_get(url, **kwargs):
        calls["get"].append({"url": url, **kwargs})
        if not queues["get"]:
            raise AssertionError(f"Unexpected GET to {url}")
        return queues["get"].pop(0)

    module = types.ModuleType("requests")
    module.post = fake_post
    module.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", module)
    # Polling sleeps 10s between attempts; tests must not actually wait.
    monkeypatch.setattr("tools.video.openrouter_video.time.sleep", lambda _s: None)
    return {"calls": calls, "queues": queues}


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class TestAvailability:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert OpenRouterVideo().get_status().value == "unavailable"

    def test_available_with_key(self, api_key):
        assert OpenRouterVideo().get_status().value == "available"

    def test_execute_refuses_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = OpenRouterVideo().execute({"prompt": "a berry"})
        assert result.success is False
        assert "OPENROUTER_API_KEY" in result.error

    def test_declares_env_dependency(self):
        assert "env:OPENROUTER_API_KEY" in OpenRouterVideo().dependencies

    def test_registers_as_a_video_generation_provider(self):
        tool = OpenRouterVideo()
        assert tool.capability == "video_generation"
        assert tool.provider == "openrouter"


class TestCostEstimate:
    def test_default_model_rate(self):
        """Defaults are 1080p audio-off, which bills $0.05/s => $0.20 for 4s.

        Measured on a real job. The earlier $0.12 was the 720p SKU applied to
        every resolution, so the default path was under-quoted by 67%.
        """
        assert OpenRouterVideo().estimate_cost({"prompt": "x", "duration": 4}) == 0.20

    def test_resolution_moves_the_estimate(self):
        tool = OpenRouterVideo()
        base = {"prompt": "x", "model": "google/veo-3.1-lite", "duration": 4}
        assert tool.estimate_cost({**base, "resolution": "720p"}) == 0.12
        assert tool.estimate_cost({**base, "resolution": "1080p"}) == 0.20
        assert tool.estimate_cost({**base, "resolution": "1080x1920"}) == 0.20

    def test_audio_moves_the_estimate(self):
        tool = OpenRouterVideo()
        base = {"prompt": "x", "model": "google/veo-3.1-lite", "duration": 4}
        assert tool.estimate_cost({**base, "resolution": "1080p", "generate_audio": True}) == 0.32
        assert tool.estimate_cost({**base, "resolution": "720p", "generate_audio": True}) == 0.20

    def test_unknown_tier_uses_the_dearest_known_sku(self):
        """An estimate that is too low is the one that causes harm."""
        tool = OpenRouterVideo()
        assert tool.estimate_cost(
            {"prompt": "x", "model": "google/veo-3.1-lite", "duration": 4, "resolution": "4K"}) == 0.20

    def test_model_without_published_skus_uses_the_flat_table(self):
        assert OpenRouterVideo().estimate_cost(
            {"prompt": "x", "model": "bytedance/seedance-2.0-mini", "duration": 4}) == round(0.03363 * 4, 4)

    def test_720p_rate_matches_the_measured_charge(self):
        """Pins the fallback rate to a real billed job.

        2026-09-21: a 4s / 720p / 9:16 / audio-off generation on
        google/veo-3.1-lite returned usage.cost == 0.12. The estimate must not
        drift away from the only charge we have actually observed.
        """
        estimate = OpenRouterVideo().estimate_cost(
            {"prompt": "x", "model": "google/veo-3.1-lite", "duration": 4,
             "resolution": "720p", "aspect_ratio": "9:16", "generate_audio": False}
        )
        assert estimate == 0.12

    def test_four_act_plan_total(self):
        """Four 4s acts at the default 1080p audio-off SKU: 4 x $0.20."""
        tool = OpenRouterVideo()
        act = {"prompt": "x", "model": "google/veo-3.1-lite", "duration": 4}
        assert round(tool.estimate_cost(act) * 4, 2) == 0.80

    def test_known_model_rates_differ(self):
        tool = OpenRouterVideo()
        cheap = tool.estimate_cost({"prompt": "x", "model": "bytedance/seedance-2.0-mini", "duration": 4})
        dear = tool.estimate_cost({"prompt": "x", "model": "bytedance/seedance-2.5", "duration": 4})
        assert cheap < dear

    def test_unknown_model_uses_fallback_rate(self):
        assert OpenRouterVideo().estimate_cost(
            {"prompt": "x", "model": "someone/not-in-the-table", "duration": 10}
        ) == 1.0


def _png(path):
    """Smallest real PNG: the guard checks magic bytes, not the extension."""
    import base64
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="))
    return path


class TestInputReferenceImages:
    """Identity anchors for what the handoff frame cannot carry.

    2026-09-21: the muddler head was submerged in pulp at every handoff, so
    three chained acts invented three different tools while glass, board, hands
    and camera held. References are the third leg - causal prompt, frame chain,
    identity anchor.

    They were previously forwarded VERBATIM, so a local path went out as a raw
    string and skipped the working-tree / magic-byte / size guards every other
    image input gets. Same bytes leave the machine, same guard.
    """

    def test_absent_when_not_supplied(self):
        assert "input_references" not in OpenRouterVideo()._build_payload({"prompt": "x"})

    def test_empty_list_is_omitted_not_nulled(self):
        payload = OpenRouterVideo()._build_payload({"prompt": "x", "input_references": []})
        assert "input_references" not in payload

    def test_https_url_is_preserved(self):
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "input_references": ["https://example.com/muddler.jpg"]})
        assert payload["input_references"] == [
            {"type": "image_url", "image_url": {"url": "https://example.com/muddler.jpg"}}]

    def test_data_uri_is_preserved(self):
        uri = "data:image/jpeg;base64,AAAA"
        payload = OpenRouterVideo()._build_payload({"prompt": "x", "input_references": [uri]})
        assert payload["input_references"][0]["image_url"]["url"] == uri

    def test_local_path_becomes_a_data_uri(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ref = _png(tmp_path / "refs" / "muddler.png")
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "input_references": [str(ref)]})
        entry = payload["input_references"][0]
        assert entry["type"] == "image_url"
        assert entry["image_url"]["url"].startswith("data:image/png;base64,")

    def test_dict_entries_keep_their_extra_keys(self):
        payload = OpenRouterVideo()._build_payload({"prompt": "x", "input_references": [
            {"image_url": "https://example.com/m.jpg", "weight": 0.8}]})
        assert payload["input_references"] == [
            {"type": "image_url", "weight": 0.8,
             "image_url": {"url": "https://example.com/m.jpg"}}]

    def test_entry_without_a_url_is_rejected(self):
        with pytest.raises(ValueError, match="no usable url"):
            OpenRouterVideo()._build_payload({"prompt": "x", "input_references": [{"weight": 1}]})

    def test_path_outside_the_working_tree_is_refused(self, tmp_path, monkeypatch):
        """The guard that stops this being a read-anything-and-exfiltrate primitive."""
        outside = _png(tmp_path / "outside" / "secret.png")
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError, match="must live under the working directory"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "input_references": [str(outside)]})

    def test_traversal_out_of_the_working_tree_is_refused(self, tmp_path, monkeypatch):
        _png(tmp_path / "outside" / "secret.png")
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError, match="must live under the working directory"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "input_references": ["../outside/secret.png"]})

    def test_non_image_bytes_are_refused(self, tmp_path, monkeypatch):
        """A secret renamed .jpg is not an image; the extension is not trusted."""
        monkeypatch.chdir(tmp_path)
        fake = tmp_path / "refs" / "env.jpg"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("OPENROUTER_API_KEY=sk-not-a-real-key")
        with pytest.raises(ValueError, match="not a recognised image format"):
            OpenRouterVideo()._build_payload({"prompt": "x", "input_references": [str(fake)]})

    def test_oversized_reference_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        big = tmp_path / "refs" / "big.png"
        big.parent.mkdir(parents=True, exist_ok=True)
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * (12 * 1024 * 1024 + 1))
        with pytest.raises(ValueError, match="too large"):
            OpenRouterVideo()._build_payload({"prompt": "x", "input_references": [str(big)]})

    def test_bad_reference_fails_before_any_submit(self, api_key, fake_requests):
        result = OpenRouterVideo().execute({"prompt": "x", "input_references": [{"weight": 1}]})
        assert result.success is False
        assert "no usable url" in result.error
        assert fake_requests["calls"]["post"] == []

    def test_first_frame_chaining_is_unchanged_by_references(self, tmp_path, monkeypatch):
        """The two mechanisms are independent: frame_type stays on frames only."""
        monkeypatch.chdir(tmp_path)
        frame = _png(tmp_path / "frames" / "A2_last.png")
        ref = _png(tmp_path / "refs" / "muddler.png")
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "first_frame": str(frame), "input_references": [str(ref)]})
        assert len(payload["frame_images"]) == 1
        assert payload["frame_images"][0]["frame_type"] == "first_frame"
        assert payload["frame_images"][0]["type"] == "image_url"
        assert payload["frame_images"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "frame_type" not in payload["input_references"][0]


class TestDryRunPayloadShape:
    """Prints and pins the exact body an anchored, chained act would send.

    A dry run, never a submit. Run with -s to read it.
    """

    def test_prints_the_anchored_chained_payload(self, tmp_path, monkeypatch, capsys):
        import json
        monkeypatch.chdir(tmp_path)
        first = _png(tmp_path / "frames" / "A2_last.png")
        ref = _png(tmp_path / "refs" / "muddler.png")
        payload = OpenRouterVideo()._build_payload({
            "prompt": "the same gloved hand keeps grinding the same muddler",
            "negative_prompt": "no new tool",
            "model": "google/veo-3.1-lite",
            "duration": 4,
            "resolution": "1080p",
            "aspect_ratio": "9:16",
            "generate_audio": False,
            "first_frame": str(first),
            "input_references": [str(ref)],
        })
        redacted = json.loads(json.dumps(payload))
        for entry in redacted["frame_images"] + redacted["input_references"]:
            entry["image_url"]["url"] = entry["image_url"]["url"][:40] + "...<base64>"
        print("\nDRY RUN - payload that WOULD be sent (not submitted):")
        print(json.dumps(redacted, indent=2))

        assert payload["model"] == "google/veo-3.1-lite"
        assert payload["resolution"] == "1080p" and payload["aspect_ratio"] == "9:16"
        assert payload["duration"] == 4 and payload["generate_audio"] is False
        assert "negative_prompt" not in payload and "Negative constraints:" in payload["prompt"]
        assert payload["frame_images"][0]["frame_type"] == "first_frame"
        assert payload["input_references"][0]["type"] == "image_url"
        assert payload["input_references"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "DRY RUN" in capsys.readouterr().out


class TestModelCapabilityLookup:
    """Capabilities come from the provider, free, and are never guessed."""

    LISTING = {"data": [
        {"id": "google/veo-3.1-lite", "supported_frame_images": ["first_frame", "last_frame"],
         "pricing_skus": {"duration_seconds_without_audio": "0.05"}},
        {"id": "other/model", "supported_frame_images": ["first_frame"]},
    ]}

    def test_fetches_one_model(self, fake_requests, api_key):
        fake_requests["queues"]["get"].append(FakeResponse(self.LISTING))
        caps = OpenRouterVideo.fetch_model_capabilities("google/veo-3.1-lite")
        assert caps["supported_frame_images"] == ["first_frame", "last_frame"]
        assert fake_requests["calls"]["get"][0]["url"].endswith("/videos/models")

    def test_unknown_model_returns_none_rather_than_a_guess(self, fake_requests, api_key):
        fake_requests["queues"]["get"].append(FakeResponse(self.LISTING))
        assert OpenRouterVideo.fetch_model_capabilities("nobody/nothing") is None

    def test_returns_the_whole_listing_when_no_model_is_named(self, fake_requests, api_key):
        fake_requests["queues"]["get"].append(FakeResponse(self.LISTING))
        assert len(OpenRouterVideo.fetch_model_capabilities()) == 2

    def test_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OpenRouterVideo.fetch_model_capabilities("google/veo-3.1-lite")

    def test_it_is_not_called_during_generation(self, fake_requests, api_key):
        """Payload building must work offline: capabilities are a planner call."""
        OpenRouterVideo()._build_payload({"prompt": "x"})
        assert fake_requests["calls"]["get"] == []


class TestLastFrameConditioning:
    """Pinning the CLOSING frame - same guards, same wire shape, different frame_type.

    Confirmed semantics, 2026-09-21: on google/veo-3.1-lite last_frame is an
    INTERPOLATION endpoint, not a standalone pin. A last_frame-only job was
    rejected by the provider with "Frame interpolation requires both an input
    image and a last frame." (not billed), so the adapter now refuses that
    combination locally for models with interpolation semantics.

    A model whose semantics have NOT been observed is left alone: last_frame
    alone still builds, because inventing a constraint for an untested model
    would break it for no evidence.
    """

    # Semantics unobserved => the adapter must not impose the Veo rule on it.
    UNTESTED_MODEL = "bytedance/seedance-2.0-fast"

    def test_absent_when_not_supplied(self):
        assert "frame_images" not in OpenRouterVideo()._build_payload({"prompt": "x"})

    def test_last_frame_alone_is_refused_for_interpolation_models(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        end = _png(tmp_path / "frames" / "end.png")
        with pytest.raises(ValueError, match="interpolation endpoint"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "model": "google/veo-3.1-lite", "last_frame": str(end)})

    def test_refusal_names_the_provider_error_and_the_fix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        end = _png(tmp_path / "frames" / "end.png")
        with pytest.raises(ValueError) as excinfo:
            OpenRouterVideo()._build_payload({"prompt": "x", "last_frame": str(end)})
        message = str(excinfo.value)
        assert "requires a first_frame" in message
        assert "Frame interpolation requires both an input image and a last frame." in message
        assert "not an object-identity mechanism" in message

    def test_last_frame_alone_fails_before_any_submit(self, api_key, fake_requests, tmp_path,
                                                      monkeypatch):
        """A provider-invalid job is never sent."""
        monkeypatch.chdir(tmp_path)
        end = _png(tmp_path / "frames" / "end.png")
        result = OpenRouterVideo().execute({"prompt": "x", "last_frame": str(end)})
        assert result.success is False
        assert "interpolation endpoint" in result.error
        assert fake_requests["calls"]["post"] == []

    def test_untested_model_keeps_last_frame_alone(self, tmp_path, monkeypatch):
        """Capability checks gate the rule; they do not generalise it."""
        monkeypatch.chdir(tmp_path)
        end = _png(tmp_path / "frames" / "end.png")
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "model": self.UNTESTED_MODEL, "last_frame": str(end)})
        assert len(payload["frame_images"]) == 1
        entry = payload["frame_images"][0]
        assert entry["type"] == "image_url"
        assert entry["frame_type"] == "last_frame"
        assert entry["image_url"]["url"].startswith("data:image/png;base64,")

    def test_frame_semantics_lookup(self):
        assert OpenRouterVideo.frame_semantics("google/veo-3.1-lite") == "interpolation"
        assert OpenRouterVideo.frame_semantics("google/veo-3.1-fast") == "interpolation"
        assert OpenRouterVideo.frame_semantics(self.UNTESTED_MODEL) is None

    def test_https_url_is_preserved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        start = _png(tmp_path / "frames" / "start.png")
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "first_frame": str(start), "last_frame": "https://example.com/end.jpg"})
        assert payload["frame_images"][1] == {
            "type": "image_url", "frame_type": "last_frame",
            "image_url": {"url": "https://example.com/end.jpg"}}

    def test_data_uri_is_preserved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        start = _png(tmp_path / "frames" / "start.png")
        uri = "data:image/jpeg;base64,AAAA"
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "first_frame": str(start), "last_frame": uri})
        assert payload["frame_images"][1]["image_url"]["url"] == uri

    def test_both_ends_together_keep_their_order_and_types(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        start = _png(tmp_path / "frames" / "start.png")
        end = _png(tmp_path / "frames" / "end.png")
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "first_frame": str(start), "last_frame": str(end)})
        assert [e["frame_type"] for e in payload["frame_images"]] == ["first_frame", "last_frame"]
        assert all(e["type"] == "image_url" for e in payload["frame_images"])

    def test_explicit_frame_images_still_append_after_both(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        start = _png(tmp_path / "frames" / "start.png")
        end = _png(tmp_path / "frames" / "end.png")
        payload = OpenRouterVideo()._build_payload({
            "prompt": "x", "first_frame": str(start), "last_frame": str(end),
            "frame_images": ["https://example.com/extra.jpg"]})
        assert [e["frame_type"] for e in payload["frame_images"]] == [
            "first_frame", "last_frame", "first_frame"]

    def test_out_of_worktree_path_is_refused(self, tmp_path, monkeypatch):
        outside = _png(tmp_path / "outside" / "secret.png")
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError, match="must live under the working directory"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "model": self.UNTESTED_MODEL, "last_frame": str(outside)})

    def test_traversal_is_refused(self, tmp_path, monkeypatch):
        _png(tmp_path / "outside" / "secret.png")
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        with pytest.raises(ValueError, match="must live under the working directory"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "model": self.UNTESTED_MODEL, "last_frame": "../outside/secret.png"})

    def test_non_image_bytes_are_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake = tmp_path / "frames" / "env.jpg"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("OPENROUTER_API_KEY=sk-not-a-real-key")
        with pytest.raises(ValueError, match="not a recognised image format"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "model": self.UNTESTED_MODEL, "last_frame": str(fake)})

    def test_missing_file_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="frame image not found"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "model": self.UNTESTED_MODEL, "last_frame": "frames/nope.png"})

    def test_oversized_image_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        big = tmp_path / "frames" / "big.png"
        big.parent.mkdir(parents=True, exist_ok=True)
        big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * (12 * 1024 * 1024 + 1))
        with pytest.raises(ValueError, match="too large"):
            OpenRouterVideo()._build_payload(
                {"prompt": "x", "model": self.UNTESTED_MODEL, "last_frame": str(big)})

    def test_bad_last_frame_fails_before_any_submit(self, api_key, fake_requests, tmp_path,
                                                    monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = OpenRouterVideo().execute(
            {"prompt": "x", "model": self.UNTESTED_MODEL, "last_frame": "frames/nope.png"})
        assert result.success is False
        assert "frame image not found" in result.error
        assert fake_requests["calls"]["post"] == []

    def test_first_frame_behaviour_is_unchanged(self, tmp_path, monkeypatch):
        """The pre-existing contract, re-pinned now that a sibling input exists."""
        monkeypatch.chdir(tmp_path)
        start = _png(tmp_path / "frames" / "start.png")
        payload = OpenRouterVideo()._build_payload({"prompt": "x", "first_frame": str(start)})
        assert len(payload["frame_images"]) == 1
        assert payload["frame_images"][0]["frame_type"] == "first_frame"
        assert "last_frame" not in payload

    def test_references_do_not_disturb_frame_conditioning(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        start = _png(tmp_path / "frames" / "start.png")
        end = _png(tmp_path / "frames" / "end.png")
        ref = _png(tmp_path / "refs" / "anchor.png")
        payload = OpenRouterVideo()._build_payload({
            "prompt": "x", "first_frame": str(start), "last_frame": str(end),
            "input_references": [str(ref)]})
        assert [e["frame_type"] for e in payload["frame_images"]] == ["first_frame", "last_frame"]
        assert len(payload["input_references"]) == 1
        assert "frame_type" not in payload["input_references"][0]


class TestResolutionIsSentAsAProviderTier:
    """The provider validates `resolution` against a tier enum, not a pixel size.

    The adapter used to default to "1080x1920", so the default-parameter path
    could not generate at all: every submit came back 400 ZodError listing
    360p|480p|720p|768p|1080p|1K|2K|4K. The 2026-09-21 A1 reseed only went
    through after the payload was corrected by hand. These tests pin the wire
    format while the delivery frame (1080x1920 at 9:16) stays what it was.
    """

    # The values the live endpoint's ZodError named, verbatim.
    PROVIDER_ENUM = {"360p", "480p", "720p", "768p", "1080p", "1K", "2K", "4K"}

    def test_default_payload_resolution_is_the_tier_enum(self):
        payload = OpenRouterVideo()._build_payload({"prompt": "x"})
        assert payload["resolution"] == "1080p"
        assert payload["resolution"] in self.PROVIDER_ENUM

    def test_default_payload_still_targets_1080x1920(self):
        """1080p + 9:16 is 1080x1920 — verified against the delivered A1 clip."""
        payload = OpenRouterVideo()._build_payload({"prompt": "x"})
        assert (payload["resolution"], payload["aspect_ratio"]) == ("1080p", "9:16")

    def test_a_pixel_size_never_reaches_the_wire(self):
        for requested in ("1080x1920", "1920x1080", "1080X1920", " 1080 x 1920 "):
            payload = OpenRouterVideo()._build_payload({"prompt": "x", "resolution": requested})
            assert payload["resolution"] == "1080p", requested
            assert "x" not in payload["resolution"].lower()

    def test_other_pixel_sizes_map_by_short_side(self):
        cases = {"1440x2560": "2K", "2560x1440": "2K", "720x1280": "720p",
                 "3840x2160": "4K", "480x854": "480p"}
        for requested, tier in cases.items():
            assert OpenRouterVideo()._normalize_resolution(requested) == tier, requested

    def test_tiers_pass_through_and_are_canonicalised(self):
        assert OpenRouterVideo()._normalize_resolution("2K") == "2K"
        assert OpenRouterVideo()._normalize_resolution("2k") == "2K"
        assert OpenRouterVideo()._normalize_resolution("1080P") == "1080p"

    def test_every_tier_the_provider_named_is_accepted(self):
        for tier in self.PROVIDER_ENUM:
            assert OpenRouterVideo()._normalize_resolution(tier) == tier

    def test_unmappable_size_is_refused_not_silently_downgraded(self):
        with pytest.raises(ValueError, match="not a provider tier"):
            OpenRouterVideo()._normalize_resolution("999x1777")

    def test_unmappable_size_fails_before_any_submit(self, api_key, fake_requests):
        """A structured failure, and no POST — a bad size must not bill."""
        result = OpenRouterVideo().execute({"prompt": "x", "resolution": "999x1777"})
        assert result.success is False
        assert "not a provider tier" in result.error
        assert fake_requests["calls"]["post"] == []


class TestPayload:
    def test_defaults_are_the_vertical_production_standard(self):
        """1080p @ 9:16 (= 1080x1920) / 5s / audio off. 2K is opt-in, never implicit.

        Superseded the earlier 720p/4s defaults when the causal-video standard
        was locked — see skills/creative/causal-video-production.md.
        """
        payload = OpenRouterVideo()._build_payload({"prompt": "  a glass  "})
        assert payload == {
            "model": "google/veo-3.1-lite",
            "prompt": "a glass",
            "duration": 5,
            "resolution": "1080p",
            "aspect_ratio": "9:16",
            "generate_audio": False,
        }

    def test_optional_fields_are_omitted_not_nulled(self):
        payload = OpenRouterVideo()._build_payload({"prompt": "x"})
        for key in ("seed", "frame_images", "input_references"):
            assert key not in payload

    def test_negative_prompt_is_never_sent_as_its_own_field(self):
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "a glass on wood", "negative_prompt": "no text, no watermark"}
        )
        assert "negative_prompt" not in payload

    def test_positive_prompt_survives_intact(self):
        positive = "A realistic workshop machine, close-up documentary cinematography."
        payload = OpenRouterVideo()._build_payload(
            {"prompt": positive, "negative_prompt": "no text, no logos"}
        )
        assert payload["prompt"].startswith(positive)

    def test_negative_prompt_is_appended_behind_the_marker(self):
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "a glass on wood", "negative_prompt": "no text, no watermark"}
        )
        assert payload["prompt"] == "a glass on wood Negative constraints: no text, no watermark"

    def test_act_without_a_negative_prompt_is_unchanged(self):
        payload = OpenRouterVideo()._build_payload({"prompt": "a glass on wood"})
        assert payload["prompt"] == "a glass on wood"
        assert "Negative constraints:" not in payload["prompt"]
        assert "negative_prompt" not in payload

    def test_empty_negative_prompt_is_ignored(self):
        payload = OpenRouterVideo()._build_payload({"prompt": "a glass on wood", "negative_prompt": "   "})
        assert payload["prompt"] == "a glass on wood"

    def test_negatives_are_not_duplicated_on_reassembly(self):
        tool = OpenRouterVideo()
        negative = "no text, no watermark"
        once = tool._build_payload({"prompt": "a glass on wood", "negative_prompt": negative})["prompt"]
        twice = tool._build_payload({"prompt": once, "negative_prompt": negative})["prompt"]
        assert once == twice
        assert twice.count("Negative constraints:") == 1
        assert twice.count(negative) == 1

    def test_negatives_already_inline_are_not_appended_again(self):
        prompt = "a glass on wood, and no text, no watermark anywhere"
        payload = OpenRouterVideo()._build_payload(
            {"prompt": prompt, "negative_prompt": "no text, no watermark"}
        )
        assert payload["prompt"] == prompt

    def test_optional_fields_pass_through_when_given(self):
        """seed is forwarded as-is; a reference image is normalised, not forwarded raw.

        This test used to assert input_references survived VERBATIM, which is
        exactly the hole closed on 2026-09-21: a bare string went out unchecked,
        skipping the working-tree / magic-byte / size guards every other image
        input gets. `seed` is a number and stays a straight passthrough.
        """
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "seed": 7,
             "input_references": [{"image_url": "https://example.com/u.jpg"}]}
        )
        assert payload["seed"] == 7
        assert payload["input_references"] == [
            {"type": "image_url", "image_url": {"url": "https://example.com/u.jpg"}}]


class TestGenerationCycle:
    def test_submit_poll_download_writes_the_mp4(self, fake_requests, api_key, tmp_path):
        out = tmp_path / "act1.mp4"
        fake_requests["queues"]["post"].append(
            FakeResponse({"id": "job_1", "polling_url": "https://openrouter.ai/api/v1/videos/job_1",
                          "status": "pending"}, status_code=202)
        )
        fake_requests["queues"]["get"].extend([
            FakeResponse({"id": "job_1", "status": "in_progress"}),
            FakeResponse({"id": "job_1", "status": "completed",
                          "unsigned_urls": ["https://openrouter.ai/api/v1/videos/job_1/content?index=0"],
                          "usage": {"cost": 0.2, "is_byok": False}}),
            FakeResponse(content=b"MP4BYTES"),
        ])

        result = OpenRouterVideo().execute({"prompt": "an empty glass", "output_path": str(out)})

        assert result.success is True
        assert out.read_bytes() == b"MP4BYTES"
        assert result.data["job_id"] == "job_1"
        assert result.data["model"] == "google/veo-3.1-lite"
        assert result.artifacts == [str(out)]

    def test_reported_cost_wins_over_the_static_estimate(self, fake_requests, api_key, tmp_path):
        out = tmp_path / "act1.mp4"
        fake_requests["queues"]["post"].append(FakeResponse({"id": "j", "status": "pending"}, status_code=202))
        fake_requests["queues"]["get"].extend([
            FakeResponse({"id": "j", "status": "completed", "unsigned_urls": ["u"],
                          "usage": {"cost": 0.37}}),
            FakeResponse(content=b"X"),
        ])
        result = OpenRouterVideo().execute({"prompt": "x", "output_path": str(out)})
        assert result.cost_usd == 0.37  # not the 0.12 the fallback table would predict
        assert result.data["reported_cost_usd"] == 0.37

    def test_static_rate_is_used_only_when_usage_cost_is_absent(self, fake_requests, api_key, tmp_path):
        out = tmp_path / "act1.mp4"
        fake_requests["queues"]["post"].append(FakeResponse({"id": "j", "status": "pending"}, status_code=202))
        fake_requests["queues"]["get"].extend([
            FakeResponse({"id": "j", "status": "completed", "unsigned_urls": ["u"]}),  # no usage block
            FakeResponse(content=b"X"),
        ])
        result = OpenRouterVideo().execute({"prompt": "x", "duration": 4, "output_path": str(out)})
        assert result.cost_usd == 0.20  # default 1080p audio-off SKU
        assert result.data["reported_cost_usd"] is None

    def test_download_sends_the_bearer_header(self, fake_requests, api_key, tmp_path):
        out = tmp_path / "act1.mp4"
        fake_requests["queues"]["post"].append(FakeResponse({"id": "j", "status": "pending"}, status_code=202))
        fake_requests["queues"]["get"].extend([
            FakeResponse({"id": "j", "status": "completed", "unsigned_urls": ["https://cdn/u"]}),
            FakeResponse(content=b"X"),
        ])
        OpenRouterVideo().execute({"prompt": "x", "output_path": str(out)})
        download = fake_requests["calls"]["get"][-1]
        assert download["url"] == "https://cdn/u"
        assert download["headers"]["Authorization"] == "Bearer test-key"

    def test_falls_back_to_the_content_endpoint_without_unsigned_urls(self, fake_requests, api_key, tmp_path):
        out = tmp_path / "act1.mp4"
        fake_requests["queues"]["post"].append(FakeResponse({"id": "j9", "status": "pending"}, status_code=202))
        fake_requests["queues"]["get"].extend([
            FakeResponse({"id": "j9", "status": "completed"}),
            FakeResponse(content=b"X"),
        ])
        OpenRouterVideo().execute({"prompt": "x", "output_path": str(out)})
        assert fake_requests["calls"]["get"][-1]["url"].endswith("/videos/j9/content?index=0")


class TestFailurePaths:
    def test_submit_http_error(self, fake_requests, api_key, tmp_path):
        fake_requests["queues"]["post"].append(FakeResponse(status_code=402, text="insufficient credits"))
        result = OpenRouterVideo().execute({"prompt": "x", "output_path": str(tmp_path / "o.mp4")})
        assert result.success is False
        assert "402" in result.error

    def test_missing_job_id(self, fake_requests, api_key, tmp_path):
        fake_requests["queues"]["post"].append(FakeResponse({"status": "pending"}, status_code=202))
        result = OpenRouterVideo().execute({"prompt": "x", "output_path": str(tmp_path / "o.mp4")})
        assert result.success is False
        assert "no job id" in result.error

    def test_failed_job_reports_the_error(self, fake_requests, api_key, tmp_path):
        fake_requests["queues"]["post"].append(FakeResponse({"id": "j", "status": "pending"}, status_code=202))
        fake_requests["queues"]["get"].append(
            FakeResponse({"id": "j", "status": "failed", "error": "content filter"})
        )
        result = OpenRouterVideo().execute({"prompt": "x", "output_path": str(tmp_path / "o.mp4")})
        assert result.success is False
        assert "content filter" in result.error

    def test_download_http_error(self, fake_requests, api_key, tmp_path):
        fake_requests["queues"]["post"].append(FakeResponse({"id": "j", "status": "pending"}, status_code=202))
        fake_requests["queues"]["get"].extend([
            FakeResponse({"id": "j", "status": "completed", "unsigned_urls": ["u"]}),
            FakeResponse(status_code=401, text="unauthorized"),
        ])
        result = OpenRouterVideo().execute({"prompt": "x", "output_path": str(tmp_path / "o.mp4")})
        assert result.success is False
        assert "401" in result.error
