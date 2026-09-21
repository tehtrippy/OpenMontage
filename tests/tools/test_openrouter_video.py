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
        # google/veo-3.1-lite at the VERIFIED $0.03/s, not the listed $0.05/s
        assert OpenRouterVideo().estimate_cost({"prompt": "x", "duration": 4}) == 0.12

    def test_verified_rate_matches_the_measured_charge(self):
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
        tool = OpenRouterVideo()
        act = {"prompt": "x", "model": "google/veo-3.1-lite", "duration": 4}
        assert round(tool.estimate_cost(act) * 4, 2) == 0.48

    def test_known_model_rates_differ(self):
        tool = OpenRouterVideo()
        cheap = tool.estimate_cost({"prompt": "x", "model": "bytedance/seedance-2.0-mini", "duration": 4})
        dear = tool.estimate_cost({"prompt": "x", "model": "bytedance/seedance-2.5", "duration": 4})
        assert cheap < dear

    def test_unknown_model_uses_fallback_rate(self):
        assert OpenRouterVideo().estimate_cost(
            {"prompt": "x", "model": "someone/not-in-the-table", "duration": 10}
        ) == 1.0


class TestPayload:
    def test_defaults_are_the_vertical_production_standard(self):
        """1080x1920 / 9:16 / 5s / audio off. 2K is opt-in, never implicit.

        Superseded the earlier 720p/4s defaults when the causal-video standard
        was locked — see skills/creative/causal-video-production.md.
        """
        payload = OpenRouterVideo()._build_payload({"prompt": "  a glass  "})
        assert payload == {
            "model": "google/veo-3.1-lite",
            "prompt": "a glass",
            "duration": 5,
            "resolution": "1080x1920",
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
        payload = OpenRouterVideo()._build_payload(
            {"prompt": "x", "seed": 7, "input_references": [{"image_url": "u"}]}
        )
        assert payload["seed"] == 7
        assert payload["input_references"] == [{"image_url": "u"}]


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
        assert result.cost_usd == 0.12
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
