"""Regression: edit_decisions.subtitles.style may be a STRING.

`schemas/artifacts/edit_decisions.schema.json` types `subtitles.style` as a
string, but the subtitle-style merge in video_compose treated it as a
field->value dict and called `.items()` on it. A schema-valid artifact that set
`style: "none"` therefore crashed the whole compose operation with
`'str' object has no attribute 'items'` — even with `subtitles.enabled: false`.

Run: pytest tests/tools/test_video_compose_subtitle_style_type.py -v
"""

from __future__ import annotations

from tools.video.video_compose import VideoCompose


def _resolve(edit_decisions):
    return VideoCompose()._resolve_subtitle_style(
        playbook=None, edit_decisions=edit_decisions, explicit_style=None
    )


class TestSubtitleStyleType:
    def test_string_style_does_not_raise(self):
        """The schema-valid string form must be tolerated, not fatal."""
        style = _resolve({"subtitles": {"enabled": False, "style": "N/A — no subtitles in this piece"}})
        assert isinstance(style, dict)

    def test_string_style_leaves_defaults_intact(self):
        baseline = _resolve({})
        assert _resolve({"subtitles": {"enabled": False, "style": "none"}}) == baseline

    def test_dict_style_still_merges(self):
        style = _resolve({"subtitles": {"enabled": True, "style": {"font": "Inter", "font_size": 64}}})
        assert style["font"] == "Inter"
        assert style["font_size"] == 64

    def test_missing_subtitles_block_is_fine(self):
        assert isinstance(_resolve({}), dict)

    def test_none_style_does_not_raise(self):
        assert isinstance(_resolve({"subtitles": {"enabled": False, "style": None}}), dict)
