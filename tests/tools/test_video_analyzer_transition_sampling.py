"""Transition sampling: the analyzer must be able to see ACTIONS, not just states.

One frame per scene (start + 0.1s) only ever shows the state a cut lands on, so an
action happening inside a scene is invisible and the brief records a state list with
the causes removed. Opt-in transition sampling adds a pre-cut frame per scene and a
`transitions` block bracketing every boundary.

The default path must stay byte-for-byte unchanged — existing pipelines depend on it.

Run: pytest tests/tools/test_video_analyzer_transition_sampling.py -v
"""

from __future__ import annotations

from tools.analysis.video_analyzer import VideoAnalyzer

SCENES = [
    {"start_seconds": 0.0, "end_seconds": 1.95, "index": 0},
    {"start_seconds": 1.95, "end_seconds": 2.93, "index": 1},
    {"start_seconds": 2.93, "end_seconds": 3.82, "index": 2},
]


class TestDefaultBehaviourUnchanged:
    def test_off_by_default_in_schema(self):
        prop = VideoAnalyzer.input_schema["properties"]["transition_sampling"]
        assert prop["default"] is False

    def test_default_emits_one_frame_per_scene(self):
        ts = VideoAnalyzer()._compute_keyframe_timestamps(SCENES, 40, "standard")
        assert ts == [0.1, 2.05, 3.03]

    def test_explicit_false_matches_default(self):
        v = VideoAnalyzer()
        assert v._compute_keyframe_timestamps(SCENES, 40, "standard") == \
               v._compute_keyframe_timestamps(SCENES, 40, "standard", False)


class TestTransitionSampling:
    def test_adds_a_pre_cut_frame_per_scene(self):
        ts = VideoAnalyzer()._compute_keyframe_timestamps(SCENES, 40, "standard", True)
        assert len(ts) == 6
        # every scene now contributes a start AND an end sample
        for s in SCENES:
            assert any(abs(t - (s["start_seconds"] + 0.1)) < 1e-6 for t in ts)
            assert any(s["start_seconds"] < t < s["end_seconds"] and t > s["start_seconds"] + 0.1 for t in ts)

    def test_pre_cut_frame_stays_inside_its_scene(self):
        ts = VideoAnalyzer()._compute_keyframe_timestamps(SCENES, 40, "standard", True)
        for t in ts:
            assert any(s["start_seconds"] <= t <= s["end_seconds"] for s in SCENES)

    def test_very_short_scenes_get_no_extra_frame(self):
        tiny = [{"start_seconds": 0.0, "end_seconds": 0.2, "index": 0}]
        assert VideoAnalyzer()._compute_keyframe_timestamps(tiny, 40, "standard", True) == [0.1]


class TestTransitionsBlock:
    def _keyframes(self):
        v = VideoAnalyzer()
        ts = v._compute_keyframe_timestamps(SCENES, 40, "standard", True)
        return [{"timestamp": t, "scene_index": v._timestamp_to_scene(t, SCENES),
                 "path": f"f_{t}.jpg", "description": ""} for t in ts]

    def test_one_entry_per_boundary(self):
        tr = VideoAnalyzer()._build_transitions(SCENES, self._keyframes())
        assert len(tr) == len(SCENES) - 1

    def test_brackets_the_cut_with_before_and_after_state(self):
        tr = VideoAnalyzer()._build_transitions(SCENES, self._keyframes())[0]
        assert tr["state_before_seconds"] < tr["at_seconds"] <= tr["state_after_seconds"]
        assert tr["from_scene"] == 0 and tr["to_scene"] == 1

    def test_reports_whether_within_scene_change_is_observable(self):
        tr = VideoAnalyzer()._build_transitions(SCENES, self._keyframes())
        assert all(t["within_scene_change_observable"] for t in tr)

    def test_tool_does_not_guess_the_action(self):
        """The tool guarantees the evidence exists; the agent writes the action."""
        tr = VideoAnalyzer()._build_transitions(SCENES, self._keyframes())[0]
        assert tr["action"] == ""
        assert tr["elided"] is None
        assert tr["confidence"] is None

    def test_single_scene_yields_no_transitions(self):
        one = [SCENES[0]]
        kf = [{"timestamp": 0.1, "scene_index": 0, "path": "f.jpg", "description": ""}]
        assert VideoAnalyzer()._build_transitions(one, kf) == []
