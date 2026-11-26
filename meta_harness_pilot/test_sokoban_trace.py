"""Offline checks for Part B record conversion using a synthetic board."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from .h0 import H0
from .mock_models import MalformedMock, MappingMock
from .ports import ModelReply
from .runner import Limits, run_decision, run_episode
from .sokoban_trace import episode_record, replay_record
from .tasks import sokoban_fixture


class _ActionMock:
    def __init__(self, *actions):
        self.actions = iter(actions)

    def respond(self, request):
        if request.stage == "propose":
            return ModelReply({"lines": []})
        return ModelReply({"action_id": next(self.actions), "plan_note": ""})


class _FailedModel:
    def respond(self, request):
        raise RuntimeError("mock provider failure")


class SokobanTraceTests(unittest.TestCase):
    def setUp(self):
        offline = patch("socket.socket", side_effect=AssertionError("offline test"))
        offline.start()
        self.addCleanup(offline.stop)

    def test_solve_records_final_direction_and_terminal(self):
        run = run_episode(sokoban_fixture(), _ActionMock("right", "right"), H0())
        self.assertEqual(episode_record(run), {
            "tokens": ["R", "R", "SOLVED"], "outcome": "solved", "moves": 2, "resets": 0,
        })

    def test_reset_and_give_up_add_no_direction_moves(self):
        run = run_episode(sokoban_fixture(), _ActionMock("right", "reset", "give_up"), H0())
        self.assertEqual(episode_record(run), {
            "tokens": ["R", "RESET", "GIVE_UP"], "outcome": "gave_up", "moves": 1, "resets": 1,
        })

    def test_cap_is_not_quit_and_scratch_does_not_enter_trajectory(self):
        run = run_episode(sokoban_fixture(), MappingMock(), H0(), Limits(max_decisions=1))
        self.assertEqual(run.decisions[0].preview_attempts, 2)
        self.assertEqual(episode_record(run), {
            "tokens": ["U", "CAP"], "outcome": "cap", "moves": 1, "resets": 0,
        })

    def test_failures_do_not_become_terminal_tokens(self):
        for mock, limits in (
            (MalformedMock(), Limits()),
            (_FailedModel(), Limits()),
            (MappingMock(), Limits(max_requests=1)),
        ):
            run = run_episode(sokoban_fixture(), mock, H0(), limits)
            with self.subTest(terminal=run.terminal), self.assertRaises(ValueError):
                episode_record(run)

    def test_inconsistent_trajectory_is_rejected(self):
        run = run_episode(sokoban_fixture(), _ActionMock("give_up"), H0())
        with self.assertRaises(ValueError):
            episode_record(replace(run, terminal="cap"))
        with self.assertRaises(ValueError):
            episode_record(replace(run, events=()))

    def test_replay_uses_actual_preview_counts_on_one_frame(self):
        task = sokoban_fixture()
        h0 = H0()
        before = task.frame()
        repeats = [
            run_decision(task, MappingMock(), h0, h0.start(), Limits(max_preview_attempts=i))
            for i in (2, 0, 1)
        ]
        self.assertEqual(replay_record(425.0, repeats), {"rt_ms": 425.0, "expansions": [2, 0, 1]})
        self.assertEqual(task.frame(), before)

    def test_replay_rejects_mixed_frames_and_failed_decisions(self):
        task = sokoban_fixture()
        h0 = H0()
        valid = run_decision(task, MappingMock(), h0, h0.start())
        failed = run_decision(task, MalformedMock(), h0, h0.start())
        task.commit("right")
        other = run_decision(task, MappingMock(), h0, h0.start())
        committed = run_episode(sokoban_fixture(), MappingMock(), h0, Limits(max_decisions=1))
        already_applied = replace(valid, trace=committed.decisions[0])
        for repeats in ((valid, valid), (valid, valid, other), (valid, valid, failed),
                        (valid, valid, already_applied)):
            with self.subTest(repeats=len(repeats)), self.assertRaises(ValueError):
                replay_record(425.0, repeats)
        for rt in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(rt=rt), self.assertRaises(ValueError):
                replay_record(rt, (valid, valid, valid))


if __name__ == "__main__":
    unittest.main()
