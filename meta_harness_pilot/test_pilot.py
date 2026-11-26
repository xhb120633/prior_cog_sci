from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from .h0 import H0, H0Config
from .mock_models import JsonTextMock, MalformedMock, MappingMock
from .ports import ModelReply
from .runner import Limits, run_decision, run_episode
from .smoke import run_smoke
from .tasks import FourInARowTask, sokoban_fixture


class PilotTests(unittest.TestCase):
    def setUp(self):
        blocker = patch("socket.socket", side_effect=AssertionError("Tests are offline"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def test_same_h0_and_config_across_task_and_model_adapters(self):
        report = run_smoke()
        self.assertEqual(report["api_calls"], 0)
        self.assertEqual(len(report["cases"]), 4)
        self.assertEqual(report["malformed_response"]["commits"], 0)

    def test_single_decision_has_real_previews_but_no_commit(self):
        for task in (sokoban_fixture(), FourInARowTask()):
            with self.subTest(task=type(task).__name__):
                before = task.reset()
                h0 = H0()
                r = run_decision(task, JsonTextMock(), h0, h0.start())
                self.assertEqual(r.trace.status, "ready")
                self.assertEqual(r.trace.preview_attempts, 2)
                self.assertTrue(all(p.result.frame != before for p in r.trace.previews))
                self.assertEqual(task.frame(), before)
                self.assertIsNone(r.trace.event)

    def test_scratch_branches_and_episode_reset_are_isolated(self):
        for task in (sokoban_fixture(), FourInARowTask()):
            with self.subTest(task=type(task).__name__):
                initial = task.reset()
                root = task.scratch()
                first, second = (a.id for a in initial.actions[:2])
                _, a = task.preview(root, first)
                _, b = task.preview(root, second)
                self.assertNotEqual(a, b)
                self.assertEqual(initial, task.frame())
                self.assertEqual(task.commit(first).after, a)
                self.assertEqual(task.reset(), initial)
                with self.assertRaises(ValueError):
                    task.preview(root, first)

    def test_task_transitions_known_outcomes(self):
        push = sokoban_fixture()
        self.assertEqual(push.commit("right").label, "push")
        self.assertEqual(push.commit("right").after.terminal, "solved")
        four = FourInARowTask()
        for action in ("p00", "p09", "p01", "p10", "p02", "p11", "p03"):
            event = four.commit(action)
        self.assertEqual(event.after.terminal, "win:X")
        self.assertFalse(event.after.actions)

    def test_malformed_or_illegal_choice_does_not_change_live_state(self):
        for model in (MalformedMock(), MalformedMock(kind="illegal"), MalformedMock(stage="propose")):
            for task in (sokoban_fixture(), FourInARowTask()):
                with self.subTest(kind=model.kind, stage=model.stage, task=type(task).__name__):
                    out = run_episode(task, model, H0(), Limits(max_decisions=1))
                    self.assertEqual(out.terminal, "invalid_model_output")
                    self.assertFalse(out.events)
                    self.assertEqual(out.initial, out.final)

    def test_caps_are_not_game_outcomes(self):
        out = run_episode(sokoban_fixture(), MappingMock(), H0(), Limits(max_decisions=1))
        self.assertEqual(out.terminal, "cap")
        self.assertIsNone(out.final.terminal)
        self.assertEqual(len(out.events), 1)
        self.assertNotEqual(out.events[0].label, "gave_up")

    def test_work_limits_are_shared(self):
        limits = Limits(max_preview_attempts=1, max_decisions=1)
        out = run_episode(FourInARowTask(), MappingMock(), H0(), limits)
        self.assertEqual(out.decisions[0].preview_attempts, 1)
        self.assertIn("preview_budget_reached", out.decisions[0].notes)
        blocked = run_episode(FourInARowTask(), MappingMock(), H0(), replace(limits, max_requests=1))
        self.assertEqual(blocked.terminal, "budget_exhausted")
        self.assertEqual(blocked.decisions[0].request_count, 1)
        self.assertFalse(blocked.events)

    def test_actor_private_plans_are_not_shared(self):
        class ActorNoteMock(MappingMock):
            def respond(self, request):
                if request.stage == "propose":
                    return super().respond(request)
                context = json.loads(request.messages[-1].text)
                return ModelReply({"action_id": request.action_ids[0],
                                   "plan_note": context["current"]["actor"]})

        task = FourInARowTask()
        model = ActorNoteMock()
        h0 = H0()
        out = run_episode(task, model, h0, Limits(max_decisions=3))
        contexts = [json.loads(d.model_calls[0].request.messages[-1].text) for d in out.decisions]
        self.assertEqual([p["active_plan"] for p in contexts], ["", "", out.initial.actor])
        self.assertEqual([len(p["recent_public_events"]) for p in contexts], [0, 1, 2])
        reset = run_episode(task, model, h0, Limits(max_decisions=1))
        self.assertEqual(contexts[0], json.loads(reset.decisions[0].model_calls[0].request.messages[-1].text))

    def test_parameters_change_memory_without_changing_policy_type(self):
        task = sokoban_fixture()
        event = task.commit("right")
        self.assertEqual(H0().observe(event, H0().start()).events, (event,))
        no_history = H0(H0Config(recent_events=0))
        self.assertEqual(no_history.observe(event, no_history.start()).events, ())
        with self.assertRaises(ValueError):
            H0Config(recent_events=9)


if __name__ == "__main__":
    unittest.main()
