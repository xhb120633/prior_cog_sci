"""End-to-end checks from a synthetic pilot trace to the unchanged scorer."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from .h0 import H0
from .mock_models import MappingMock
from .runner import Limits, run_decision, run_episode
from .sokoban_scoring import REPO, _compare, score_sokoban
from .sokoban_trace import episode_record, replay_record
from .tasks import sokoban_fixture


class SokobanScoringTests(unittest.TestCase):
    def test_pilot_records_reach_both_frozen_metrics(self):
        # A deliberately constructed scoring fixture, not a behavioral result.
        # The observed RT order matches the imposed scratch-work counts 0, 1, 2.
        with patch("socket.socket", side_effect=AssertionError("offline check")):
            h0, model = H0(), MappingMock()
            run = run_episode(sokoban_fixture(), model, h0, Limits(max_decisions=1))
            model_episode = episode_record(run)
            task = sokoban_fixture()
            rows = [replay_record(rt, [run_decision(
                task, model, h0, h0.start(), Limits(max_preview_attempts=work),
            ) for _ in range(3)]) for rt, work in ((150, 0), (350, 1), (550, 2))]
            repo = Path(os.environ.get("PART_B_TEST_REPO", REPO))
            out = score_sokoban(
                {"fixture": [dict(model_episode), {"outcome": "incomplete"}]},
                {"fixture": [model_episode]}, {"fixture": rows}, {"fixture_family": ["fixture"]},
                repo=repo,
            )
        self.assertEqual(out["macro"]["trajectory_distance"], 0.0)
        self.assertAlmostEqual(out["macro"]["rt_spearman"], 1.0)
        self.assertEqual(out["macro"]["n_rt"], 3)
        self.assertEqual(out["macro"]["human_diagnostics"]["n"], 1)

    def test_archive_parity_checks_values_and_coverage(self):
        self.assertEqual(_compare({"a": [0.3, 2]}, {"a": [0.3, 2]}), (2, 0))
        with self.assertRaisesRegex(ValueError, "Score mismatch"):
            _compare({"a": [0.31, 2]}, {"a": [0.3, 2]})
        with self.assertRaisesRegex(ValueError, "Field mismatch"):
            _compare({"b": [0.3, 2]}, {"a": [0.3, 2]})
