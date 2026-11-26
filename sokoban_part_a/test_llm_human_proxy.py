from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import patch

from .env import Board
from .llm_human_proxy import DEFAULT_DECISIONS, DEFAULT_EFFORT, DEFAULT_MODELS, SCHEMA, OpenRouter, Reply, ProviderError, main, parse_decision, readable_reasoning, reasoning_config, run_episode


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.board = Board.from_author_json("secret-board-label", {
            "map": [[8, 8, 8, 8, 8], [8, 5, 1, 7, 8], [8, 8, 8, 8, 8]], "optimal_cost": -1234,
        })
        self.rows = []

    def actor(self, *actions):
        data = iter(actions)

        def action(_, messages):
            prompt = json.dumps(messages)
            self.assertNotIn("secret-board-label", prompt)
            self.assertNotIn("-1234", prompt)
            x = next(data)
            if isinstance(x, Exception):
                raise x
            if isinstance(x, Reply):
                return x
            return Reply(json.dumps({"rationale": "The box can move toward the goal.", "action": x}),
                         finish_reason="stop", usage={"cost": 0.01})
        return action

    def test_one_action_per_decision_and_solving_stops(self):
        summary = run_episode("sol", self.board, 9, self.actor("up", "right"), self.rows.append)
        self.assertEqual((summary["outcome"], summary["decisions"], summary["blocks"], summary["moves"]),
                         ("solved", 2, 1, 1))
        self.assertEqual(self.rows[0]["before"], self.rows[0]["after"])
        self.assertEqual(self.rows[1]["rationale"], "The box can move toward the goal.")
        self.assertEqual(summary["reported_cost_usd"], 0.02)

    def test_censoring_is_not_give_up(self):
        budget = run_episode("sol", self.board, 1, self.actor("up"), self.rows.append)
        give_up = run_episode("sol", self.board, 1, self.actor("give_up"), self.rows.append)
        self.assertEqual((budget["outcome"], give_up["outcome"]), ("budget_exhausted", "gave_up"))

    def test_default_cap_counts_blocked_actions_and_resets(self):
        self.assertEqual(DEFAULT_DECISIONS, 100)
        actions = ["reset", "up"] * 50
        budget = run_episode("sol", self.board, DEFAULT_DECISIONS, self.actor(*actions), self.rows.append)
        self.assertEqual((budget["outcome"], budget["decisions"], budget["api_requests"]),
                         ("budget_exhausted", 100, 100))
        self.assertEqual((budget["blocks"], budget["resets"], budget["moves"]), (50, 50, 0))
        self.assertEqual(self.rows[-1]["context_messages"], 200)

    def test_bad_format_does_not_reach_env(self):
        actor = self.actor(Reply('{"action":"right"}', finish_reason="stop"), "right")
        result = run_episode("sol", self.board, 2, actor, self.rows.append)
        self.assertEqual((result["invalid_decisions"], result["game_actions"], result["decisions"]), (1, 1, 2))
        self.assertEqual(self.rows[0]["before"], self.rows[1]["before"])

    def test_provider_failure_is_not_a_game_action(self):
        result = run_episode("sol", self.board, 2, self.actor(ProviderError("http_401")), self.rows.append)
        self.assertEqual((result["outcome"], result["api_requests"], result["decisions"]), ("provider_error", 1, 0))
        self.assertEqual(self.rows[0]["status"], "provider_error")

    def test_truncated_reply_is_not_a_game_action(self):
        result = run_episode("sol", self.board, 2, self.actor(Reply("", finish_reason="length")), self.rows.append)
        self.assertEqual((result["outcome"], result["game_actions"]), ("provider_error", 0))

    def test_parse_rejects_multi_action(self):
        with self.assertRaises(ValueError):
            parse_decision('{"rationale":"Push it.","action":"up,right"}')

    def test_reasoning_uses_readable_fields_only(self):
        message = {"reasoning_details": [
            {"type": "reasoning.encrypted", "data": "encrypted-blob"},
            {"type": "reasoning.text", "text": "Think ahead", "signature": "opaque-signature"},
            {"type": "reasoning.summary", "summary": "Summary"},
            {"type": "reasoning.text", "text": "[REDACTED]"},
        ], "reasoning": "Duplicate compatibility text"}
        self.assertEqual(readable_reasoning(message, "gemma"), (
            {"kind": "thinking_text", "source": "reasoning.text", "text": "Think ahead"},
            {"kind": "summary", "source": "reasoning.summary", "text": "Summary"},
        ))
        self.assertEqual(readable_reasoning(message, "opus")[0]["kind"], "summary")
        self.assertEqual(readable_reasoning({"reasoning": "Fallback"}, "gemma")[0]["kind"],
                         "untyped_provider_text")
        self.assertEqual(readable_reasoning({"reasoning_details": message["reasoning_details"][:1]}, "sol"), ())
        self.assertEqual(reasoning_config("gemma", "low"), {"enabled": True})
        self.assertEqual(reasoning_config("sol", "low"), {"effort": "low", "summary": "concise"})

    def test_reasoning_is_logged_but_not_replayed(self):
        reply = Reply('{"rationale":"Try up.","action":"up"}', finish_reason="stop",
                      provider_reasoning=({"kind": "summary", "source": "reasoning.summary", "text": "A side channel."},))
        calls = []

        def actor(alias, messages):
            calls.append(messages)
            if len(calls) == 1:
                return reply
            self.assertIn("Try up.", json.dumps(messages))
            self.assertNotIn("A side channel.", json.dumps(messages))
            return Reply('{"rationale":"Push right.","action":"right"}', finish_reason="stop")

        summary = run_episode("sol", self.board, 2, actor, self.rows.append)
        self.assertEqual(self.rows[0]["provider_reasoning"], reply.provider_reasoning)
        self.assertEqual(summary["requests_with_provider_reasoning"], 1)
        self.assertEqual(summary["outcome"], "solved")

    def test_qwen_uses_same_schema_without_reasoning_parameters(self):
        fake = {"model": "qwen/qwen3-235b-a22b-07-25", "provider": "Parasail", "choices": [
            {"finish_reason": "stop", "message": {"content": '{"rationale":"Move up.","action":"up"}'}}
        ]}
        actor = OpenRouter("mock-key", 1000, "low", 1)
        with patch("sokoban_part_a.llm_human_proxy.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(fake).encode()
            result = actor("qwen", [{"role": "user", "content": "board"}])
            payload = json.loads(opener.return_value.open.call_args.args[0].data)
        self.assertEqual(result.response_model, fake["model"])
        self.assertEqual(result.provider_reasoning, ())
        self.assertEqual(payload["response_format"], SCHEMA)
        self.assertEqual(payload["provider"]["only"], ["parasail/fp8"])
        self.assertEqual(payload["provider"]["max_price"], {"prompt": 0.14, "completion": 0.8})
        self.assertNotIn("reasoning", payload)
        self.assertNotIn("include_reasoning", payload)

    def test_default_model_configures_low_reasoning_on_all_three(self):
        self.assertEqual(DEFAULT_MODELS, ("sol", "opus", "qwen38"))
        self.assertEqual(DEFAULT_EFFORT, "low")
        for alias in DEFAULT_MODELS:
            self.assertEqual(reasoning_config(alias, DEFAULT_EFFORT)["effort"], "low")
        fake = {"model": "qwen/qwen3.8-27b", "provider": "DeepInfra", "choices": [
            {"finish_reason": "stop", "message": {"content": '{"rationale":"Move up.","action":"up"}',
             "reasoning_details": [{"type": "reasoning.text", "text": "Check the open square."}]}}
        ]}
        actor = OpenRouter("mock-key", 1000, DEFAULT_EFFORT, 1)
        with patch("sokoban_part_a.llm_human_proxy.build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(fake).encode()
            reply = actor("qwen38", [{"role": "user", "content": "board"}])
            payload = json.loads(opener.return_value.open.call_args.args[0].data)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["response_format"], SCHEMA)
        self.assertEqual(payload["provider"]["only"], ["deepinfra/fp8"])
        self.assertEqual(payload["provider"]["max_price"], {"prompt": 0.15, "completion": 1.875})
        self.assertEqual(reply.provider_reasoning[0]["kind"], "thinking_text")

    def test_full_conversation_is_append_only_past_eight_moves_and_reset(self):
        actions = iter(["up", "reset"] + ["up"] * 8 + ["right"])
        sent = []

        def actor(_, messages):
            sent.append(messages)
            return Reply(json.dumps({"rationale": f"Visible plan {len(sent)}", "action": next(actions)}),
                         finish_reason="stop")

        summary = run_episode("sol", self.board, 11, actor, self.rows.append)
        self.assertEqual(summary["outcome"], "solved")
        self.assertEqual(len(sent[-1]), 22)
        for previous, current in zip(sent, sent[1:]):
            self.assertEqual(current[:-2], previous)
            self.assertEqual([m["role"] for m in current[-2:]], ["assistant", "user"])
        self.assertIn("#PBG#", sent[-1][1]["content"])
        self.assertIn('"Visible plan 1"', sent[-1][2]["content"])
        self.assertIn("up -> blocked", sent[-1][3]["content"])
        self.assertIn("reset -> reset", sent[-1][5]["content"])
        self.assertEqual([row["context_messages"] for row in self.rows], list(range(2, 23, 2)))

    def test_five_parallel_repeats_start_with_identical_isolated_contexts(self):
        gate = Barrier(5)
        received = []

        def actor(alias, messages):
            received.append(messages)
            gate.wait(timeout=5)
            return Reply('{"rationale":"I think this board is unsolvable.","action":"give_up"}', finish_reason="stop")

        with (TemporaryDirectory() as temp,
              patch("sokoban_part_a.llm_human_proxy.read_openrouter_key", return_value="mock-key"),
              patch("sokoban_part_a.llm_human_proxy.OpenRouter", return_value=actor),
              redirect_stdout(StringIO())):
            run_dir = Path(temp) / "repeats"
            argv = ["runner", "--models", "opus", "--boards", "30011", "--repeats", "5", "--workers", "5",
                    "--provider-pin", "claude-on-aws", "--run-dir", str(run_dir)]
            with patch("sys.argv", argv):
                main()
            rows = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["replicate"] for row in rows], [1, 2, 3, 4, 5])
            self.assertTrue(all(row["outcome"] == "gave_up" for row in rows))
            self.assertEqual(len(list(run_dir.glob("opus-30011-r*.jsonl"))), 5)
        self.assertEqual(len(received), 5)
        self.assertTrue(all(msgs == received[0] and len(msgs) == 2 for msgs in received))
        self.assertEqual(len({id(msgs) for msgs in received}), 5)


if __name__ == "__main__":
    unittest.main()
