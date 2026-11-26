from dataclasses import replace
import unittest

from .env import Action, Board, Outcome, SokobanEnv, load_boards
from .verify import shortest_path


class SokobanTests(unittest.TestCase):
    def setUp(self):
        self.board = Board.from_author_json("fixture", {"map": [
            [8, 8, 8, 8, 8],
            [8, 5, 1, 7, 8],
            [8, 4, 4, 4, 8],
            [8, 8, 8, 8, 8],
        ]})
        self.env = SokobanEnv(self.board)

    def test_walk_and_blocked_wall(self):
        before = self.env.observation
        result = self.env.step("up")
        self.assertEqual((result.event, result.reason), ("blocked", "player_path_blocked"))
        self.assertEqual(result.observation, before)
        result = self.env.step("down")
        self.assertEqual(result.event, "move")
        self.assertEqual(result.observation.player, (2, 1))
        self.assertEqual(result.observation.box, before.box)

    def test_solve_and_terminal(self):
        result = self.env.step("right")
        self.assertEqual(result.event, "solved")
        self.assertEqual(result.observation.box, result.observation.goal)
        self.assertEqual(result.observation.outcome, Outcome.SOLVED)
        self.assertIn("*", result.observation.render())
        with self.assertRaises(RuntimeError):
            self.env.step(Action.RESET)

    def test_push_into_boundary_cannot_wrap(self):
        board = Board.from_author_json("edge", {"map": [[1, 5, 7, 4]]})
        env = SokobanEnv(board)
        result = env.step("left")
        self.assertEqual((result.event, result.reason), ("blocked", "box_path_blocked"))
        self.assertEqual(env.observation.box, (0, 0))
        self.assertEqual(env.observation.player, (0, 1))
        self.assertEqual(env.observation.total_moves, 0)

    def test_push_off_goal_is_not_part_of_this_task(self):
        self.assertEqual(self.env.step("right").observation.outcome, Outcome.SOLVED)
        with self.assertRaises(RuntimeError):
            self.env.step("left")

    def test_reset_and_give_up_are_distinct(self):
        self.env.step("down")
        result = self.env.step("reset")
        self.assertEqual(result.event, "reset")
        self.assertEqual(result.observation.player, self.board.player_start)
        self.assertEqual(result.observation.box, self.board.box_start)
        self.assertEqual(result.observation.moves_since_reset, 0)
        self.assertEqual(result.observation.total_moves, 1)
        self.assertEqual(result.observation.resets, 1)
        result = self.env.step("give_up")
        self.assertEqual(result.observation.outcome, Outcome.GAVE_UP)
        self.assertNotIn("*", result.observation.render())

    def test_no_author_labels_in_observation(self):
        self.assertFalse(hasattr(self.env.observation, "author_optimal_cost"))
        self.assertFalse(hasattr(self.env.observation, "id"))
        self.assertEqual(self.env.observation.render(), "#####\n#PBG#\n#...#\n#####")
        other = SokobanEnv(replace(self.board, id="different", author_optimal_cost=-1))
        self.assertEqual(self.env.observation, other.observation)

    def test_push_then_reset_restores_box_and_player(self):
        board = Board.from_author_json("push", {"map": [[8, 5, 1, 4, 7, 8]]})
        env = SokobanEnv(board)
        result = env.step("right")
        self.assertEqual(result.event, "push")
        self.assertEqual(result.observation.player, board.box_start)
        self.assertEqual(result.observation.box, (0, 3))
        result = env.step("reset")
        self.assertEqual(result.observation.box, board.box_start)
        self.assertEqual(result.observation.player, board.player_start)

    def test_original_corpus_oracle_and_replay(self):
        boards = load_boards()
        self.assertEqual(set(boards), {str(n) for n in (*range(30001, 30016), *range(30026, 30034))})
        costs = {}
        for board in boards.values():
            path = shortest_path(board)
            costs[board.id] = len(path) if path is not None else -1
            if path is None:
                continue
            env = SokobanEnv(board)
            for action in path:
                self.assertNotEqual(env.step(action).event, "blocked", board.id)
            self.assertEqual(env.observation.outcome, Outcome.SOLVED, board.id)
        self.assertEqual(sum(cost != -1 for cost in costs.values()), 14)
        self.assertEqual(costs["30031"], 26)
        self.assertEqual({bid for bid, board in boards.items() if costs[bid] != board.author_optimal_cost},
                         {"30031"})

    def test_original_human_trace_includes_blocked_inputs(self):
        env = SokobanEnv(load_boards()["30009"])
        for direction in ("up", "up", "left", "left", "left", "up", "left"):
            self.assertNotEqual(env.step(direction).event, "blocked")
        self.assertEqual((env.observation.player, env.observation.box), ((2, 1), (3, 1)))
        for _ in range(2):
            self.assertEqual(env.step("down").reason, "box_path_blocked")
        self.assertEqual(env.observation.total_moves, 7)


if __name__ == "__main__":
    unittest.main()
