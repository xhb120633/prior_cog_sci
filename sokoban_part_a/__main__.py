"""Offline smoke test and manual play. No model or network calls."""

import argparse

from .env import Action, Outcome, SokobanEnv, load_boards


_SHORT = {"u": "up", "d": "down", "l": "left", "r": "right", "reset": "reset", "n": "give_up"}


def _action(token: str) -> Action:
    token = token.strip().lower()
    return Action(_SHORT.get(token, token))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("board", nargs="?", help="canonical actor-study board ID")
    cli.add_argument("--list", action="store_true", help="list board IDs")
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument("--actions", help="comma-separated actions, e.g. up,up,right,reset")
    modes.add_argument("--play", action="store_true", help="play interactively in the terminal")
    args = cli.parse_args()
    boards = load_boards()
    if args.list:
        print(" ".join(boards))
        return
    if args.board not in boards:
        cli.error("Pass a valid board ID, or use --list")

    env = SokobanEnv(boards[args.board])
    print(env.observation.render())

    def show(result):
        label = f"{result.action.value} -> {result.event}"
        if result.reason:
            label += f" ({result.reason})"
        print(f"\n{label}")
        print(result.observation.render())

    if args.actions is not None:
        try:
            actions = [_action(token) for token in args.actions.split(",")]
        except ValueError as error:
            cli.error(str(error))
        for action in actions:
            if env.observation.outcome != Outcome.ACTIVE:
                break
            show(env.step(action))
    elif args.play:
        while env.observation.outcome == Outcome.ACTIVE:
            try:
                token = input("u/d/l/r, reset, n (give up), q (exit)> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if token == "q":
                break
            try:
                show(env.step(_action(token)))
            except ValueError:
                print("Unknown action")

    state = env.observation
    print(f"\nOutcome: {state.outcome.value}; legal moves: {state.total_moves}; resets: {state.resets}")


if __name__ == "__main__":
    main()
