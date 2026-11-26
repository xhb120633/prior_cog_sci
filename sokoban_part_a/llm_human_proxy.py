"""Run independent LLM-human-proxy episodes through OpenRouter."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from .env import Action, Board, Observation, Outcome, SokobanEnv, load_boards


URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {
    "sol": ("openai/gpt-6.2-sol", 2, 10),
    "astra": ("openai/gpt-6.2-astra", 8, 40),
    "opus": ("anthropic/claude-opus-5.5", 4, 20),
    "qwen38": ("qwen/qwen3.8-27b", 0.15, 1.875),
    "qwen": ("qwen/qwen3-235b-a22b-2507", 0.14, 0.8),
    "gemma": ("google/gemma-4-31b-it", 0.14, 0.4),
    "deepseek": ("deepseek/deepseek-v4.1-flash-1017", 0.3, 1.2),
}
PINNED_PROVIDERS = {"gemma": "crusoe/bf16", "qwen": "parasail/fp8", "qwen38": "deepinfra/fp8"}
MODEL_RESPONSE_ALIASES = {"qwen": {"qwen/qwen3-235b-a22b-07-25"}}
SUMMARY_ONLY_MODELS = {"sol", "astra", "opus"}
BOARD_IDS = ("30010", "30011", "30009")
DEFAULT_DECISIONS = 100
DEFAULT_MODELS = ("sol", "opus", "qwen38")
DEFAULT_EFFORT = "low"
SYSTEM = """You are the actor in a one-box Sokoban puzzle. Play the game, one decision at a time. Move the box onto the goal using as few legal moves as you can.

The grid is fully visible. # is a wall, . is floor, P is you, B is the box, G is the goal, + is you standing on the goal, and * is the box on the goal. Counts start at zero. Up/down change row, left/right change column. A direction moves you one square. If that square has the box, the same input pushes it one square if the square beyond is open. You cannot pull the box, move diagonally, or push through walls. Blocked moves do not change the board. Reset restores the original board. Some starting boards may be unsolvable. Give up only if you believe the original puzzle cannot be solved, not merely because a move was blocked or you need to try a different route.

Write one or two short sentences stating the immediate consideration or plan for this move, then choose exactly one action. Do not list a sequence of actions as your answer. Return a JSON object with `rationale` first and `action` second. State observations and hypotheses as such; do not claim an action already happened. The action must be up, down, left, right, reset, or give_up. The environment will return the result."""
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sokoban_actor_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"rationale": {"type": "string"},
                           "action": {"type": "string", "enum": [a.value for a in Action]}},
            "additionalProperties": False,
            "required": ["rationale", "action"],
        },
    },
}


def actor_turn(observation: Observation, feedback: str | None = None) -> dict:
    grid = observation.render().splitlines()
    board = "    " + "".join(str(i % 10) for i in range(observation.width)) + "\n"
    board += "\n".join(f"{i:2d}  {line}" for i, line in enumerate(grid))
    counters = (f"Moves made this attempt: {observation.moves_since_reset}. "
                f"Total legal moves: {observation.total_moves}. Resets: {observation.resets}.")
    prefix = f"Environment feedback: {feedback}\n\n" if feedback is not None else ""
    return {"role": "user", "content": f"{prefix}Current board (row labels at left, column digits above):\n"
            f"{board}\n\n{counters}\n\nChoose one action now."}


def read_openrouter_key(dotenv: Path) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip('"\'')
                break
    if not key or any(c.isspace() for c in key):
        raise RuntimeError("OPENROUTER_API_KEY is missing or malformed. Configure it locally.")
    return key


def tls_context():
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class ProviderError(Exception):
    def __init__(self, label: str, transient: bool = False):
        super().__init__(label)
        self.transient = transient


@dataclass(frozen=True)
class Reply:
    content: str
    response_model: str | None = None
    provider: str | None = None
    generation_id: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    provider_reasoning: tuple[dict[str, str], ...] = ()


def reasoning_config(alias: str, effort: str) -> dict | None:
    if alias == "qwen":
        return None  # Dedicated non-thinking checkpoint. Do not request unsupported parameters.
    if alias == "gemma":
        return {"enabled": True}
    if alias in {"sol", "astra"}:
        return {"effort": effort, "summary": "concise"}
    return {"effort": effort}


def readable_reasoning(message: dict, alias: str) -> tuple[dict[str, str], ...]:
    """Keep only readable provider-returned text, never encrypted blocks or signatures."""
    rows = []
    details = message.get("reasoning_details")
    if isinstance(details, list):
        for block in details:
            if not isinstance(block, dict):
                continue
            source = block.get("type")
            if source == "reasoning.summary":
                text = block.get("summary")
            elif source == "reasoning.text":
                text = block.get("text")
            else:
                continue
            if isinstance(text, str) and text.strip() not in {"", "[REDACTED]"}:
                kind = "summary" if source == "reasoning.summary" or alias in SUMMARY_ONLY_MODELS else "thinking_text"
                rows.append({"kind": kind, "source": source, "text": text})
    if not rows:
        fallback = message.get("reasoning")
        if isinstance(fallback, str) and fallback.strip() not in {"", "[REDACTED]"}:
            kind = "summary" if alias in SUMMARY_ONLY_MODELS else "untyped_provider_text"
            rows.append({"kind": kind, "source": "reasoning", "text": fallback})
    return tuple(rows)


class OpenRouter:
    def __init__(self, key: str, max_tokens: int, effort: str, timeout: float,
                 provider_pins: dict[str, str] | None = None):
        self._key = key
        self.max_tokens = max_tokens
        self.effort = effort
        self.timeout = timeout
        self.provider_pins = PINNED_PROVIDERS if provider_pins is None else provider_pins

    def __call__(self, alias: str, messages: list[dict]) -> Reply:
        model, input_cap, output_cap = MODELS[alias]
        provider = {"require_parameters": True, "max_price": {"prompt": input_cap, "completion": output_cap}}
        if alias in self.provider_pins:
            provider.update(only=[self.provider_pins[alias]], allow_fallbacks=False)
        payload = {"model": model, "messages": messages, "stream": False,
                   "max_tokens": self.max_tokens, "response_format": SCHEMA,
                   "provider": provider, "usage": {"include": True}}
        reasoning = reasoning_config(alias, self.effort)
        if reasoning is not None:
            payload["reasoning"] = reasoning
        request = Request(URL, json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        try:
            opener = build_opener(HTTPSHandler(context=tls_context()), NoRedirect())
            with opener.open(request, timeout=self.timeout) as response:
                result = json.loads(response.read(2 * 1024 * 1024))
        except HTTPError as error:
            raise ProviderError(f"http_{error.code}", error.code in {408, 429, 502, 503, 504}) from None
        except (URLError, OSError, TimeoutError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            raise ProviderError(f"transport_{type(reason).__name__}", True) from None
        except (ValueError, UnicodeError):
            raise ProviderError("invalid_provider_response") from None

        if not isinstance(result, dict):
            raise ProviderError("invalid_provider_response")
        if result.get("error"):
            error = result["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise ProviderError(f"provider_{code}" if type(code) is int else "provider_error")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderError("missing_provider_choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ProviderError("missing_provider_message")
        content = message.get("content")
        usage = result.get("usage")
        # Usage fields are numeric metadata. Only separately allowlisted reasoning text is retained.
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        usage = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
                 if type(usage.get(k)) in (int, float)} if isinstance(usage, dict) else {}
        if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
            usage["reasoning_tokens"] = details["reasoning_tokens"]
        if isinstance(result.get("model"), str):
            returned = result["model"].split(":", 1)[0]
            # Some providers append a dated snapshot to the requested base ID.
            if returned != model and returned not in MODEL_RESPONSE_ALIASES.get(alias, ()) and not returned.startswith(model + "-"):
                raise ProviderError("unexpected_model_id")
        return Reply(content if isinstance(content, str) else "", result.get("model"), result.get("provider"),
                     result.get("id"), choice.get("finish_reason"), usage, readable_reasoning(message, alias))


def parse_decision(content: str) -> tuple[Action, str]:
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"rationale", "action"}:
        raise ValueError("wrong response shape")
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 600:
        raise ValueError("missing or overlong rationale")
    return Action(value["action"]), rationale.strip()


def snapshot(observation: Observation) -> dict:
    return {"grid": observation.render(), "player": observation.player, "box": observation.box,
            "goal": observation.goal, "outcome": observation.outcome.value,
            "moves_since_reset": observation.moves_since_reset, "total_moves": observation.total_moves,
            "resets": observation.resets}


def run_episode(alias: str, board: Board, limit: int, actor, record, *, replicate: int = 1) -> dict:
    env = SokobanEnv(board)
    conversation = [{"role": "system", "content": SYSTEM}, actor_turn(env.observation)]
    requests_made = 0
    decisions = game_actions = blocks = pushes = format_errors = 0
    reason = "budget_exhausted"
    cost = 0.0
    cost_records = 0
    requests_with_provider_reasoning = 0
    provider_reasoning_kinds = set()

    while decisions < limit and requests_made < limit + 2 and env.observation.outcome == Outcome.ACTIVE:
        before = snapshot(env.observation)
        requests_made += 1
        row = {"model": MODELS[alias][0], "board_id": board.id, "replicate": replicate, "request": requests_made,
               "context_messages": len(conversation), "before": before}
        start = time.monotonic()
        try:
            reply = actor(alias, list(conversation))
        except ProviderError as error:
            row.update(status="provider_error", error=str(error), latency_s=round(time.monotonic() - start, 3))
            record(row)
            if error.transient and requests_made - decisions == 1:
                time.sleep(1)
                continue
            reason = "provider_error"
            break

        row.update(latency_s=round(time.monotonic() - start, 3), response_model=reply.response_model,
                   provider=reply.provider, generation_id=reply.generation_id,
                   finish_reason=reply.finish_reason, usage=reply.usage,
                   provider_reasoning=reply.provider_reasoning)
        if reply.provider_reasoning:
            requests_with_provider_reasoning += 1
            provider_reasoning_kinds.update(block["kind"] for block in reply.provider_reasoning)
        used = (reply.usage or {}).get("cost")
        if used is not None:
            cost += used
            cost_records += 1

        if reply.finish_reason != "stop" or not reply.content.strip():
            row.update(status="provider_error", error="completion_truncated_or_missing")
            record(row)
            reason = "provider_error"
            break

        decisions += 1
        row["decision"] = decisions
        row["actor_message"] = reply.content
        conversation.append({"role": "assistant", "content": reply.content})
        try:
            action, rationale = parse_decision(reply.content)
        except (ValueError, TypeError):
            format_errors += 1
            row.update(status="invalid_decision", error="expected_rationale_and_one_action")
            record(row)
            if format_errors >= 2:
                reason = "format_error"
                break
            conversation.append(actor_turn(env.observation, "No action was applied. Return a rationale and one valid action."))
            continue

        result = env.step(action)
        event = {"action": action.value, "event": result.event, "reason": result.reason, "rationale": rationale}
        game_actions += 1
        blocks += result.event == "blocked"
        pushes += result.event in {"push", "solved"}
        row.update(event, status="applied", after=snapshot(result.observation))
        record(row)
        if result.observation.outcome == Outcome.ACTIVE:
            feedback = f"{action.value} -> {result.event}"
            if result.reason:
                feedback += f" ({result.reason})"
            conversation.append(actor_turn(result.observation, feedback + "."))

    if env.observation.outcome != Outcome.ACTIVE:
        reason = env.observation.outcome.value
    return {"model": MODELS[alias][0], "board_id": board.id, "replicate": replicate, "outcome": reason,
            "api_requests": requests_made, "decisions": decisions, "game_actions": game_actions,
            "moves": env.observation.total_moves, "blocks": blocks, "pushes": pushes,
            "resets": env.observation.resets, "invalid_decisions": format_errors,
            "reported_cost_usd": round(cost, 8) if cost_records else None,
            "requests_with_cost": cost_records,
            "requests_with_provider_reasoning": requests_with_provider_reasoning,
            "provider_reasoning_kinds": sorted(provider_reasoning_kinds)}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--models", nargs="+", choices=MODELS, default=DEFAULT_MODELS)
    cli.add_argument("--boards", nargs="+", default=BOARD_IDS)
    cli.add_argument("--decisions", type=int, default=DEFAULT_DECISIONS,
                     help=f"maximum model decisions per episode (default: {DEFAULT_DECISIONS})")
    cli.add_argument("--max-tokens", type=int, default=1000)
    cli.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high"])
    cli.add_argument("--workers", type=int, default=6)
    cli.add_argument("--repeats", type=int, default=1, help="independent episodes per model and board")
    cli.add_argument("--provider-pin", help="exact OpenRouter endpoint tag (when selecting one model)")
    cli.add_argument("--timeout", type=float, default=30)
    cli.add_argument("--run-dir", type=Path, required=True)
    cli.add_argument("--dotenv", type=Path, default=Path(".env"))
    args = cli.parse_args()
    if min(args.decisions, args.max_tokens, args.workers, args.repeats, args.timeout) <= 0:
        cli.error("Limits must be positive")
    if len(set(args.models)) != len(args.models) or len(set(args.boards)) != len(args.boards):
        cli.error("Each model and board should appear once per run")
    boards = load_boards()
    if set(args.boards) - boards.keys():
        cli.error("Unknown actor board ID")
    pins = dict(PINNED_PROVIDERS)
    if args.provider_pin:
        if len(args.models) != 1:
            cli.error("--provider-pin requires exactly one model")
        pins[args.models[0]] = args.provider_pin
    try:
        actor = OpenRouter(read_openrouter_key(args.dotenv), args.max_tokens, args.effort, args.timeout, pins)
    except RuntimeError as error:
        cli.error(str(error))
    args.run_dir.mkdir(parents=True, exist_ok=False)
    config = {"actor": "LLM-human-proxy", "started_utc": datetime.now(timezone.utc).isoformat(),
              "models": {a: {"id": MODELS[a][0], "max_price_usd_per_million": MODELS[a][1:],
                             "reasoning": reasoning_config(a, args.effort), "provider_pin": pins.get(a)}
                         for a in args.models}, "board_ids": args.boards, "decisions": args.decisions,
              "repeats": args.repeats, "workers": args.workers, "timeout_seconds": args.timeout,
              "max_completion_tokens": args.max_tokens, "provider_reasoning_policy": "readable_side_channel_not_replayed",
              "history_policy": "full_episode_append_only", "system_prompt": SYSTEM, "response_format": SCHEMA}
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def job(alias: str, board_id: str, replicate: int) -> dict:
        suffix = f"-r{replicate:02d}" if args.repeats > 1 else ""
        path = args.run_dir / f"{alias}-{board_id}{suffix}.jsonl"
        with path.open("x", encoding="utf-8") as log:
            def record(row):
                print(json.dumps(row, ensure_ascii=True), file=log, flush=True)
            return run_episode(alias, boards[board_id], args.decisions, actor, record, replicate=replicate)

    jobs = [(alias, bid, repeat) for bid in args.boards for alias in args.models
            for repeat in range(1, args.repeats + 1)]
    print(f"LLM-human-proxy: {len(jobs)} independent episodes, {args.decisions} decisions each.", flush=True)
    summaries = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = [pool.submit(job, alias, bid, repeat) for alias, bid, repeat in jobs]
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=True), flush=True)
    summaries.sort(key=lambda row: (row["model"], row["board_id"], row["replicate"]))
    (args.run_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
