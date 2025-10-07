# Agents (AI / Bots) for Gem-Merchant (engine-only)

This document describes the recommended agent contract, data shapes, examples, testing guidance, and determinism rules for implementing players (human wrappers and AIs) for the Gem-Merchant game engine in this repository.

Use this as the single source of truth for how the engine will call into player code. Keeping agents simple, deterministic (when seeded), and side-effect free makes automated matches, tournaments, and unit tests reliable.

## Goals

- Define a minimal, clear Python interface for agents so the engine can call them interchangeably.
- Provide small example agents (random, greedy) as templates.
- Document edge cases, error handling, and determinism requirements.
- Describe testing and evaluation harness suggestions.

## Contract (summary)

An Agent is a Python object with the following minimal behavior:

- Construction: Agent(seat_id: int, rng: Optional[random.Random] = None)
- act(state: GameState, legal_actions: Sequence[Action], *, timeout: Optional[float]=None) -> Action
- observe(state: GameState) -> None  (optional; called for information-only updates)
- reset(seed: Optional[int] = None) -> None  (optional; reset internal state / re-seed RNG)

Requirements/assumptions the engine will make when calling an agent:

- act is synchronous and must return one Action from the provided legal_actions within the timeout if provided. If the agent fails to return in time, the engine may apply a default action or forfeit the player.
- act must not mutate the passed `state` or the `legal_actions` objects. Treat inputs as read-only.
- Agents must be deterministic for a fixed RNG seed. If a seed is provided via reset(seed) or in construction, repeated runs should produce the same decisions.
- Agents may keep internal state (memory) between turns, but that state must be reset via reset().

## Data shapes (recommended)

- GameState: An engine defined read-only object representing the complete public game state (players, board, bank, visible cards, scores, turn number, etc.). Do not rely on private engine internals; use the stable API functions provided by the engine.
- Action: Prefer a small typed object or simple dict with keys `type` (str) and `payload` (dict). Examples: {"type": "take_gems", "payload": {"gems": {"ruby":1,"sapphire":1}}}
- Legal actions: Sequence[Action] — the engine computes these and passes them to act. Agents must pick one of these exact objects (identity or value equality depending on engine implementation).

Example minimal Action dataclass (engine side)

```python
# engine-side example (for reference)
from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    type: str
    payload: dict
```

Agents should treat Action as opaque except for reading fields to make decisions.

## Recommended Agent base class (example)

```python
import random
from typing import Optional, Sequence

class Agent:
    def __init__(self, seat_id: int, rng: Optional[random.Random] = None):
        self.seat_id = seat_id
        self.rng = rng or random.Random()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

    def observe(self, state):
        # optional: receive each full state after every action
        pass

    def act(self, state, legal_actions: Sequence, *, timeout: Optional[float] = None):
        """Return one element from legal_actions."""
        raise NotImplementedError
```

## Simple example agents

- RandomAgent: selects a random legal action using the provided RNG.
- GreedyAgent: scores each legal action with a fast heuristic and picks the highest score.

Example RandomAgent:

```python
class RandomAgent(Agent):
    def act(self, state, legal_actions, *, timeout=None):
        if not legal_actions:
            raise ValueError("No legal actions available")
        return self.rng.choice(legal_actions)
```

Example GreedyAgent (very small heuristic):

```python
class GreedyAgent(Agent):
    def act(self, state, legal_actions, *, timeout=None):
        # score action by a quick heuristic defined by the engine's accessor functions
        best = None
        best_score = float('-inf')
        for a in legal_actions:
            s = quick_score(state, self.seat_id, a)
            if s > best_score:
                best_score, best = s, a
        return best
```

Note: `quick_score` is a tiny scoring function you implement in the agent or shared evaluation utilities. Keep it very fast — avoid deep simulations unless you implement a timeout-aware planner.

## Advanced agents

- MCTS / Monte-Carlo: run many random playouts; be careful to respect timeout and seeding. Use deterministic pseudo-random streams and consider using a separate RNG instance per simulation to avoid cross-talk.
- Search-based: if you implement minimax/MCTS, ensure the planner can be cleanly interrupted (respect `timeout`) and returns the best found legal action so far.

Guidelines:

- Keep per-turn allocations small to reduce GC pauses for tournaments.
- If an agent spawns threads or subprocesses, the engine must be aware — prefer single-threaded agents unless explicitly allowed.

## Determinism and seeding

- The engine will call agent.reset(seed=N) before matches if deterministic behavior is required. Agents must use the provided RNG or accept the provided seed and seed all internal RNGs accordingly.
- Avoid using global randomness (random.random()) without seeding from the engine-provided RNG.

## Timeouts and safety

- The engine may pass a `timeout` (seconds) to `act`. Agents should return before the timeout. Implementations that cannot meet real-time deadlines should still return quickly and rely on simpler heuristics.
- If an agent raises an exception during act, the engine should treat it as a fault; possible engine policies: pass, skip turn, random fallback action, or immediate forfeit. Document the chosen policy in the engine README and tests.

## Testing agents

Unit test ideas:

- Determinism: seed the agent, run the same match twice (same RNG seeds) and assert identical decisions.
- Legal action selection: assert act only returns an action from legal_actions for a variety of states.
- Timeout behavior: call act with a very small timeout and assert the agent either returns quickly or the engine fallback is triggered.
- Integration smoke: run headless matches between RandomAgent instances for N games and assert no exceptions and valid score ranges.

Example minimal test (pytest style):

```python
def test_random_agent_chooses_legal():
    agent = RandomAgent(seat_id=0, rng=random.Random(1))
    state = make_test_state()
    legal = compute_legal_actions(state, seat=0)
    action = agent.act(state, legal)
    assert action in legal
```

## Logging and debugging

- Agents may log decisions (level DEBUG). Keep logs optional so bulk tournaments can silence them.
- Provide hooks in the engine to capture per-turn chosen action and brief reasoning text (a short string). Do not log full states unless explicitly requested.

## Serialization & persistence

- If an agent needs to be persisted (for training checkpoints), expose a `to_bytes()/from_bytes()` pair or use `pickle` carefully. Document the format and be mindful of cross-version compatibility.

## Multi-agent tournaments

- Tournament harnesses should run many matches with deterministic reseeding. For fairness, randomize seat assignment between runs and record seeds to allow exact replay.

## Edge cases and pitfalls

- Empty legal_actions: engine should never pass this; if it happens, agent should raise or return a special NOOP action if supported.
- Mutating the GameState: agents must treat the state as immutable.
- Blocking I/O: agents must not perform long blocking I/O during act. If necessary, prefetch outside act.

## Quick checklist for implementing a new agent

1. Provide a constructor Agent(seat_id, rng=None).
2. Implement act(state, legal_actions, *, timeout=None) and always return a member of legal_actions.
3. Use engine-provided accessors; do not depend on engine internals.
4. Support reset(seed) to re-seed internal RNGs.
5. Add unit tests for determinism and legality.

---

If you'd like, I can:

- Add concrete example agent implementations under `gems/agents/` (RandomAgent and GreedyAgent).
- Add a small test harness and a few pytest tests that run headless matches.

Tell me which of those you'd like next and I'll implement them.

# Code style
- All Python code should have indentation of 2 spaces per level.
