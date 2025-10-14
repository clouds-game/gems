<!--
Agents (AI / Bots) for the Gem-Merchant engine

This file describes a small, practical contract for implementing player agents
that the engine can call interchangeably. Keep agents simple, deterministic
(when seeded), and side-effect-free so matches and tests stay reliable.
-->

# Agent contract (summary)

An agent is a Python object that the engine can construct and call to pick
actions. The minimal contract is:

- __init__(seat_id: int, rng: Optional[random.Random] = None)
- act(state: GameState, legal_actions: Sequence[Action], *, timeout: Optional[float]=None) -> Action
- reset(seed: Optional[int] = None) -> None  (optional but recommended)
- observe(player: PlayerState, state: GameState) -> None  (optional; for informational updates)

Rules the engine expects:

- act must return one of the provided `legal_actions` (by identity or value)
    before the optional `timeout`. If the agent fails to return in time the
    engine may apply a fallback action or forfeit the player.
- act must not mutate `state` or `legal_actions` objects.
- Agents must be deterministic for a fixed RNG seed (use the provided RNG or
    re-seed internal RNGs in `reset`).

## Data shapes (recommended)

- GameState: read-only object provided by the engine with public game data
    (players, board, bank, visible cards, scores, turn number, etc.).
- Action: small typed object or dict; prefer {"type": str, "payload": dict}.
- legal_actions: Sequence[Action] computed by the engine — choose one of
    these exact objects.

Treat actions as opaque: read fields as needed, but expect the engine to
serialize/compare actions according to its own rules.

## Minimal example base class

Use this as a starting point. Keep implementations small and fast.

```python
import random
from typing import Optional, Sequence

class Agent:
    def __init__(self, seat_id: int, rng: Optional[random.Random] = None):
        self.seat_id = seat_id
        # Use a local RNG instance to guarantee reproducible behavior
        self.rng = rng or random.Random()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def observe(self, player, state) -> None:
        # optional hook: receive state updates for internal bookkeeping
        pass

    def act(self, state, legal_actions: Sequence, *, timeout: Optional[float] = None):
        """Return one element from legal_actions."""
        raise NotImplementedError()
```

## Small example agents

RandomAgent: picks uniformly from legal actions using the provided RNG.

```python
class RandomAgent(Agent):
    def act(self, state, legal_actions, *, timeout=None):
        if not legal_actions:
            raise ValueError("No legal actions available")
        return self.rng.choice(legal_actions)
```

GreedyAgent: evaluates actions with a tiny fast heuristic and picks the best.

```python
def quick_score(state, seat_id, action) -> float:
    # Implement a fast, shallow heuristic specific to the engine's accessors.
    # Keep this deterministic and cheap.
    return 0.0

class GreedyAgent(Agent):
    def act(self, state, legal_actions, *, timeout=None):
        best = None
        best_score = float('-inf')
        for a in legal_actions:
            s = quick_score(state, self.seat_id, a)
            if s > best_score:
                best_score = s
                best = a
        return best
```

## Determinism and timeouts

- The engine may call `reset(seed)` before matches. Agents must use the
    provided RNG or reseed their own RNGs from that seed to ensure repeatable
    behavior.
- If your agent uses multiple RNGs (e.g. per-simulation), derive them from a
    single seeded RNG to avoid cross-talk.
- Respect the `timeout` argument: if you run planning/search, make the
    planner interruptible and always have a quick fallback action ready.

## Safety and best practices

- Avoid blocking I/O in `act`.
- Do not mutate the `state` or shared objects.
- Keep per-turn allocations small to reduce GC and improve tournament
    stability.

## Testing guidance

- Determinism: seed an agent and assert repeated runs produce identical
    actions for the same sequence of inputs.
- Legality: assert `act` always returns an element from `legal_actions`.
- Timeout behavior: call `act` with a small timeout and verify it returns a
    fallback action or that the engine policy handles the timeout.

Example pytest snippet:

```python
def test_random_agent_chooses_legal():
    import random
    agent = RandomAgent(seat_id=0, rng=random.Random(1))
    state = make_test_state()
    legal = compute_legal_actions(state, seat=0)
    action = agent.act(state, legal)
    assert action in legal
```

## Where to put agents in this repo

- Suggested path for implementations: `gems/agents/` (create as needed).
- Tests: `tests/test_agents.py` or extend existing engine tests in `tests/`.

Note: the helper `PlayerState.can_afford` already accounts for permanent
discounts awarded by purchased cards when evaluating affordability. Agents
can rely on this behavior when constructing heuristics or simulating
possible purchases.

## Quick checklist

1. Provide __init__(seat_id, rng=None).
2. Implement act(state, legal_actions, *, timeout=None) -> member of legal_actions.
3. Support reset(seed) for determinism.
4. Add unit tests for determinism and legality.

---

If you want, I can now:

- Add `gems/agents/random_agent.py` and `gems/agents/greedy_agent.py` with
    minimal implementations and tests.
- Add a small test harness under `tests/test_agents.py` and run pytest.

Tell me which next step you'd like.

<!--
The following instruction are written by user and should be KEPT as is.
-->
# Python
- TAKE CARE OF THE INDENTATION.
- All Python code should have indentation of 2 spaces per level.
- You could run `uv add` to add some dependencies when needed.
- use `runTests` provided by vscode to run the tests.
- prefer modifying existing test cases instead of adding new ones.
- we don't use `if __name__ == "__main__":` or cli in our code.
- we use `# %%` to split cells in scripts for vscode, consider it as a notebook.
