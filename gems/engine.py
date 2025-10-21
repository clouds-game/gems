"""Game engine helpers: initialization and summarization utilities.

This module provides a small, well-documented public API the rest of the
project (and agents) can import: `init_game` and `print_summary`.

The functions intentionally mirror the simple helpers used during
development in the repository root so callers have a stable, package-level
entrypoint.
"""

from typing import TypeVar
from collections.abc import Sequence

from .agents.core import Agent, BaseAgent
from .consts import GameConfig

from .typings import ActionType, Gem, Card, Role
from .state import PlayerState, GameState
from .actions import Action
from pathlib import Path
import random


class Engine:
  """A tiny, stateful wrapper around the engine helpers.

  This class is intentionally small and suitable for development, tests,
  and simple interactive sessions. It stores the current `GameState` and
  exposes a couple of convenience methods:

  - `reset(...)` to re-initialize the game
  - `get_state()` to access the current immutable GameState
  - `print_summary()` to display a human readable summary
  """

  config: GameConfig
  _num_players: int
  _state: GameState
  decks_by_level: dict[int, list[Card]]
  roles_deck: list[Role]
  _rng: random.Random
  _action_history: list[Action]
  _actions_to_replay: list[Action]
  def __init__(
      self,
      *,
      num_players: int,
      names: list[str] | None,
      state: GameState,
      decks_by_level: dict[int, list[Card]],
      roles_deck: list[Role],
      rng: random.Random,
      config: GameConfig | None = None,
      seed: int | None = None,
      all_noops_last_round: bool = False,
      action_history: list[Action] | None = None,
  ) -> None:
    self._num_players = num_players
    self._names = names
    self._state = state
    self.decks_by_level = decks_by_level
    self.roles_deck = roles_deck
    self._rng = rng
    # store/use provided configuration (fall back to sensible default)
    self.config = config or GameConfig(num_players=num_players)
    # preserve the seed used to construct the RNG when available so serialized
    # engines can be reproduced deterministically
    self._seed = seed
    self._all_noops_last_round = all_noops_last_round
    self._action_history = list(action_history) if action_history is not None else []
    self._actions_to_replay = []

  @staticmethod
  def new(
      num_players: int = 4,
      names: list[str] | None = None,
      seed: int | None = None,
      config: GameConfig | None = None,
  ) -> "Engine":
    # basic validation: at least 1 player
    if num_players < 1:
      raise ValueError("num_players must be positive")

    cfg = config or GameConfig(num_players=num_players)
    # create minimal starting state using the provided config
    state = Engine.create_game(num_players, names, cfg)
    engine = Engine(
      num_players=num_players,
      names=names,
      state=state,
      decks_by_level={},
      roles_deck=[],
      rng=random.Random(seed),
      config=cfg,
      seed=seed,
    )
    engine.load_and_shuffle_assets()
    visible_cards: list[Card] = []
    for lvl in engine.config.card_levels:
      drawn = engine.draw_from_deck(lvl, engine.config.card_visible_count)
      visible_cards.extend(reversed(drawn))
    roles_to_draw = engine._num_players + 1
    visible_roles = []
    for _ in range(min(roles_to_draw, len(engine.roles_deck))):
      visible_roles.append(engine.roles_deck.pop())
    engine._state = GameState(config=cfg, players=engine._state.players, bank=engine._state.bank,
                              visible_cards_in=visible_cards, visible_roles_in=visible_roles,
                              turn=engine._state.turn)
    engine._all_noops_last_round = False
    engine._action_history = []
    return engine

  def clone(self, seed: int | None = None) -> "Engine":
    engine = Engine(
        num_players=self._num_players,
        names=self._names,
        state=self._state,
        decks_by_level={lvl: list(deck) for lvl, deck in self.decks_by_level.items()},
        roles_deck=list(self.roles_deck),
        rng=random.Random(seed),  # new RNG instance
        config=self.config,
        seed=self._seed,
        all_noops_last_round=self._all_noops_last_round,
        action_history=list(self._action_history),
    )
    return engine

  def serialize(self) -> dict:
    """Return a JSON-serializable dict describing this Engine.

    The serialized form includes a minimal set of fields requested by
    consumers/tests: number of players, player names, seed used to create
    the RNG (if any), and the action history as a list of action dicts.
    """

    return {
      'num_players': self._num_players,
      'config': self.config.serialize(),
      'names': self._names,
      'seed': self._seed,
      'action_history': [a.serialize() for a in self._action_history],
    }

  @classmethod
  def deserialize(cls, d: dict) -> "Engine":
    """Reconstruct an Engine from a dict produced by `serialize`.

    This creates a fresh Engine via `Engine.new(...)` using the stored
    num_players/names/seed and then repopulates the `action_history`
    with deserialized Action objects. Note: the returned Engine is a
    fresh instance (assets shuffled using the seed) and does not replay
    the action history against the GameState.
    """
    num_players = d.get('num_players')
    if num_players is None:
      raise ValueError("deserialize requires 'num_players' field")
    config_dict = d.get('config')
    if config_dict is None:
      raise ValueError("deserialize requires 'config' field")
    config = GameConfig.deserialize(config_dict)
    num_players = int(num_players)
    names = d.get('names')
    seed = d.get('seed', None)
    if seed is not None:
      seed = int(seed)
    engine = cls.new(num_players=num_players, names=names, seed=seed, config=config)
    raw_actions = d.get('action_history', []) or []
    actions: list[Action] = []
    for a in raw_actions:
      actions.append(Action.deserialize(a))
    # store as actions needing replay; caller may choose to call apply_replay()
    engine._actions_to_replay = actions
    return engine

  def replay(self, actions: Sequence[Action] | None = None) -> list[GameState]:
    """Apply a sequence of actions to the engine, returning intermediate states.

    Parameters:
      actions: Optional sequence of `Action` objects to apply in order. If
        omitted (or None) the engine will apply any actions stored in the
        internal replay buffer (`_actions_to_replay`), which is the list of
        actions produced by `Engine.deserialize`.

    Behavior:
      - The first element of the returned list is the current state *before*
        any actions are applied.
      - Each subsequent element is the new immutable `GameState` after that
        action has been applied.
      - Successfully applied actions are appended to `_action_history`.
      - When replaying (actions is None) the `_actions_to_replay` buffer is
        cleared after successful application.

    Returns:
      list[GameState]: `[state_before, state_after_action1, ...]`.

    Raises:
      ValueError: if any action cannot be applied to the current state.
    """
    to_apply: list[Action]
    if actions is None:
      # use (and then clear) replay buffer
      to_apply = list(self._actions_to_replay)
    else:
      to_apply = list(actions)
    state_list: list[GameState] = [self._state]
    for action in to_apply:
      self._state = action.apply(self._state)
      self.advance_turn()
      self._action_history.append(action)
      state_list.append(self._state)
    if actions is None:
      # clear replay buffer only when we consumed it implicitly
      self._actions_to_replay = []
    return state_list

  @staticmethod
  def create_game(num_players: int = 4, names: list[str] | None = None, config: GameConfig | None = None) -> GameState:
    """Create and return a minimal starting GameState.

    - num_players: between 2 and 4 (inclusive).
    - names: optional list of player display names; defaults to "Player 1"...
    """
    cfg = config or GameConfig(num_players=num_players)

    names = names or [f"Player {i + 1}" for i in range(num_players)]
    if len(names) < num_players:
      # extend with default names if caller provided too few
      names = names + [f"Player {i + 1}" for i in range(len(names), num_players)]

    players = [PlayerState(seat_id=i, name=names[i]) for i in range(num_players)]

    coin_default_init = cfg.coin_init
    # Typical gem counts for a game (from GameConfig):
    bank = (
      (Gem.RED, coin_default_init),
      (Gem.BLUE, coin_default_init),
      (Gem.WHITE, coin_default_init),
      (Gem.BLACK, coin_default_init),
      (Gem.GREEN, coin_default_init),
      (Gem.GOLD, cfg.coin_gold_init),
    )

    visible_cards = tuple()

    return GameState(config=cfg, players=tuple(players), bank_in=bank, visible_cards_in=visible_cards, turn=0)

  def reset(self, num_players: int | None = None, names: list[str] | None = None) -> None:
    """Reset the engine's internal GameState.

    If `num_players` or `names` are omitted the values provided at
    construction time are used.
    """
    if num_players is None:
      num_players = self._num_players
    if names is None:
      names = self._names
    self._state = self.create_game(num_players, names, self.config)
    self._num_players = num_players
    self._names = names

  def get_state(self) -> GameState:
    """Return the current (immutable) GameState object."""
    return self._state

  def print_summary(self) -> None:
    """Print a short, human-readable summary of the given GameState.

    This is a convenience for development and quick debugging; callers should
    avoid parsing the printed output in tests.
    """
    print("--" * 20)
    print(f"Round: {self._state.round} Turn: {self._state.turn}")
    print("Players:")
    for p in self._state.players:
      print(f"  seat={p.seat_id} name={p.name!r} score={p.score} gems={p.gems.normalized()} discounts={p.discounts.normalized()} cards={len(p.purchased_cards)} reserved={len(p.reserved_cards)}")
    print(f"Bank: {self._state.bank.normalized()}")
    # Build a simple table: deck-count followed by visible card titles for each level
    cards_table: list[str] = []
    for lvl in self.config.card_levels:
      deck_count = len(self.decks_by_level.get(lvl, ()))
      visible = [str(c) for c in self._state.visible_cards.get_level(lvl)]
      if visible:
        line = f"{deck_count:3d}\t" + "\t".join([f"{s:25}" for s in visible])
      else:
        line = f"{deck_count:3d}\t"
      cards_table.append(line)

    print("Visible cards:\n" + "\n".join([line for line in cards_table if line.strip() and not line.strip().startswith('0')]))

  def load_and_shuffle_assets(self, path: str | None = None) -> None:
    """Load assets from disk and shuffle them into decks on this Engine.

    - path: optional path to config file (falls back to package assets)

    The engine's RNG (self._rng) is used for deterministic shuffling when
    seeded at Engine construction.
    """
    cards, roles = load_assets(path)
    levels, roles_list = shuffle_assets(cards, roles, rng=self._rng)
    # store on instance for consumers
    self.decks_by_level = levels
    self.roles_deck = roles_list

  def get_deck(self, level: int) -> list[Card]:
    return list(self.decks_by_level.get(level, []))

  def get_roles(self) -> list[Role]:
    return list(self.roles_deck)

  def draw_from_deck(self, level: int, n: int = 1) -> list[Card]:
    """Remove and return up to `n` cards from the deck of `level`.

    Pops from the end of the level list (treating the end as the top of the
    deck) which is efficient for Python lists.
    """
    deck = self.decks_by_level.get(level, [])
    drawn: list[Card] = []
    for _ in range(min(n, len(deck))):
      drawn.append(deck.pop())
    return drawn

  def peek_deck(self, level: int, n: int = 1) -> list[Card]:
    """Return up to `n` cards from the top of the deck without removing them."""
    deck = self.decks_by_level.get(level, [])
    if not deck:
      return []
    return list(deck[-n:]) if n > 0 else []

  def get_legal_actions(self, seat_id: int | None = None) -> list[Action]:
    """Return a list of legal `Action` objects for the given player seat.

    This is a lightweight implementation used by tests and agents to
    enumerate plausible moves. It intentionally implements a permissive
    set of actions:

    - take_3_different: any triple of distinct gem types with at least 1 in
      the bank.
    - take_2_same: any gem type with at least 4 tokens in the bank (caller may
      choose a stricter rule if desired).
    - reserve_card: reserve any visible card (present on `visible_cards`).
    - buy_card: a permissive buy action for visible cards (detailed payment
      validation is left to the engine's apply/validation logic).

    The method does not mutate engine state.
    """
    seat_id = seat_id if seat_id is not None else self._state.turn % len(self._state.players)
    return self._state.players[seat_id].get_legal_actions(self._state)

  def advance_turn(self) -> None:
    """Advance the turn to the next player.

    This is a convenience method that updates the internal GameState's
    `turn` counter. It does not mutate any other part of the state.
    """
    self._state = self._state.advance_turn(self.decks_by_level)

  def play_one_round(self, agents: list[BaseAgent], debug=True) -> None:
    """Play a full round (one turn per player) using specific Agents.

    This is a convenience for quick simulations and testing. It does not
    check for game end conditions.
    """
    all_noops = True
    num_players = len(self._state.players)
    for seat in range(num_players):
      state = self.get_state()
      agent = agents[seat]
      actions = self.get_legal_actions(seat)
      if not all(a.type == ActionType.NOOP for a in actions):
        all_noops = False
      action = agent.act(state, actions)
      if debug:
        print(f"Turn {state.turn} — player {seat} performs: {action}")
      # apply action and update engine state
      self._state = action.apply(state)
      self._action_history.append(action)
      # print a brief summary after the move
      if debug:
        self.print_summary()
      self.advance_turn()
    if all_noops:
      self._all_noops_last_round = True

  def game_end(self) -> bool:
    """Return True if any player has reached the winning score (15 points)."""
    if self._all_noops_last_round:
      return True
    winners = self.game_winners()
    return len(winners) > 0

  def game_winners(self) -> list[PlayerState]:
    """Return a list of players who have reached the winning score (15 points)."""
    return [p for p in self._state.players if p.score >= 15]


def load_assets(path: str | None = None):
  """Load cards and roles from a JSON config file and return (cards, roles).

  The config file is expected to contain top-level `cards` and `roles` arrays
  matching the `Card.from_dict` / `Role.from_dict` shapes.
  """
  import yaml
  p = Path(path) if path is not None else Path(__file__).parent / "assets" / "config.yaml"
  with p.open('r', encoding='utf8') as fh:
    j = yaml.safe_load(fh)

  cards = [Card.from_dict(c) for c in j.get('cards', [])]
  roles = [Role.from_dict(r) for r in j.get('roles', [])]
  return cards, roles


def shuffle_assets(cards: Sequence[Card], roles: Sequence[Role], rng: random.Random | None = None):
  """Shuffle cards by level and shuffle roles.

  Returns a dict mapping level->list[Card] and a list of roles. The RNG may
  be a `random.Random` instance; if omitted a new one is created.
  """
  rng = rng or random.Random()
  # group cards by level
  levels: dict[int, list[Card]] = {}
  for c in cards:
    levels.setdefault(c.level, []).append(c)

  # shuffle each level's deck
  for lvl, lst in levels.items():
    rng.shuffle(lst)

  # shuffle roles
  roles_list = list(roles)
  rng.shuffle(roles_list)

  return levels, roles_list
