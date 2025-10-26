"""Game engine helpers: initialization and summarization utilities.

This module provides a small, well-documented public API the rest of the
project (and agents) can import: `init_game` and `print_summary`.

The functions intentionally mirror the simple helpers used during
development in the repository root so callers have a stable, package-level
entrypoint.
"""

from typing import Any
from collections.abc import Sequence

from pydantic import BaseModel

from .agents.core import Agent, BaseAgent
from .consts import GAME_ASSETS_DEFAULT, GAME_ASSETS_EMPTY, GameAssets, GameConfig

from .typings import ActionType, Gem, Card, Role
from .state import PlayerState, GameState
from .actions import Action, BuyCardAction, NoopAction, ReserveCardAction, Take2Action, Take3Action
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
  _initial_assets: GameAssets

  def __init__(
      self,
      *,
      num_players: int,
      names: list[str] | None,
      state: GameState,
      assets: GameAssets,
      decks_by_level: dict[int, list[Card]] | None = None,
      roles_deck: list[Role] | None = None,
      rng: random.Random,
      config: GameConfig | None = None,
      seed: int | None = None,
      all_noops_last_round: bool = False,
      action_history: list[Action] = [],
  ) -> None:
    self._num_players = num_players
    self._names = names
    self._state = state
    self._initial_assets = assets
    self.decks_by_level = assets.new_decks_by_level() if decks_by_level is None else decks_by_level
    self.roles_deck = assets.new_roles_deck() if roles_deck is None else roles_deck
    self._rng = rng
    # store/use provided configuration (fall back to sensible default)
    self.config = config or GameConfig(num_players=num_players)
    # preserve the seed used to construct the RNG when available so serialized
    # engines can be reproduced deterministically
    self._seed = seed
    self._all_noops_last_round = all_noops_last_round
    self._action_history = action_history

  @staticmethod
  def new(
      num_players: int | None = None,
      names: list[str] | None = None,
      seed: int | None = None,
      config: GameConfig | None = None,
  ) -> "Engine":
    # basic validation: at least 1 player

    # create minimal starting state using the provided config
    state = Engine.create_game(num_players, names, config)
    config = state.config
    num_players = state.config.num_players
    assets = GAME_ASSETS_DEFAULT.shuffle(seed)
    engine = Engine(
      num_players=num_players,
      names=names,
      state=state,
      assets=assets,
      rng=random.Random(seed),
      config=config,
      seed=seed,
    )
    visible_cards: list[Card] = []
    for lvl in engine.config.card_levels:
      drawn = engine.draw_from_deck(lvl, engine.config.card_visible_count)
      visible_cards.extend(reversed(drawn))
    roles_to_draw = engine._num_players + 1
    visible_roles = []
    for _ in range(min(roles_to_draw, len(engine.roles_deck))):
      visible_roles.append(engine.roles_deck.pop())
    engine._state = GameState(config=config, players=engine._state.players, bank=engine._state.bank,
                              visible_cards_in=visible_cards, visible_roles_in=visible_roles,
                              turn=engine._state.turn)
    engine._all_noops_last_round = False
    engine._action_history = []
    return engine

  def step(self, action: Action) -> GameState:
    """Apply the given action to the current GameState, updating engine state."""
    self._state = action.apply(self._state)
    self._action_history.append(action)
    self.advance_turn()
    return self._state

  def clone(self, seed: int | None = None) -> "Engine":
    engine = Engine(
        num_players=self._num_players,
        names=self._names,
        state=self._state,
        decks_by_level={lvl: list(deck) for lvl, deck in self.decks_by_level.items()},
        roles_deck=list(self.roles_deck),
        assets=self._initial_assets,
        rng=random.Random(seed),  # new RNG instance
        config=self.config,
        seed=self._seed,
        all_noops_last_round=self._all_noops_last_round,
        action_history=list(self._action_history),
    )
    return engine


  def export(self) -> "Replay":
    """Export this Engine's state and history as a Replay object."""
    return Replay(
      config=self.config,
      assets=self._initial_assets,
      player_names=self._names or [],
      action_history=self._action_history, # type: ignore
      metadata={
        'seed': self._seed,
      },
    )


  @staticmethod
  def create_game(num_players: int | None = None, names: list[str] | None = None, config: GameConfig | None = None) -> GameState:
    """Create and return a minimal starting GameState.

    - num_players: between 2 and 4 (inclusive).
    - names: optional list of player display names; defaults to "Player 1"...
    """
    if num_players is None:
      if config is None:
        raise ValueError("Either num_players or config must be provided")
      num_players = config.num_players
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

  def reset(self, names: list[str] | None = None) -> None:
    """Reset the engine's internal GameState.

    If `num_players` or `names` are omitted the values provided at
    construction time are used.
    """
    names = names or self._names
    if names is not None:
      assert len(names) == self._num_players
    self._state = self.create_game(self._num_players, names, self.config)
    self._num_players = self._num_players
    self._names = names
    self.decks_by_level = self._initial_assets.new_decks_by_level()
    self.roles_deck = self._initial_assets.new_roles_deck()
    self._action_history = []
    self._all_noops_last_round = False

  def get_state(self) -> GameState:
    """Return the current (immutable) GameState object."""
    return self._state

  def print_summary(self) -> None:
    """Print a short, human-readable summary of the given GameState.

    This is a convenience for development and quick debugging; callers should
    avoid parsing the printed output in tests.
    """
    # print("--" * 20)
    # print(f"Round: {self._state.round} Turn: {self._state.turn}")
    # print("Players:")
    # for p in self._state.players:
    #   print(f"  seat={p.seat_id} name={p.name!r} score={p.score} gems={p.gems.normalized()} discounts={p.discounts.normalized()} cards={len(p.purchased_cards)} reserved={len(p.reserved_cards)}")
    # print(f"Bank: {self._state.bank.normalized()}")
    self._state.print_summary(show_visible_cards=False)
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

  def play_one_round(self, agents: list[BaseAgent], debug=True) -> GameState:
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
      # apply action and update engine state
      self._state = action.apply(state)
      self._action_history.append(action)
      if debug:
        print(f"Turn {state.turn} — player {seat} performs: {action}")
        self.print_summary()
      self.advance_turn()
    if all_noops:
      self._all_noops_last_round = True
    return self._state

  def game_end(self) -> bool:
    """Return True if any player has reached the winning score (15 points)."""
    if self._all_noops_last_round:
      return True
    winners = self.game_winners()
    return len(winners) > 0

  def game_winners(self) -> list[PlayerState]:
    """Return a list of players who have reached the winning score (15 points)."""
    return [p for p in self._state.players if p.score >= 15]


class Replay(BaseModel):
  config: GameConfig
  assets: GameAssets
  player_names: list[str]
  action_history: list[Take3Action | Take2Action | ReserveCardAction | BuyCardAction | NoopAction]
  metadata: dict[str, Any] # seed and others

  def replay(self) -> tuple[list[GameState], Engine]:
    """Replay the stored action history, returning the list of GameStates."""
    engine = Engine.new(
      num_players=self.config.num_players,
      names=self.player_names,
      seed=self.metadata.get('seed', None),
      config=self.config,
      )
    states: list[GameState] = [engine.get_state()]
    for actions_one_round in [self.action_history[i:i + engine._num_players] for i in range(0, len(self.action_history), engine._num_players)]:
      for action in actions_one_round:
        engine._state = action.apply(engine._state)
        engine.advance_turn()
      states.append(engine.get_state())
    return states, engine
