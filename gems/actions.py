from dataclasses import dataclass, field
from typing import Optional, Mapping, Tuple
from abc import ABC, abstractmethod

from .consts import COIN_MIN_COUNT_TAKE2_IN_DECK
from .typings import Gem, ActionType, GemList, Card
from .state import PlayerState, GameState
from .utils import _replace_tuple


@dataclass(frozen=True)
class Action(ABC):
  """Minimal base Action used as a type tag for polymorphism.

  Concrete action types should subclass this and add strongly-typed
  fields. Keeping this minimal lets engine and agents switch on
  `type` safely.
  """
  type: ActionType

  @classmethod
  def take3(cls, *gems: Gem) -> 'Take3Action':
    return Take3Action.create(*gems)

  @classmethod
  def take2(cls, gem: Gem, count: int = 2) -> 'Take2Action':
    return Take2Action.create(gem, count)

  @classmethod
  def buy(cls, card: Card, payment: Optional[Mapping[Gem, int]] = None) -> 'BuyCardAction':
    return BuyCardAction.create(card, payment=payment)

  @classmethod
  def reserve(cls, card: Card, take_gold: bool = True) -> 'ReserveCardAction':
    return ReserveCardAction.create(card, take_gold=take_gold)

  @classmethod
  def noop(cls) -> 'NoopAction':
    return NoopAction.create()

  def apply(self, state: GameState) -> GameState:
    """Apply this action for the current-turn player and return a new GameState.

    This is a thin wrapper that validates the action can be applied
    for the current player in the given state, then calls the concrete
    `_apply` method implemented by subclasses.

    Implementations must not mutate the provided `state` and must
    return a new GameState with `last_action` set to this action.
    Implementations MUST NOT modify the `turn` field; turn advancement
    is the responsibility of the caller/engine.
    """
    num_players = len(state.players)
    if num_players == 0:
      raise ValueError("state must contain at least one player")

    seat = state.turn % num_players
    player = state.players[seat]

    if not self._check(player, state):
      raise ValueError(f"Action {self} cannot be applied for player {player} in state {state}")

    return self._apply(player, state)

  @abstractmethod
  def _apply(self, player: PlayerState, state: GameState) -> GameState:
    """Apply this action for the current-turn player and return a new GameState.

    Concrete action classes must implement this. Implementations must not
    mutate the provided `state` and must return a new GameState with
    `last_action` set to this action. Implementations MUST NOT modify
    the `turn` field; turn advancement is the responsibility of the
    caller/engine.
    """

  @abstractmethod
  def _check(self, player: PlayerState, state: GameState) -> bool:
    """Return True if this action could be applied for `player` in `state`.
    Implementations should perform non-mutating validation equivalent to
    what `apply` would enforce.
    """


@dataclass(frozen=True)
class Take3Action(Action):
  gems: Tuple[Gem, ...] = field(default_factory=tuple)

  @classmethod
  def create(cls, *gems: Gem) -> 'Take3Action':
    return cls(type=ActionType.TAKE_3_DIFFERENT, gems=tuple(gems))

  def __str__(self) -> str:
    gem_str = ''.join(g.short_str() for g in self.gems)
    return f"Action.Take3(gems=[{gem_str}])"

  def _apply(self, player: PlayerState, state: GameState) -> GameState:
    # Mutable working copies: convert GemList/tuples into mutable dicts/lists
    bank = dict(state.bank)
    player_gems = dict(player.gems)
    visible_cards = list(state.visible_cards)

    # remove one of each requested gem from bank and add to player's gems
    for g in getattr(self, 'gems', ()):  # type: ignore[attr-defined]
      if bank.get(g, 0) <= 0:
        raise ValueError(f"Not enough {g} in bank to take")
      bank[g] = bank.get(g, 0) - 1
      player_gems[g] = player_gems.get(g, 0) + 1

    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems_in=player_gems, score=player.score,
                             reserved_cards_in=player.reserved_cards,
                             purchased_cards_in=player.purchased_cards)
    players = _replace_tuple(state.players, player.seat_id, new_player)

    return GameState(players=players, bank_in=bank,
                     visible_cards_in=visible_cards, turn=state.turn,
                     last_action=self)

  def _check(self, player: PlayerState, state: GameState) -> bool:
    bank = dict(state.bank)
    for g in self.gems:
      if bank.get(g, 0) <= 0:
        return False
    return True


@dataclass(frozen=True)
class Take2Action(Action):
  gem: Gem
  count: int = 2

  @classmethod
  def create(cls, gem: Gem, count: int = 2) -> 'Take2Action':
    return cls(type=ActionType.TAKE_2_SAME, gem=gem, count=count)

  def __str__(self) -> str:
    if self.count != 2:
      return f"Action.Take2({self.gem.short_str()}{self.count})"
    return f"Action.Take2({self.gem.short_str()})"

  def _apply(self, player: PlayerState, state: GameState) -> GameState:
    # Mutable working copies: convert GemList/tuples into mutable dicts/lists
    bank = dict(state.bank)
    player_gems = dict(player.gems)
    visible_cards = list(state.visible_cards)

    gem = getattr(self, 'gem')
    count = getattr(self, 'count', 2)
    if gem == Gem.GOLD:
      raise ValueError("Cannot take two gold tokens")
    if bank.get(gem, 0) < count:
      raise ValueError(f"Not enough {gem} in bank to take {count}")
    bank[gem] = bank.get(gem, 0) - count
    player_gems[gem] = player_gems.get(gem, 0) + count

    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems_in=player_gems, score=player.score,
                             reserved_cards_in=player.reserved_cards,
                             purchased_cards_in=player.purchased_cards)
    players = _replace_tuple(state.players, player.seat_id, new_player)

    return GameState(players=players, bank_in=bank,
                     visible_cards_in=visible_cards, turn=state.turn,
                     last_action=self)

  def _check(self, player: PlayerState, state: GameState) -> bool:
    if self.gem == Gem.GOLD:
      return False
    if state.bank.get(self.gem) < COIN_MIN_COUNT_TAKE2_IN_DECK:
      return False
    return True

@dataclass(frozen=True)
class BuyCardAction(Action):
  card: Card
  payment: GemList = field(default_factory=GemList)

  @classmethod
  def create(cls, card: Card, payment: Optional[Mapping[Gem, int]] = None) -> 'BuyCardAction':
    pay = GemList(dict(payment) if payment is not None else {})
    return cls(type=ActionType.BUY_CARD, card=card, payment=pay)

  def __str__(self) -> str:
    pay_str = ''.join(f"{g.short_str()}{n}" for g, n in self.payment)
    return f"Action.Buy({self.card.id}, {pay_str})"

  def _apply(self, player: PlayerState, state: GameState) -> GameState:
    # Mutable working copies
    bank = dict(state.bank)
    player_gems = dict(player.gems)
    visible_cards = list(state.visible_cards)

    card = self.card
    payment = dict(getattr(self, 'payment', ()))
    # locate card either in visible_cards or in player's reserved_cards
    found = None
    from_reserved = False
    reserved_list: list = []
    for i, c in enumerate(visible_cards):
      if c.id == card.id:
        found = visible_cards.pop(i)
        from_reserved = False
        break
    if found is None:
      # check reserved
      for i, c in enumerate(player.reserved_cards):
        if c.id == card.id:
          # remove from player's reserved list
          reserved_list = list(player.reserved_cards)
          found = reserved_list.pop(i)
          from_reserved = True
          break
    if found is None:
      raise ValueError("Card to buy not found")

    # apply payment: deduct from player_gems and add to bank
    for g, amt in payment.items():
      if player_gems.get(g, 0) < amt:
        raise ValueError(f"Player does not have enough {g} to pay")
      player_gems[g] = player_gems.get(g, 0) - amt
      bank[g] = bank.get(g, 0) + amt

    # update player's purchased cards and score
    new_purchased = tuple(player.purchased_cards) + (found,)
    new_score = player.score + getattr(found, 'points', 0)
    # if bought from reserved, remove it from reserved_cards; otherwise keep as-is
    if from_reserved:
      new_reserved = tuple(reserved_list)
    else:
      new_reserved = tuple(player.reserved_cards)

    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems_in=player_gems, score=new_score,
                             reserved_cards_in=new_reserved,
                             purchased_cards_in=new_purchased)
    players = _replace_tuple(state.players, player.seat_id, new_player)

    return GameState(players=players, bank_in=bank,
                     visible_cards_in=visible_cards, turn=state.turn,
                     last_action=self)

  def _check(self, player: PlayerState, state: GameState) -> bool:
    card = self.card
    # Ensure card is either among visible or reserved
    if state.visible_cards.get(card.id) is None and player.reserved_cards.get(card.id) is None:
      return False
    return player.check_afford(card, dict(self.payment))


@dataclass(frozen=True)
class ReserveCardAction(Action):
  card: Card
  take_gold: bool = True

  @classmethod
  def create(cls, card: Card, take_gold: bool = True) -> 'ReserveCardAction':
    return cls(type=ActionType.RESERVE_CARD, card=card, take_gold=bool(take_gold))

  def __str__(self) -> str:
    if self.take_gold:
      return f"Action.Reserve({self.card.id}, D)"
    return f"Action.Reserve({self.card.id})"

  def _apply(self, player: PlayerState, state: GameState) -> GameState:
    # Mutable working copies
    bank = dict(state.bank)
    player_gems = dict(player.gems)
    visible_cards = list(state.visible_cards)

    take_gold = getattr(self, 'take_gold', True)
    # find card in visible_cards
    found = None
    for i, c in enumerate(visible_cards):
      if c.id == self.card.id:
        found = visible_cards.pop(i)
        break
    if found is None:
      raise ValueError("Card to reserve not found in visible cards")
    # give gold if requested and available
    if take_gold and bank.get(Gem.GOLD, 0) > 0:
      bank[Gem.GOLD] = bank.get(Gem.GOLD, 0) - 1
      player_gems[Gem.GOLD] = player_gems.get(Gem.GOLD, 0) + 1
    # create new player with reserved card added
    new_reserved = tuple(player.reserved_cards) + (found,)
    if len(new_reserved) > 3:
      raise ValueError("Cannot reserve more than 3 cards")
    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems_in=player_gems, score=player.score,
                             reserved_cards_in=new_reserved,
                             purchased_cards_in=player.purchased_cards)
    players = _replace_tuple(state.players, player.seat_id, new_player)

    return GameState(players=players, bank_in=bank,
                     visible_cards_in=visible_cards, turn=state.turn,
                     last_action=self)

  def _check(self, player: PlayerState, state: GameState) -> bool:
    if not player.can_reserve():
      return False
    if state.visible_cards.get(self.card.id) is None:
      return False
    if self.take_gold and state.bank.get(Gem.GOLD) <= 0:
      return False
    return True

@dataclass(frozen=True)
class NoopAction(Action):
  """A no-op/fallback action that advances the turn without mutating state."""

  @classmethod
  def create(cls) -> 'NoopAction':
    return cls(type=ActionType.NOOP)

  def __str__(self) -> str:  # pragma: no cover - trivial
    return "Action.Noop()"

  def _apply(self, player: PlayerState, state: GameState) -> GameState:
    # simply return a new GameState with last_action set (do not modify turn)
    return GameState(players=state.players, bank=state.bank,
                     visible_cards=state.visible_cards, turn=state.turn,
                     last_action=self)

  def _check(self, player: PlayerState, state: GameState) -> bool:
    return True
