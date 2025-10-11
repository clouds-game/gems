from dataclasses import dataclass, field
from typing import Optional, Mapping, Tuple, Dict, Iterable
from abc import ABC, abstractmethod

from .typings import Gem, ActionType, _to_kv_tuple, GameState, PlayerState, Card


@dataclass(frozen=True)
class Action(ABC):
  """Minimal base Action used as a type tag for polymorphism.

  Concrete action types should subclass this and add strongly-typed
  fields. Keeping this minimal lets engine and agents switch on
  `type` safely.
  """
  type: ActionType

  # Backwards-compatible convenience constructors forwarding to the
  # concrete action dataclasses. These preserve prior call-site
  # ergonomics while returning the new concrete types.
  @classmethod
  def take3(cls, *gems: Gem) -> 'Take3Action':
    return Take3Action.create(*gems)

  @classmethod
  def take2(cls, gem: Gem, count: int = 2) -> 'Take2Action':
    return Take2Action.create(gem, count)

  @classmethod
  def buy(cls, card_id: str, payment: Optional[Mapping[Gem, int]] = None) -> 'BuyCardAction':
    return BuyCardAction.create(card_id, payment=payment)

  @classmethod
  def reserve(cls, card_id: str, take_gold: bool = True) -> 'ReserveCardAction':
    return ReserveCardAction.create(card_id, take_gold=take_gold)

  @abstractmethod
  def apply(self, state: GameState) -> GameState:
    """Apply this action for the current-turn player and return a new GameState.

    Concrete action classes must implement this. Implementations must not
    mutate the provided `state` and must return a new GameState with
    `turn` incremented and `last_action` set to this action.
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

  def apply(self, state: GameState) -> GameState:
    num_players = len(state.players)
    if num_players == 0:
      raise ValueError("state must contain at least one player")

    seat = state.turn % num_players
    player = state.players[seat]

    # Mutable working copies
    bank = _kv_tuple_to_dict(state.bank)
    player_gems = _kv_tuple_to_dict(player.gems)
    visible_cards = list(state.visible_cards)
    players = list(state.players)

    # remove one of each requested gem from bank and add to player's gems
    for g in getattr(self, 'gems', ()):  # type: ignore[attr-defined]
      if bank.get(g, 0) <= 0:
        raise ValueError(f"Not enough {g} in bank to take")
      bank[g] = bank.get(g, 0) - 1
      player_gems[g] = player_gems.get(g, 0) + 1

    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems=_dict_to_kv_tuple(player_gems), score=player.score,
                             reserved_cards=tuple(player.reserved_cards),
                             purchased_cards_in=tuple(player.purchased_cards))
    players[seat] = new_player

    new_bank = _dict_to_kv_tuple(bank)
    new_visible = tuple(visible_cards)
    return GameState(players=tuple(players), bank=new_bank,
                     visible_cards=new_visible, turn=state.turn + 1,
                     last_action=self)


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

  def apply(self, state: GameState) -> GameState:
    num_players = len(state.players)
    if num_players == 0:
      raise ValueError("state must contain at least one player")

    seat = state.turn % num_players
    player = state.players[seat]

    # Mutable working copies
    bank = _kv_tuple_to_dict(state.bank)
    player_gems = _kv_tuple_to_dict(player.gems)
    visible_cards = list(state.visible_cards)
    players = list(state.players)

    gem = getattr(self, 'gem')
    count = getattr(self, 'count', 2)
    if gem == Gem.GOLD:
      raise ValueError("Cannot take two gold tokens")
    if bank.get(gem, 0) < count:
      raise ValueError(f"Not enough {gem} in bank to take {count}")
    bank[gem] = bank.get(gem, 0) - count
    player_gems[gem] = player_gems.get(gem, 0) + count

    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems=_dict_to_kv_tuple(player_gems), score=player.score,
                             reserved_cards=tuple(player.reserved_cards),
                             purchased_cards_in=tuple(player.purchased_cards))
    players[seat] = new_player

    new_bank = _dict_to_kv_tuple(bank)
    new_visible = tuple(visible_cards)
    return GameState(players=tuple(players), bank=new_bank,
                     visible_cards=new_visible, turn=state.turn + 1,
                     last_action=self)


@dataclass(frozen=True)
class BuyCardAction(Action):
  card_id: str = ''
  payment: Tuple[Tuple[Gem, int], ...] = field(default_factory=tuple)

  @classmethod
  def create(cls, card_id: str, payment: Optional[Mapping[Gem, int]] = None) -> 'BuyCardAction':
    pay = _to_kv_tuple(dict(payment) if payment is not None else {})
    return cls(type=ActionType.BUY_CARD, card_id=card_id, payment=pay)

  def __str__(self) -> str:
    pay_str = ''.join(f"{g.short_str()}{n}" for g, n in self.payment)
    return f"Action.Buy({self.card_id}, {pay_str})"

  def apply(self, state: GameState) -> GameState:
    num_players = len(state.players)
    if num_players == 0:
      raise ValueError("state must contain at least one player")

    seat = state.turn % num_players
    player = state.players[seat]

    # Mutable working copies
    bank = _kv_tuple_to_dict(state.bank)
    player_gems = _kv_tuple_to_dict(player.gems)
    visible_cards = list(state.visible_cards)
    players = list(state.players)

    card_id = getattr(self, 'card_id')
    payment = dict(getattr(self, 'payment', ()))
    # locate card either in visible_cards or in player's reserved_cards
    found = None
    from_reserved = False
    reserved_list = []
    for i, c in enumerate(visible_cards):
      if getattr(c, 'id', None) == card_id:
        found = visible_cards.pop(i)
        from_reserved = False
        break
    if found is None:
      # check reserved
      for i, c in enumerate(player.reserved_cards):
        if getattr(c, 'id', None) == card_id:
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
                             gems=_dict_to_kv_tuple(player_gems), score=new_score,
                             reserved_cards=new_reserved,
                             purchased_cards_in=new_purchased)
    players[seat] = new_player

    new_bank = _dict_to_kv_tuple(bank)
    new_visible = tuple(visible_cards)
    return GameState(players=tuple(players), bank=new_bank,
                     visible_cards=new_visible, turn=state.turn + 1,
                     last_action=self)


@dataclass(frozen=True)
class ReserveCardAction(Action):
  card_id: str = ''
  take_gold: bool = True

  @classmethod
  def create(cls, card_id: str, take_gold: bool = True) -> 'ReserveCardAction':
    return cls(type=ActionType.RESERVE_CARD, card_id=card_id, take_gold=bool(take_gold))

  def __str__(self) -> str:
    if self.take_gold:
      return f"Action.Reserve({self.card_id}, D)"
    return f"Action.Reserve({self.card_id})"

  def apply(self, state: GameState) -> GameState:
    num_players = len(state.players)
    if num_players == 0:
      raise ValueError("state must contain at least one player")

    seat = state.turn % num_players
    player = state.players[seat]

    # Mutable working copies
    bank = _kv_tuple_to_dict(state.bank)
    player_gems = _kv_tuple_to_dict(player.gems)
    visible_cards = list(state.visible_cards)
    players = list(state.players)

    card_id = getattr(self, 'card_id')
    take_gold = getattr(self, 'take_gold', True)
    # find card in visible_cards
    found = None
    for i, c in enumerate(visible_cards):
      if getattr(c, 'id', None) == card_id:
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
    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems=_dict_to_kv_tuple(player_gems), score=player.score,
                             reserved_cards=new_reserved,
                             purchased_cards_in=tuple(player.purchased_cards))
    players[seat] = new_player

    new_bank = _dict_to_kv_tuple(bank)
    new_visible = tuple(visible_cards)
    return GameState(players=tuple(players), bank=new_bank,
                     visible_cards=new_visible, turn=state.turn + 1,
                     last_action=self)


def _kv_tuple_to_dict(t: Iterable[Tuple[Gem, int]]) -> Dict[Gem, int]:
  return {g: n for g, n in t}


def _dict_to_kv_tuple(d: Mapping[Gem, int]) -> Tuple[Tuple[Gem, int], ...]:
  return _to_kv_tuple(dict(d))


def apply_action(state: GameState, action: Action) -> GameState:
  """Apply `action` for the current-turn player and return a new GameState.

  The function does not mutate the provided `state`. It treats the player
  whose turn it is as `state.turn % len(state.players)`.

  Supported actions: TAKE_3_DIFFERENT, TAKE_2_SAME, RESERVE_CARD, BUY_CARD.
  Basic validation is performed (e.g. bank has enough tokens). More
  sophisticated rule checks are intentionally left to higher-level code.
  """
  return action.apply(state)
