from dataclasses import dataclass, field
from collections.abc import Mapping
from abc import ABC, abstractmethod

from .consts import COIN_MAX_COUNT_PER_PLAYER, COIN_MIN_COUNT_TAKE2_IN_DECK
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
  def take3(cls, *gems: Gem, ret_map: Optional[Mapping[Gem, int]] = None) -> 'Take3Action':
    return Take3Action.create(*gems, ret_map=ret_map)

  @classmethod
  def take2(cls, gem: Gem, count: int = 2, ret_map: Optional[Mapping[Gem, int]] = None) -> 'Take2Action':
    return Take2Action.create(gem, count, ret_map=ret_map)

  @classmethod
  def buy(cls, card: Card, payment: Mapping[Gem, int] | None = None) -> 'BuyCardAction':
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
  gems: tuple[Gem, ...] = field(default_factory=tuple)
  # optional returned gems to satisfy max-10 hand size after taking
  ret: GemList | None = None

  @classmethod
  def create(cls, *gems: Gem, ret_map: Mapping[Gem, int] | None = None) -> 'Take3Action':
    ret = GemList(ret_map) if ret_map is not None else None
    return cls(type=ActionType.TAKE_3_DIFFERENT, gems=tuple(gems), ret=ret)

  def __str__(self) -> str:
    gem_str = ''.join(g.color_circle() for g in self.gems)
    if self.ret:
      return f"Action.Take3({gem_str}-{self.ret})"
    return f"Action.Take3({gem_str})"

  def _apply(self, player: PlayerState, state: GameState) -> GameState:
    # Mutable working copies: convert GemList/tuples into mutable dicts/lists
    bank = dict(state.bank)
    player_gems = dict(player.gems)
    visible_cards = list(state.visible_cards)

    # remove one of each requested gem from bank and add to player's gems
    for g in self.gems:
      bank[g] = bank.get(g, 0) - 1
      player_gems[g] = player_gems.get(g, 0) + 1

    # apply returns (if any): validate player has those gems and move back to bank
    if self.ret:
      for g, amt in self.ret:
        player_gems[g] = player_gems.get(g, 0) - amt
        bank[g] = bank.get(g, 0) + amt

    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems_in=player_gems, score=player.score,
                             reserved_cards_in=player.reserved_cards,
                             purchased_cards_in=player.purchased_cards)
    players = _replace_tuple(state.players, player.seat_id, new_player)

    return GameState(players=players, bank_in=bank,
                     visible_cards_in=visible_cards, turn=state.turn,
                     last_action=self)

  def _check(self, player: PlayerState, state: GameState) -> bool:
    # Ensure each gem to take is available in bank
    bank = {g: amt for g, amt in state.bank}
    for g in self.gems:
      if bank.get(g, 0) <= 0:
        return False

    # If returns provided, ensure player would have enough tokens after taking to return
    # Compute player's gems after taking
    player_gems = {g: n for g, n in player.gems}
    if self.ret:
      for g, amt in self.ret:
        if player_gems.get(g, 0) < amt:
          return False

    # check final total does not exceed 10
    total_after = sum(player_gems.values()) + len(self.gems) - \
        (sum(n for _, n in self.ret) if self.ret else 0)
    if total_after > COIN_MAX_COUNT_PER_PLAYER:
      return False
    return True

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState) -> list["Take3Action"]:
    # Enumerate legal take3 actions: either any combination of 3 distinct non-gold gems
    # with at least 1 in bank, or (if <3 available) a single action taking all remaining
    # distinct gems (permissive fallback matching previous logic).
    bank = {g: amt for g, amt in state.bank}
    available = [g for g, amt in bank.items() if amt > 0 and g != Gem.GOLD]
    total = sum(n for _, n in player.gems)
    actions: list[Take3Action] = []
    if len(available) == 0:
      return actions  # no gems available to take

    from itertools import combinations

    max_take = min(len(available), 3)
    if total + max_take <= COIN_MAX_COUNT_PER_PLAYER:
      # can take max_take without exceeding 10, so enumerate all combos of that size
      combos = combinations(available, max_take)
      for combo in combos:
        actions.append(cls.create(*combo))
    else:
      max_return = total + max_take - COIN_MAX_COUNT_PER_PLAYER
      for return_num in range(0, max_return + 1):
        take_num = COIN_MAX_COUNT_PER_PLAYER + return_num - total
        combos = combinations(available, take_num)
        for combo in combos:
          ret_available = [[g] * amt for g, amt in player.gems if g not in combo and g != Gem.GOLD]
          ret_available = [g for sublist in ret_available for g in sublist]
          return_combos = set(combinations(ret_available, return_num))
          for ret in return_combos:
            ret_map: Mapping[Gem, int] = {}
            for g in ret:
              ret_map[g] = ret_map.get(g, 0) + 1
            actions.append(cls.create(*combo, ret_map=ret_map))
    return actions


@dataclass(frozen=True)
class Take2Action(Action):
  gem: Gem
  count: int = 2
  # optional returned gems to satisfy max tokens per player after taking
  ret: GemList | None = None

  @classmethod
  def create(cls, gem: Gem, count: int = 2, ret_map: Mapping[Gem, int] | None = None) -> 'Take2Action':
    ret = GemList(ret_map) if ret_map is not None else None
    return cls(type=ActionType.TAKE_2_SAME, gem=gem, count=count, ret=ret)

  def __str__(self) -> str:
    base = f"Action.Take2({self.count}{self.gem.color_circle()})"
    if self.ret:
      return base[:-1] + f"-{self.ret})"
    return base

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

    # apply returns if provided
    if self.ret:
      for g, amt in self.ret:
        player_gems[g] = player_gems.get(g, 0) - amt
        bank[g] = bank.get(g, 0) + amt

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

    player_gems = player.gems.to_dict()

    if self.ret:
      for g, amt in self.ret:
        if player_gems.get(g, 0) < amt:
          return False

    total_after = sum(player_gems.values()) + self.count - (sum(n for _, n in self.ret) if self.ret else 0)
    if total_after > COIN_MAX_COUNT_PER_PLAYER:
      return False
    return True

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState) -> list["Take2Action"]:
    # Any non-gold gem with >= COIN_MIN_COUNT_TAKE2_IN_DECK (typically 4) is legal to take 2 of.
    actions: list[Take2Action] = []
    bank = {g: amt for g, amt in state.bank}
    total = sum(n for _, n in player.gems)
    for g, amt in bank.items():
      if g == Gem.GOLD:
        continue
      if amt < COIN_MIN_COUNT_TAKE2_IN_DECK:
        continue

      # compute if taking 2 would exceed the max
      need_return = max(0, total + 2 - COIN_MAX_COUNT_PER_PLAYER)
      if need_return == 0:
        actions.append(cls.create(g, 2))
        continue

      # enumerate return combinations from player's holdings excluding the gem being taken and gold
      ret_available = [[gg] * cnt for gg, cnt in player.gems if gg != g and gg != Gem.GOLD]
      ret_available = [x for sub in ret_available for x in sub]
      from itertools import combinations
      return_combos = set(combinations(ret_available, need_return))
      for ret in return_combos:
        ret_map: Mapping[Gem, int] = {}
        for gg in ret:
          ret_map[gg] = ret_map.get(gg, 0) + 1
        actions.append(cls.create(g, 2, ret_map=ret_map))

    return actions


@dataclass(frozen=True)
class BuyCardAction(Action):
  card: Card
  payment: GemList = field(default_factory=GemList)

  @classmethod
  def create(cls, card: Card, payment: Mapping[Gem, int] | None = None) -> 'BuyCardAction':
    pay = GemList(dict(payment) if payment is not None else {})
    return cls(type=ActionType.BUY_CARD, card=card, payment=pay)

  def __str__(self) -> str:
    return f"Action.Buy(<{self.card.id}>, {self.payment})"

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

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState) -> list["BuyCardAction"]:
    # For each visible or reserved card, enumerate all exact payment dicts the player can afford.
    actions: list[BuyCardAction] = []
    for card in state.visible_cards + player.reserved_cards:
      payments = player.can_afford(card)
      for payment in payments:
        actions.append(cls.create(card, payment=payment))
    return actions


@dataclass(frozen=True)
class ReserveCardAction(Action):
  card: Card
  take_gold: bool = True

  @classmethod
  def create(cls, card: Card, take_gold: bool = True) -> 'ReserveCardAction':
    return cls(type=ActionType.RESERVE_CARD, card=card, take_gold=bool(take_gold))

  def __str__(self) -> str:
    if self.take_gold:
      return f"Action.Reserve(<{self.card.id}>, {Gem.GOLD.color_circle()})"
    return f"Action.Reserve(<{self.card.id}>)"

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

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState) -> list["ReserveCardAction"]:
    actions: list[ReserveCardAction] = []
    if not player.can_reserve():
      return actions
    gold_in_bank = state.bank.get(Gem.GOLD)
    take_gold = gold_in_bank > 0
    for card in state.visible_cards:
      # Card must be visible to reserve; include gold token if available
      actions.append(cls.create(card, take_gold=take_gold))
    return actions


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

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState) -> list["NoopAction"]:
    # Always legal; only used as fallback if no other actions exist.
    return [cls.create()]
