from dataclasses import dataclass, field
from collections.abc import Mapping
from abc import ABC, abstractmethod

from .consts import GameConfig
from .typings import Gem, ActionType, GemList, Card, CardIdx
from typing import cast
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
  def take3(cls, *gems: Gem, ret_map: Mapping[Gem, int] | None = None) -> 'Take3Action':
    return Take3Action.create(*gems, ret_map=ret_map)

  @classmethod
  def take2(cls, gem: Gem, count: int = 2, ret_map: Mapping[Gem, int] | None = None) -> 'Take2Action':
    return Take2Action.create(gem, count, ret_map=ret_map)

  @classmethod
  def buy(cls, card: Card | None, payment: Mapping[Gem, int] | None = None, visible_idx: int | None = None, reserve_idx: int | None = None) -> 'BuyCardAction':
    # maintain friendly legacy signature: build CardIdx from provided indices
    idx = None
    if visible_idx is not None or reserve_idx is not None:
      idx = CardIdx(visible_idx=visible_idx, reserve_idx=reserve_idx)
    if idx is None:
      raise ValueError("Must provide at least one of visible_idx or reserve_idx to identify card to buy")
    return BuyCardAction.create(idx, card, payment=payment)

  @classmethod
  def buy_gold(cls, card: Card | None, gold_payment: Mapping[Gem, int] | None = None, visible_idx: int | None = None, reserve_idx: int | None = None) -> 'BuyCardActionGold':
    # maintain friendly legacy signature: build CardIdx from provided indices
    idx = None
    if visible_idx is not None or reserve_idx is not None:
      idx = CardIdx(visible_idx=visible_idx, reserve_idx=reserve_idx)
    if idx is None:
      raise ValueError("Must provide at least one of visible_idx or reserve_idx to identify card to buy")
    return BuyCardActionGold.create(idx, card, payment=gold_payment)

  @classmethod
  def reserve(cls, card: Card | None, take_gold: bool = True, visible_idx: int | None = None) -> 'ReserveCardAction':
    idx = CardIdx(visible_idx=visible_idx) if visible_idx is not None else None
    if idx is None:
      raise ValueError("Must provide visible_idx to identify card to reserve")
    return ReserveCardAction.create(idx, card, take_gold=take_gold)

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
    config = state.config

    if not self._check(player, state, config):
      raise ValueError(f"Action {self} cannot be applied for player {player} in state {state}")

    return self._apply(player, state, config)

  # Serialization helpers
  def serialize(self) -> dict:
    """Return a JSON-serializable dict representation of this Action.

    This is a thin alias for `to_dict` implemented on subclasses.
    """
    return self.to_dict()

  @classmethod
  def deserialize(cls, d: dict) -> 'Action':
    """Reconstruct an Action object from a dict produced by `serialize`.

    The dict must contain a 'type' key whose value is one of the
    ActionType enum values.
    """
    atype = ActionType(d.get('type'))
    if atype == ActionType.TAKE_3_DIFFERENT:
      return Take3Action.from_dict(d)
    if atype == ActionType.TAKE_2_SAME:
      return Take2Action.from_dict(d)
    if atype == ActionType.BUY_CARD:
      return BuyCardAction.from_dict(d)
    if atype == ActionType.RESERVE_CARD:
      return ReserveCardAction.from_dict(d)
    if atype == ActionType.NOOP:
      return NoopAction.from_dict(d)
    raise ValueError(f"Unknown action type for deserialization: {d.get('type')}")

  @abstractmethod
  def to_dict(self) -> dict:
    """Return a JSON-serializable dict representation of this Action."""

  @classmethod
  def from_dict(cls, d: dict) -> 'Action':
    """Reconstruct an Action from a dict. Subclasses should override."""
    raise NotImplementedError()

  @abstractmethod
  def _apply(self, player: PlayerState, state: GameState, config: GameConfig) -> GameState:
    """Apply this action for the current-turn player and return a new GameState.

    Concrete action classes must implement this. Implementations must not
    mutate the provided `state` and must return a new GameState with
    `last_action` set to this action. Implementations MUST NOT modify
    the `turn` field; turn advancement is the responsibility of the
    caller/engine.
    """

  def _check(self, player: PlayerState, state: GameState, config: GameConfig) -> bool:
    """Backward-compatible check wrapper: combine stateless and stateful checks.

    New split API:
    - _check_without_state(config) -> bool: checks that only depend on config
      (or other global rules) and do not require player/state.
    - _check_with_state(player, state, config) -> bool: checks that require
      access to the current player and full game state.

    For now, we keep this helper to preserve existing behaviour: both parts
    must return True for the action to be considered valid.
    """
    return self._check_without_state(config) and self._check_with_state(player, state, config)

  def _check_without_state(self, config: GameConfig) -> bool:
    """Return True if this action is allowed based only on global config/state.

    Subclasses may override this if there are checks that don't require the
    current player or full GameState. Default implementation allows the action.
    """
    return True

  @abstractmethod
  def _check_with_state(self, player: PlayerState, state: GameState, config: GameConfig) -> bool:
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

  def to_dict(self) -> dict:
    return {
        'type': self.type.value,
        'gems': [g.value for g in self.gems],
        'ret': [(g.value, n) for g, n in self.ret] if self.ret else None,
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'Take3Action':
    gems = tuple(Gem(g) for g in d.get('gems', ()))
    ret_raw = d.get('ret')
    ret = None
    if ret_raw:
      ret = {Gem(g): n for g, n in ret_raw}
    return cls.create(*gems, ret_map=ret)

  def _apply(self, player: PlayerState, state: GameState, config: GameConfig) -> GameState:
    # Mutable working copies: convert GemList/tuples into mutable dicts/lists
    bank = dict(state.bank)
    player_gems = dict(player.gems)

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

    return GameState(config=state.config, players=players, bank_in=bank,
                     visible_cards=state.visible_cards, turn=state.turn,
                     last_action=self)

  def _check_with_state(self, player: PlayerState, state: GameState, config: GameConfig) -> bool:
    # Ensure each gem to take is available in bank
    for g in self.gems:
      if state.bank.get(g) <= 0:
        return False

    # If returns provided, ensure player would have enough tokens after taking to return
    # Compute player's gems after taking
    if self.ret:
      for g, amt in self.ret:
        if player.gems.get(g) < amt:
          return False

    # check final total does not exceed 10
    total_after = player.gems.count() + len(self.gems) - \
        (self.ret.count() if self.ret else 0)
    if total_after > config.coin_max_count_per_player:
      return False
    return True

  def _check_without_state(self, config: GameConfig) -> bool:
    # stateless validation: ensure action parameters are sane before checking state
    # must not request more than 3 gems
    if len(self.gems) > 3:
      return False
    # gems must be distinct
    gems_set = set(self.gems)
    if len(gems_set) != len(self.gems):
      return False
    # cannot return gems that are also being taken
    if self.ret and any(g in gems_set for g, _ in self.ret):
      return False
    # none of the gems may be gold (gold cannot be taken as part of take-3)
    if any(g == Gem.GOLD for g in self.gems):
      return False
    return True

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState, config: GameConfig) -> list["Take3Action"]:
    # Enumerate legal take3 actions: either any combination of 3 distinct non-gold gems
    # with at least 1 in bank, or (if <3 available) a single action taking all remaining
    # distinct gems (permissive fallback matching previous logic).
    bank = {g: amt for g, amt in state.bank}
    available = [g for g, amt in bank.items() if amt > 0 and g != Gem.GOLD]
    total = player.gems.count()
    actions: list[Take3Action] = []
    if len(available) == 0:
      return actions  # no gems available to take

    from itertools import combinations

    max_take = min(len(available), 3)
    if total + max_take <= config.coin_max_count_per_player:
      # can take max_take without exceeding 10, so enumerate all combos of that size
      combos = combinations(available, max_take)
      for combo in combos:
        actions.append(cls.create(*combo))
    else:
      max_return = total + max_take - config.coin_max_count_per_player
      for return_num in range(0, max_return + 1):
        take_num = config.coin_max_count_per_player + return_num - total
        if take_num == 0:
          continue
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

  def to_dict(self) -> dict:
    return {
        'type': self.type.value,
        'gem': self.gem.value,
        'count': self.count,
        'ret': [(g.value, n) for g, n in self.ret] if self.ret else None,
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'Take2Action':
    gem = Gem(d.get('gem'))
    count = int(d.get('count', 2))
    ret_raw = d.get('ret')
    ret = None
    if ret_raw:
      ret = {Gem(g): n for g, n in ret_raw}
    return cls.create(gem, count, ret_map=ret)

  def _apply(self, player: PlayerState, state: GameState, config: GameConfig) -> GameState:
    # Mutable working copies: convert GemList/tuples into mutable dicts/lists
    bank = dict(state.bank)
    player_gems = dict(player.gems)

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

    return GameState(config=state.config, players=players, bank_in=bank,
                     visible_cards=state.visible_cards, turn=state.turn,
                     last_action=self)

  def _check_with_state(self, player: PlayerState, state: GameState, config: GameConfig) -> bool:
    # stateful checks that require bank/player info
    if state.bank.get(self.gem) < config.coin_min_count_take2_in_deck:
      return False

    player_gems = player.gems.to_dict()

    if self.ret:
      for g, amt in self.ret:
        if player_gems.get(g, 0) < amt:
          return False

    total_after = sum(player_gems.values()) + self.count - (self.ret.count() if self.ret else 0)
    if total_after > config.coin_max_count_per_player:
      return False
    return True

  def _check_without_state(self, config: GameConfig) -> bool:
    # Can't take two gold tokens
    if self.gem == Gem.GOLD:
      return False
    # must take exactly 2
    if self.count != 2:
      return False
    # cannot return gems that are also being taken
    if any(g == self.gem for g, _ in self.ret or ()):
      return False
    return True

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState, config: GameConfig) -> list["Take2Action"]:
    # Any non-gold gem with >= COIN_MIN_COUNT_TAKE2_IN_DECK (typically 4) is legal to take 2 of.
    actions: list[Take2Action] = []
    bank = {g: amt for g, amt in state.bank}
    total = player.gems.count()
    for g, amt in bank.items():
      if g == Gem.GOLD:
        continue
      if amt < config.coin_min_count_take2_in_deck:
        continue

      # compute if taking 2 would exceed the max
      need_return = max(0, total + 2 - config.coin_max_count_per_player)
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
  idx: CardIdx | None
  card: Card | None
  payment: GemList = field(default_factory=GemList)

  def __post_init__(self) -> None:
    if self.idx is None and self.card is None:
      raise ValueError("BuyCardAction requires at least one of idx or card to identify the card to buy")

  @classmethod
  def create(cls, card_idx: CardIdx | None, card: Card | None = None, payment: Mapping[Gem, int] | None = None) -> 'BuyCardAction':
    pay = GemList(dict(payment) if payment is not None else {})
    return cls(type=ActionType.BUY_CARD, idx=card_idx, card=card, payment=pay)

  def __str__(self) -> str:
    cid = self.card.id if self.card is not None else None
    if self.idx is None:
      return f"Action.Buy(<{cid}>, {self.payment})"
    return f"Action.Buy({self.idx.to_str(cid)}, {self.payment})"

  def to_dict(self) -> dict:
    return {
        'type': self.type.value,
        'card': self.card.to_dict() if self.card is not None else None,
        'payment': [(g.value, n) for g, n in self.payment],
        'idx': {
            'visible_idx': self.idx.visible_idx if self.idx is not None else None,
            'reserve_idx': self.idx.reserve_idx if self.idx is not None else None,
            'deck_head_level': self.idx.deck_head_level if self.idx is not None else None,
        } if self.idx is not None else None,
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'BuyCardAction':
    card_d = d.get('card')
    card = Card.from_dict(card_d) if card_d is not None else None
    pay_raw = d.get('payment', [])
    payment = {Gem(g): n for g, n in pay_raw}
    idx_raw = d.get('idx') or {}
    visible_idx = idx_raw.get('visible_idx')
    reserve_idx = idx_raw.get('reserve_idx')
    deck_head_level = idx_raw.get('deck_head_level')
    idx = CardIdx(visible_idx=visible_idx, reserve_idx=reserve_idx, deck_head_level=deck_head_level) if idx_raw else None
    return cls.create(idx, card, payment=payment)

  def _apply(self, player: PlayerState, state: GameState, config: GameConfig) -> GameState:
    # Mutable working copies
    bank = dict(state.bank)
    player_gems = dict(player.gems)
    visible_cards = list(state.visible_cards)

    # locate card according to idx
    if self.idx is None:
      raise ValueError("BuyCardAction requires an idx to locate the card")
    found = None
    from_reserved = False
    reserved_list: list[Card] = []
    if self.idx.visible_idx is not None:
      vi = int(self.idx.visible_idx)
      if vi < 0 or vi >= len(visible_cards):
        raise ValueError("visible_idx out of range")
      found = visible_cards.pop(vi)
      if self.card is not None and found.id != self.card.id:
        raise ValueError("Card at visible_idx does not match provided card")
      from_reserved = False
    elif self.idx.reserve_idx is not None:
      ri = int(self.idx.reserve_idx)
      reserved_list = list(player.reserved_cards)
      if ri < 0 or ri >= len(reserved_list):
        raise ValueError("reserve_idx out of range")
      found = reserved_list.pop(ri)
      if self.card is not None and found.id != self.card.id:
        raise ValueError("Card at reserve_idx does not match provided card")
      from_reserved = True
    else:
      # deck_head_level not supported for apply (would require drawing)
      raise ValueError("deck_head_level idx is not supported for BuyCardAction.apply")
    payment = self._get_payment(found)

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

    return GameState(config=state.config, players=players, bank_in=bank,
                     visible_cards_in=visible_cards, turn=state.turn,
                     last_action=self)

  def _get_payment(self, card: Card):
    return dict(self.payment)

  def _check_afford(self, player: PlayerState, card: Card) -> bool:
    return player.check_afford(card, self._get_payment(card))

  def _check_with_state(self, player: PlayerState, state: GameState, config: GameConfig) -> bool:
    # require idx to be set and match
    if self.idx is None:
      return False
    if self.idx.visible_idx is not None:
      vi = self.idx.visible_idx
      try:
        c = state.visible_cards[vi]
      except Exception:
        return False
      if self.card is not None and c.id != self.card.id:
        return False
      return self._check_afford(player, c)
    if self.idx.reserve_idx is not None:
      ri = int(self.idx.reserve_idx)
      try:
        c = player.reserved_cards[ri]
      except Exception:
        return False
      if self.card is not None and c.id != self.card.id:
        return False
      return self._check_afford(player, c)
    # deck_head_level can't be checked here
    return False

  def _check_without_state(self, config: GameConfig) -> bool:
    if self.idx is not None:
      if self.idx.deck_head_level is not None:
        # deck_head_level idx is not supported for BuyCardAction
        return False
    elif self.card is None:
      # must provide at least one of idx or card
      return False
    return True

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState, config: GameConfig) -> list["BuyCardAction"]:
    # For each visible or reserved card, enumerate all exact payment dicts the player can afford.
    actions: list[BuyCardAction] = []
    # visible cards with indices
    for i, card in enumerate(state.visible_cards):
      payments = player.can_afford(card)
      for payment in payments:
        idx = CardIdx(visible_idx=i)
        actions.append(cls.create(idx, card, payment=payment))
    # reserved cards with indices
    for i, card in enumerate(player.reserved_cards):
      payments = player.can_afford(card)
      for payment in payments:
        idx = CardIdx(reserve_idx=i)
        actions.append(cls.create(idx, card, payment=payment))
    return actions

@dataclass(frozen=True)
class BuyCardActionGold(BuyCardAction):
  gold_payment: GemList = field(default_factory=GemList)

  def __str__(self) -> str:
    cid = self.card.id if self.card is not None else None
    gold_count = self.gold_payment.count()
    card_str = f"<{cid}>" if self.idx is None else self.idx.to_str(cid)
    if gold_count:
      return f"Action.Buy({card_str}, {{{GemList({Gem.GOLD: gold_count})}={self.gold_payment}}})"
    return f"Action.Buy({card_str}, {{0{Gem.GOLD.color_circle()}}})"

  @classmethod
  def create(cls, card_idx: CardIdx | None, card: Card | None = None, payment: Mapping[Gem, int] | None = None) -> 'BuyCardActionGold':
    gp = GemList(dict(payment) if payment is not None else {})
    return cls(type=ActionType.BUY_CARD, idx=card_idx, card=card, gold_payment=gp)

  def to_dict(self) -> dict:
    d = super().to_dict()
    d['gold_payment'] = [(g.value, n) for g, n in self.gold_payment]
    del d['payment']
    return d

  @classmethod
  def from_dict(cls, d: dict) -> 'BuyCardActionGold':
    card_d = d.get('card')
    card = Card.from_dict(card_d) if card_d is not None else None
    gold_pay_raw = d.get('gold_payment', [])
    gold_payment = {Gem(g): n for g, n in gold_pay_raw}
    idx_raw = d.get('idx') or {}
    visible_idx = idx_raw.get('visible_idx')
    reserve_idx = idx_raw.get('reserve_idx')
    deck_head_level = idx_raw.get('deck_head_level')
    idx = CardIdx(visible_idx=visible_idx, reserve_idx=reserve_idx, deck_head_level=deck_head_level) if idx_raw else None
    return cls.create(idx, card, payment=gold_payment)

  def _get_payment(self, card: Card):
    payment = {Gem.GOLD: self.gold_payment.count()}
    for g, cost in card.cost:
      count = cost - self.gold_payment.get(g)
      if count > 0:
        payment[g] = count
    return payment

  def normalize(self, card: Card) -> 'BuyCardAction':
    payment = self._get_payment(card)
    return BuyCardAction.create(self.idx, self.card, payment=payment)

@dataclass(frozen=True)
class ReserveCardAction(Action):
  idx: CardIdx | None
  card: Card | None
  take_gold: bool = True
  # optional single gem to return if taking gold would push player over the limit
  ret: Gem | None = None

  def __post_init__(self) -> None:
    if self.idx is None and self.card is None:
      raise ValueError("ReserveCardAction requires at least one of idx or card to identify the card to reserve")

  @classmethod
  def create(cls, card_idx: CardIdx | None, card: Card | None = None, take_gold: bool = True, ret: Gem | None = None) -> 'ReserveCardAction':
    return cls(type=ActionType.RESERVE_CARD, idx=card_idx, card=card, take_gold=bool(take_gold), ret=ret)

  def __str__(self) -> str:
    cid = self.card.id if self.card is not None else None
    ext = ""
    if self.take_gold:
      ext = f"{Gem.GOLD.color_circle()}"
      if self.ret:
        ext = f"{ext}-{self.ret.color_circle()}"
      ext = f", {ext}"
    elif self.ret:
      ext = f", -{self.ret.color_circle()}"
    if self.idx is None:
      return f"Action.Reserve(<{cid}>{ext})"
    return f"Action.Reserve({self.idx.to_str(cid)}{ext})"

  def to_dict(self) -> dict:
    return {
        'type': self.type.value,
        'card': self.card.to_dict() if self.card is not None else None,
        'take_gold': bool(self.take_gold),
        'ret': self.ret.value if self.ret is not None else None,
        'idx': {
            'visible_idx': self.idx.visible_idx if self.idx is not None else None,
            'reserve_idx': self.idx.reserve_idx if self.idx is not None else None,
            'deck_head_level': self.idx.deck_head_level if self.idx is not None else None,
        } if self.idx is not None else None,
    }

  @classmethod
  def from_dict(cls, d: dict) -> 'ReserveCardAction':
    card_d = d.get('card')
    card = Card.from_dict(card_d) if card_d is not None else None
    take_gold = bool(d.get('take_gold', True))
    ret_raw = d.get('ret')
    ret = Gem(ret_raw) if ret_raw is not None else None
    idx_raw = d.get('idx') or {}
    visible_idx = idx_raw.get('visible_idx')
    reserve_idx = idx_raw.get('reserve_idx')
    deck_head_level = idx_raw.get('deck_head_level')
    idx = CardIdx(visible_idx=visible_idx, reserve_idx=reserve_idx, deck_head_level=deck_head_level) if idx_raw else None
    return cls.create(idx, card, take_gold=take_gold, ret=ret)

  def _apply(self, player: PlayerState, state: GameState, config: GameConfig) -> GameState:
    # Mutable working copies
    bank = dict(state.bank)
    player_gems = dict(player.gems)
    visible_cards = list(state.visible_cards)

    # locate card according to idx; default to searching visible_cards if idx not provided
    found = None
    # find by visible id or by scanning
    if self.idx is not None:
      if self.idx.visible_idx is not None:
        vi = self.idx.visible_idx
        found = visible_cards.pop(vi)
      else:
        # deck_head_level/reserve_idx not supported for reserve.apply
        raise ValueError("ReserveCardAction requires a visible_idx in apply")
    elif self.card is not None:
      # fallback: scan visible_cards for matching id
      for i, c in enumerate(visible_cards):
        if self.card is not None and c.id == self.card.id:
          found = visible_cards.pop(i)
          break
      if found is None:
        raise ValueError("Card to reserve not found in visible cards")
    else:
      # reserve by idx.reserve_idx not supported for reserve.apply
      raise ValueError("ReserveCardAction requires a visible_idx or card in apply")

    # give gold if requested and available
    if self.take_gold and bank.get(Gem.GOLD, 0) > 0:
      bank[Gem.GOLD] = bank.get(Gem.GOLD, 0) - 1
      player_gems[Gem.GOLD] = player_gems.get(Gem.GOLD, 0) + 1
    if self.ret:
      bank[self.ret] = bank.get(self.ret, 0) + 1
      player_gems[self.ret] = player_gems.get(self.ret, 0) - 1
    # create new player with reserved card added
    new_reserved = tuple(player.reserved_cards) + (found,)
    new_player = PlayerState(seat_id=player.seat_id, name=player.name,
                             gems_in=player_gems, score=player.score,
                             reserved_cards_in=new_reserved,
                             purchased_cards_in=player.purchased_cards)
    players = _replace_tuple(state.players, player.seat_id, new_player)

    return GameState(config=state.config, players=players, bank_in=bank,
                     visible_cards_in=visible_cards, turn=state.turn,
                     last_action=self)

  def _check_with_state(self, player: PlayerState, state: GameState, config: GameConfig) -> bool:
    if not player.can_reserve(config):
      return False
    # ensure card is visible
    if self.idx is not None:
      if self.idx.visible_idx is not None:
        try:
          c = state.visible_cards[self.idx.visible_idx]
        except Exception:
          return False
        if self.card is not None and c.id != self.card.id:
          return False
    elif self.card is not None:
      if state.visible_cards.find(self.card.id) is None:
        return False
    if self.take_gold and state.bank.get(Gem.GOLD) <= 0:
      return False

    if self.ret == Gem.GOLD:
      return False
    if self.ret:
      if player.gems.get(self.ret) <= 0:
        return False

      total = player.gems.count()
      if total != config.coin_max_count_per_player:
        return False

    return True

  def _check_without_state(self, config: GameConfig) -> bool:
    # could only return a gem if taking gold
    if self.ret is not None and not self.take_gold:
      print(self.ret, self.take_gold)
      assert False
      return False
    # self.idx must specify visible_idx (deck_head_level/reserve_idx not supported)
    if self.idx is not None:
      if self.idx.visible_idx is None and self.idx.deck_head_level is None:
        assert False
        return False
    elif self.card is None:
      assert False
      return False
    return True

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState, config: GameConfig) -> list["ReserveCardAction"]:
    actions: list[ReserveCardAction] = []
    if not player.can_reserve(config):
      return actions
    gold_in_bank = state.bank.get(Gem.GOLD)
    take_gold = gold_in_bank > 0
    total = player.gems.count()
    for i, card in enumerate(state.visible_cards):
      # Card must be visible to reserve; include gold token if available
      if take_gold and total + 1 > config.coin_max_count_per_player:
        # if taking gold would exceed, enumerate possible single-gem returns

        # enumerate distinct gems the player can return (exclude gold)
        for g, cnt in player.gems:
          if g == Gem.GOLD:
            continue
          if cnt <= 0:
            continue
          idx = CardIdx(visible_idx=i)
          actions.append(cls.create(idx, card, take_gold=True, ret=g))
      else:
        idx = CardIdx(visible_idx=i)
        actions.append(cls.create(idx, card, take_gold=take_gold))
    return actions


@dataclass(frozen=True)
class NoopAction(Action):
  """A no-op/fallback action that advances the turn without mutating state."""

  @classmethod
  def create(cls) -> 'NoopAction':
    return cls(type=ActionType.NOOP)

  def __str__(self) -> str:  # pragma: no cover - trivial
    return "Action.Noop()"

  def to_dict(self) -> dict:  # pragma: no cover - trivial
    return {'type': self.type.value}

  @classmethod
  def from_dict(cls, d: dict) -> 'NoopAction':  # pragma: no cover - trivial
    return cls.create()

  def _apply(self, player: PlayerState, state: GameState, config: GameConfig) -> GameState:
    # simply return a new GameState with last_action set (do not modify turn)
    return GameState(config=state.config, players=state.players, bank=state.bank,
                     visible_cards=state.visible_cards, turn=state.turn,
                     last_action=self)

  def _check_with_state(self, player: PlayerState, state: GameState, config: GameConfig) -> bool:
    return True

  def _check_without_state(self, config: GameConfig) -> bool:
    return True

  @classmethod
  def _get_legal_actions(cls, player: PlayerState, state: GameState, config: GameConfig) -> list["NoopAction"]:
    # Always legal; only used as fallback if no other actions exist.
    return [cls.create()]
