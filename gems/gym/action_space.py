"""Structured ActionSpace for gymnasium-based agents.

This module contains the ActionSpace class which encodes and decodes
engine Action objects into a fixed dict/ndarray structure usable by
RL agents and gymnasium interfaces.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypeAlias, TypeVar, Sequence, TypedDict, cast, Generic

import numpy as np
from gymnasium import spaces

from ..actions import Action, BuyCardAction, BuyCardActionGold, ReserveCardAction, Take2Action, Take3Action
from ..typings import Gem, ActionType, CardIdx
from ..consts import GameConfig

from ._common import NDArray1D, Scalar
from .sampling import sample_exact, sample_single


# Generic scalar type for array elements; default to np.int8 for backward compat
T = TypeVar('T', bound=np.generic, default=np.int8)
U = TypeVar('U', bound=np.generic, default=np.uint16)

class Take3Dict(TypedDict, Generic[T]):
  gems_count: Scalar[T]
  ret_count: Scalar[T]
  gems: NDArray1D[T]
  ret: NDArray1D[T]

class Take2Dict(TypedDict, Generic[T]):
  gem: Scalar[T]
  count: Scalar[T]
  ret_count: Scalar[T]
  ret: NDArray1D[T]

class BuyDict(TypedDict, Generic[T, U]):
  # flattened card index (0..visible+reserved-1)
  card_idx: Scalar[U]
  payment_count: Scalar[T]
  payment: NDArray1D[T]

class ReserveDict(TypedDict, Generic[T, U]):
  card_idx: Scalar[U]
  take_gold: Scalar[T]
  ret_count: Scalar[T]
  ret: NDArray1D[T]

class ActionDict(TypedDict, Generic[T, U]):
  type: Scalar[T]
  take3: "Take3Dict[T]"
  take2: "Take2Dict[T]"
  buy: "BuyDict[T, U]"
  reserve: "ReserveDict[T, U]"


class ActionSpaceConfig(GameConfig):
  gem_list: list[Gem] = list(Gem)
  gem_idx: dict[Gem, int] = {g: i for i, g in enumerate(Gem)}

  def __init__(self, config: GameConfig):
    super().__init__(**asdict(config))

  @property
  def gold_idx(self) -> int:
    return self.gem_idx[Gem.GOLD]

  @property
  def max_card_index(self) -> int:
    return self.card_visible_total_count + self.card_max_count_reserved + self.card_level_count

  def flatten_card_idx(self, idx: CardIdx) -> int:
    """Flatten a CardIdx into an integer index using this config.

    Layout (flattened index):
    - 0..(reserve-1) => reserve_idx
    - reserve..(reserve+visible-1) => visible_idx (offset by reserve)
    - reserve+visible.. => deck_head_level (offset by reserve+visible), one entry per level

    More concretely this uses the config attributes:
    - reserve indices are 0..(card_max_count_reserved-1)
    - visible indices are offset by card_max_count_reserved
    - deck head levels are offset by card_max_count_reserved + card_visible_total_count

    Returns -1 when idx is None or no field is set.
    """
    if idx.reserve_idx is not None:
      return idx.reserve_idx
    if idx.visible_idx is not None:
      return self.card_max_count_reserved + idx.visible_idx
    if idx.deck_head_level is not None:
      level = idx.deck_head_level - 1
      return self.card_visible_total_count + self.card_max_count_reserved + level
    return -1

  def unflatten_card_idx(self, flat: int) -> CardIdx | None:
    """Inverse of flatten_card_idx: turn integer index back into CardIdx.

    Returns None only if `flat` is negative. Note: callers may pass 0
    which is a valid visible index (0) so treat 0 normally.
    """
    try:
      idx = int(flat)
    except Exception:
      return None
    if idx < 0:
      return None
    if idx < self.card_max_count_reserved:
      return CardIdx(reserve_idx=idx)
    if idx < self.card_visible_total_count + self.card_max_count_reserved:
      return CardIdx(visible_idx=idx - self.card_max_count_reserved)
    if idx < self.card_visible_total_count + self.card_max_count_reserved + self.card_level_count:
      level = (idx - self.card_visible_total_count - self.card_max_count_reserved) + 1
      return CardIdx(deck_head_level=level)
    return None

  def decode_gems_list(self, vec) -> dict[Gem, int] | None:
    """Convert a return-vector (iterable of ints) into a Gem->int dict or None.

    Assumes input is valid (iterable with self._gem_count entries). Returns None
    when no gems are returned (all zeros).
    """
    if vec is None:
      return None
    total = sum(int(x) for x in vec)
    if total == 0:
      return None
    return {self.gem_list[i]: int(vec[i]) for i in range(len(vec)) if int(vec[i]) > 0}


class Take3Space(spaces.Dict):
  def __init__(self, config: ActionSpaceConfig | GameConfig, *, seed = None, **spaces_kwargs):
    self.config = config if isinstance(config, ActionSpaceConfig) else ActionSpaceConfig(config)
    self._gems = spaces.Box(low=0, high=1, shape=(config.gem_count,), dtype=np.int8)
    self._ret = spaces.Box(low=0, high=config.coin_max_count_per_player, shape=(config.gem_count,), dtype=np.int8)
    super().__init__({
      'gems_count': spaces.Box(1, config.take3_count + 1),
      'ret_count': spaces.Box(0, config.take3_count + 1),
      'gems': self._gems,
      'ret': self._ret,
    }, seed, **spaces_kwargs)

  def _encode(self, data: "Take3Dict", action: Take3Action):
    take3 = data
    take3['gems'][...] = 0
    for gem in action.gems:
      take3['gems'][self.config.gem_idx[gem]] = 1
    take3['gems_count'][...] = len(action.gems)
    take3['ret'][...] = 0
    ret_count = 0
    for gem, amount in action.ret or ():
      ret_count += amount
      take3['ret'][self.config.gem_idx[gem]] = int(amount)
    take3['ret_count'][...] = ret_count

  def _decode(self, data: "Take3Dict") -> Take3Action:
    gems_vec = data['gems']
    gems = tuple(gem for gem, v in zip(self.config.gem_list, gems_vec) if int(v) != 0)
    ret_vec = data['ret']
    ret = self.config.decode_gems_list(ret_vec)
    return Take3Action.create(*gems, ret_map=ret)

  def _sample(self, mask: Take3Dict[np.bool] | None = None, probability: Take3Dict[np.floating] | None = None) -> Take3Dict:
    # build masks/weights and ensure GOLD is never selected as part of Take3
    gems_mask = None if mask is None else np.asarray(mask['gems'], dtype=bool).copy()
    gems_p = None if probability is None else probability['gems']

    # always exclude GOLD from candidate gems
    gold_idx = self.config.gold_idx
    if gems_mask is None:
      gems_mask = np.ones(self.config.gem_count, dtype=bool)
    gems_mask[gold_idx] = False

    ret_mask = None if mask is None else np.asarray(mask['ret'], dtype=bool).copy()
    ret_p = None if probability is None else probability['ret']

    # sample up to 3 distinct non-gold gems
    gems_sampled = sample_exact(self.config.gem_count, 3, dtype=np.int8, mask=gems_mask, p=gems_p, replacement=False, rng=self._gems._np_random)
    gems_count = int(gems_sampled.sum())

    # ensure returned gems do not overlap with taken gems by masking them out
    if ret_mask is None:
      ret_mask_final = np.ones(self.config.gem_count, dtype=bool)
    else:
      ret_mask_final = np.asarray(ret_mask, dtype=bool).copy()
    # zero-out indices for gems that were taken
    ret_mask_final = ret_mask_final & (gems_sampled == 0)

    ret_count = int(self.np_random.integers(0, gems_count + 1))
    ret_sampled = sample_exact(self.config.gem_count, int(ret_count), dtype=np.int8, mask=ret_mask_final, p=ret_p, replacement=True, rng=self._gems._np_random)

    return {
      'gems_count': np.array(gems_count, dtype=np.int8),
      'ret_count': np.array(ret_count, dtype=np.int8),
      'gems': gems_sampled,
      'ret': ret_sampled,
    }

  def sample(self, mask = None, probability = None) -> dict[str, Any]:
    return self._sample(mask=mask, probability=probability) # type: ignore[TypedDict]

class Take2Space(spaces.Dict):
  def __init__(self, config: ActionSpaceConfig | GameConfig, *, seed = None, **spaces_kwargs):
    self.config = config if isinstance(config, ActionSpaceConfig) else ActionSpaceConfig(config)
    super().__init__({
      'gem': spaces.Discrete(config.gem_count),
      'count': spaces.Discrete(3),
      'ret_count': spaces.Box(0, 3),
      'ret': spaces.Box(low=0, high=config.coin_max_count_per_player, shape=(config.gem_count,), dtype=np.int8),
    }, seed, **spaces_kwargs)

  def _encode(self, data: "Take2Dict", action: Take2Action):
    take2 = data
    take2['gem'][...] = self.config.gem_idx[action.gem]
    take2['count'][...] = int(action.count)
    take2['ret_count'][...] = action.ret.count() if action.ret is not None else 0
    take2['ret'][...] = 0
    for gem_ret, amount in action.ret or ():
      take2['ret'][self.config.gem_idx[gem_ret]] = int(amount)

  def _decode(self, data: "Take2Dict") -> Take2Action:
    gem_idx = data['gem']
    gem = self.config.gem_list[int(gem_idx)]
    count = int(data['count'])
    ret = self.config.decode_gems_list(data['ret'])
    return Take2Action.create(gem, count, ret_map=ret)

  def _sample(self, mask: Take2Dict[np.bool] | None = None, probability: Take2Dict[np.floating] | None = None) -> Take2Dict:
    # build masks/weights and ensure GOLD is never selected for Take2
    gem_mask = None if mask is None else np.asarray(mask['gem'], dtype=bool).copy()
    gem_p = None if probability is None else probability['gem']

    gold_idx = self.config.gold_idx
    if gem_mask is None:
      gem_mask = np.ones(self.config.gem_count, dtype=bool)
    gem_mask[gold_idx] = False

    # sample a single gem index (as a one-hot/count vector)
    gem_idx = sample_single(self.config.gem_count, dtype=np.int8, mask=gem_mask, p=gem_p, rng=self.np_random)

    # choose count: 2
    count = 2

    # ret mask: exclude the taken gem
    ret_mask = None if mask is None else np.asarray(mask['ret'], dtype=bool).copy()
    ret_p = None if probability is None else probability['ret']
    if ret_mask is None:
      ret_mask_final = np.ones(self.config.gem_count, dtype=bool)
    else:
      ret_mask_final = np.asarray(ret_mask, dtype=bool).copy()
    # cannot return the gem being taken
    ret_mask_final[gem_idx] = False

    ret_count = int(self.np_random.integers(0, count + 1))
    ret_sampled = sample_exact(self.config.gem_count, int(ret_count), dtype=np.int8, mask=ret_mask_final, p=ret_p, replacement=True, rng=self.np_random)

    return {
      'gem': np.array(gem_idx, dtype=np.int8),
      'count': np.array(count, dtype=np.int8),
      'ret_count': np.array(ret_count, dtype=np.int8),
      'ret': ret_sampled,
    }

  def sample(self, mask = None, probability = None) -> dict[str, Any]:
    return self._sample(mask=mask, probability=probability) # type: ignore[TypedDict]

class BuyCardSpace(spaces.Dict):
  def __init__(self, config: ActionSpaceConfig | GameConfig, *, seed = None, **spaces_kwargs):
    self.config = config if isinstance(config, ActionSpaceConfig) else ActionSpaceConfig(config)
    super().__init__({
      'card_idx': spaces.Discrete(config.card_visible_total_count + config.card_max_count_reserved + config.card_level_count),
      'payment': spaces.Box(low=0, high=255, shape=(config.gem_count,), dtype=np.int32),
    }, seed, **spaces_kwargs)

  def _encode(self, data: "BuyDict", action: BuyCardActionGold):
    buy = data
    buy['card_idx'][...] = 0
    if action.idx is not None:
      buy['card_idx'][...] = self.config.flatten_card_idx(action.idx)
    buy['payment_count'][...] = action.gold_payment.count()
    buy['payment'][...] = 0
    for gem, amount in action.gold_payment or ():
      buy['payment'][self.config.gem_idx[gem]] = int(amount)

  def _decode(self, data: "BuyDict") -> BuyCardActionGold:
    flat = int(data['card_idx'])
    idx = self.config.unflatten_card_idx(flat)
    pay_vec = data['payment']
    payment = self.config.decode_gems_list(pay_vec)
    return BuyCardActionGold.create(idx, None, payment=payment)

  def _sample(self, mask: BuyDict[np.bool] | None = None, probability: BuyDict[np.floating] | None = None) -> BuyDict:
    # card_idx: choose among flattened indices (visible + reserve + deck levels)
    card_mask = None if mask is None else np.asarray(mask['card_idx'], dtype=bool).copy()
    card_p = None if probability is None else probability['card_idx']

    max_card_index = self.config.max_card_index
    if card_mask is None:
      card_mask = np.ones(max_card_index, dtype=bool)
    else:
      card_mask = np.asarray(card_mask, dtype=bool).copy()
    card_mask[-self.config.card_level_count:] = False
    card_idx = sample_single(max_card_index, dtype=np.uint16, mask=card_mask, p=card_p, rng=self.np_random)

    # payment: sample a payment vector across gems (including gold). We allow 0..max_cost per gem
    pay_mask = None if mask is None else np.asarray(mask['payment'], dtype=bool).copy()
    pay_p = None if probability is None else probability['payment']

    # choose how many payment tokens are used in total (0..coin_gold_init)
    # keep it small for sampling: up to coin_gold_init
    max_pay_total = self.config.coin_gold_init
    payment_count = int(self.np_random.integers(0, max_pay_total + 1))

    if pay_mask is None:
      pay_mask_final = np.ones(self.config.gem_count, dtype=bool)
    else:
      pay_mask_final = np.asarray(pay_mask, dtype=bool).copy()
    pay_mask_final[self.config.gold_idx] = False  # exclude gold from payment sampling

    # sample_exact allows replacement so we can distribute payment_count across gems
    payment_sampled = sample_exact(self.config.gem_count, int(payment_count), dtype=np.int8, mask=pay_mask_final, p=pay_p, replacement=True, rng=self.np_random)

    return {
      'card_idx': np.array(card_idx, dtype=np.uint16),
      'payment_count': np.array(payment_count, dtype=np.int8),
      'payment': payment_sampled,
    }

  def sample(self, mask = None, probability = None) -> dict[str, Any]:
    return self._sample(mask=mask, probability=probability) # type: ignore[TypedDict]

class ReserveCardSpace(spaces.Dict):
  def __init__(self, config: ActionSpaceConfig | GameConfig, *, seed = None, **spaces_kwargs):
    self.config = config if isinstance(config, ActionSpaceConfig) else ActionSpaceConfig(config)
    super().__init__({
      'card_idx': spaces.Discrete(config.card_visible_total_count + config.card_max_count_reserved + config.card_level_count),
      'take_gold': spaces.Discrete(2),
      'ret_count': spaces.Box(0, 1),
      'ret': spaces.Box(low=0, high=1, shape=(config.gem_count,), dtype=np.int8),
    }, seed, **spaces_kwargs)

  def _encode(self, data: "ReserveDict", action: ReserveCardAction):
    reserve = data
    reserve['card_idx'][...] = 0
    if action.idx is not None:
      reserve['card_idx'][...] = self.config.flatten_card_idx(action.idx)
    reserve['take_gold'][...] = int(bool(action.take_gold))
    reserve['ret_count'][...] = 1 if action.ret is not None else 0
    reserve['ret'][...] = 0
    if action.ret is not None:
      reserve['ret'][self.config.gem_idx[action.ret]] = 1

  def _decode(self, data: "ReserveDict") -> ReserveCardAction:
    flat = int(data['card_idx'])
    idx = self.config.unflatten_card_idx(flat)
    take_gold = bool(int(data['take_gold']))
    _ret = self.config.decode_gems_list(data['ret'])
    ret = None
    for g in _ret or ():
      ret = g
      break
    return ReserveCardAction.create(idx, None, take_gold=take_gold, ret=ret)

  def _sample(self, mask: ReserveDict[np.bool] | None = None, probability: ReserveDict[np.floating] | None = None) -> ReserveDict:
    # card_idx: choose among flattened indices
    card_mask = None if mask is None else np.asarray(mask['card_idx'], dtype=bool).copy()
    card_p = None if probability is None else probability['card_idx']

    max_card_index = self.config.card_visible_total_count + self.config.card_max_count_reserved + self.config.card_level_count
    if card_mask is None:
      card_mask = np.ones(max_card_index, dtype=bool)
    else:
      card_mask = np.asarray(card_mask, dtype=bool).copy()
    # cannot select reserve indices when performing a reserve action
    card_mask[:self.config.card_max_count_reserved] = False
    card_idx = sample_single(max_card_index, dtype=np.uint16, mask=card_mask, p=card_p, rng=self.np_random)

    # take_gold: sample 0 or 1
    take_gold = int(self.np_random.integers(0, 2))

    # ret: at most one non-gold gem may be returned
    ret_mask = None if mask is None else np.asarray(mask['ret'], dtype=bool).copy()
    ret_p = None if probability is None else probability['ret']
    if ret_mask is None:
      ret_mask_final = np.ones(self.config.gem_count, dtype=bool)
    else:
      ret_mask_final = np.asarray(ret_mask, dtype=bool).copy()
    # cannot return gold when reserving
    ret_mask_final[self.config.gold_idx] = False

    ret_count = int(self.np_random.integers(0, take_gold + 1))
    if ret_count == 0:
      ret_sampled = np.zeros(self.config.gem_count, dtype=np.int8)
    else:
      ret_sampled = sample_exact(self.config.gem_count, 1, dtype=np.int8, mask=ret_mask_final, p=ret_p, replacement=False, rng=self.np_random)

    return {
      'card_idx': np.array(card_idx, dtype=np.uint16),
      'take_gold': np.array(take_gold, dtype=np.int8),
      'ret_count': np.array(ret_count, dtype=np.int8),
      'ret': ret_sampled,
    }

  def sample(self, mask = None, probability = None) -> dict[str, Any]:
    return self._sample(mask=mask, probability=probability) # type: ignore[TypedDict]


class ActionSpace(spaces.Dict):
  """Structured encoding of Action objects for gymnasium agents.

  The `type` field selects which subset of data is meaningful for the action.
  Other subsets remain zeroed so a single dict shape can represent all actions.
  """

  def __init__(self, config: GameConfig, *, seed = None):
    self.config = ActionSpaceConfig(config)
    self._type_order = tuple(ActionType)
    self._type_index = {atype: idx for idx, atype in enumerate(self._type_order)}
    # max index space for cards = visible cards + reserved capacity (3) + deck head levels
    # layout (flattened index):
    # 0..(visible-1) => visible_idx
    # visible..(visible+reserved-1) => reserve_idx (offset by visible)
    # visible+reserved.. => deck_head_level (offset by visible+reserved), one entry per level
    self._visible_count = config.card_visible_total_count
    self._reserve_count = config.card_max_count_reserved
    self._deck_levels = config.card_level_count
    self._max_card_index = self._visible_count + self._reserve_count + self._deck_levels
    self._gem_count = config.gem_count

    self._take2_space = Take2Space(self.config, seed=seed)
    self._take3_space = Take3Space(self.config, seed=seed)
    self._buy_space = BuyCardSpace(self.config, seed=seed)
    self._reserve_space = ReserveCardSpace(self.config, seed=seed)
    super().__init__({
      'type': spaces.Discrete(len(self._type_order)),
      'take3': self._take3_space,
      'take2': self._take2_space,
      'buy': self._buy_space,
      'reserve': self._reserve_space,
    }, seed=seed)

  def empty(self) -> "ActionDict":
    return {
      'type': np.array(0, dtype=np.int8),
      'take3': {
        'gems_count': np.array(0, dtype=np.int8),
        'ret_count': np.array(0, dtype=np.int8),
        'gems': np.zeros(self._gem_count, dtype=np.int8),
        'ret': np.zeros(self._gem_count, dtype=np.int8),
      },
      'take2': {
        'gem': np.array(0, dtype=np.int8),
        'count': np.array(0, dtype=np.int8),
        'ret_count': np.array(0, dtype=np.int8),
        'ret': np.zeros(self._gem_count, dtype=np.int8),
      },
      'buy': {
        'card_idx': np.array(0, dtype=np.uint16),
        'payment_count': np.array(0, dtype=np.int8),
        'payment': np.zeros(self._gem_count, dtype=np.int8),
      },
      'reserve': {
        'card_idx': np.array(0, dtype=np.uint16),
        'take_gold': np.array(0, dtype=np.int8),
        'ret_count': np.array(0, dtype=np.int8),
        'ret': np.zeros(self._gem_count, dtype=np.int8),
      },
    }

  def encode(self, action: Action) -> "ActionDict":
    data = self.empty()
    data['type'][...] = self._type_index[action.type]
    if isinstance(action, Take3Action):
      self._take3_space._encode(data["take3"], action)
      return data
    if isinstance(action, Take2Action):
      self._take2_space._encode(data['take2'], action)
      return data
    if isinstance(action, BuyCardActionGold):
      self._buy_space._encode(data['buy'], action)
      return data
    if isinstance(action, ReserveCardAction):
      self._reserve_space._encode(data['reserve'], action)
      return data
    if action.type == ActionType.NOOP:
      return data
    raise ValueError(f"Unsupported action type: {action.type}")

  def encode_many(self, actions: Sequence[Action]) -> list["ActionDict"]:
    return [self.encode(action) for action in actions]

  def decode(self, data: "ActionDict") -> Action:
    """Decode a structured ActionDict (as produced by `encode`) back into an Action object.

    This method simply dispatches to small, focused helpers for each
    action type to keep implementations readable and testable.
    """
    # resolve action type
    tval = int(data['type']) if hasattr(data['type'], '__int__') else int(data['type'].item())
    if tval < 0 or tval >= len(self._type_order):
      raise ValueError(f"Invalid action type index: {tval}")
    atype = self._type_order[tval]

    if atype == ActionType.TAKE_3_DIFFERENT:
      return self._take3_space._decode(data['take3'])
    if atype == ActionType.TAKE_2_SAME:
      return self._take2_space._decode(data['take2'])
    if atype == ActionType.BUY_CARD:
      return self._buy_space._decode(data['buy'])
    if atype == ActionType.RESERVE_CARD:
      return self._reserve_space._decode(data['reserve'])
    if atype == ActionType.NOOP:
      return Action.noop()
    raise ValueError(f"Unsupported action type for decode: {atype}")

  def decode_many(self, actions: Sequence["ActionDict"]) -> list[Action]:
    return [self.decode(a) for a in actions]


__all__ = ["ActionSpace"]
