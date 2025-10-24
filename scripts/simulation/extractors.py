from collections.abc import Callable

from gems.state import GameState


EXTRACTORS: dict[str, tuple[Callable, bool]] = {}


def extractor(need_group=False):
  def decorator(func):
    EXTRACTORS[func.__name__] = (func, need_group)

    def wrapper(*args, **kwargs):
      return func(*args, **kwargs)
    return wrapper
  return decorator


def _extract_scores(states_list: list[list[GameState]], seat_id: int) -> list[list[int]]:
  scores_list: list[list[int]] = []
  for states in states_list:
    scores: list[int] = [state.players[seat_id].score for state in states]
    scores_list.append(scores)
  return scores_list


@extractor()
def single_player_extract_scores(states_list: list[list[GameState]]) -> list[list[int]]:
  return _extract_scores(states_list, seat_id=0)


def _average_scores(scores_list: list[list[int]]) -> list[float]:
  max_len = max(len(scores) for scores in scores_list)
  averages: list[float] = []
  for turn in range(max_len):
    # gather scores for this turn from games that lasted at least this long
    vals: list[int] = [scores[turn] for scores in scores_list if len(scores) > turn]
    averages.append(sum(vals) / len(vals))
  return averages


@extractor(True)
def single_player_extract_average_scores(states_list: list[list[GameState]]) -> list[float]:
  """Compute per-turn average scores across multiple games.
  - states_list: list of game states (one per game). Games may have
    different lengths; shorter games contribute only to their existing turns.

  Returns a list of floats where element i is the average score at turn i+1.
  """
  if not states_list:
    return []
  scores_list = single_player_extract_scores(states_list)
  return _average_scores(scores_list)


@extractor()
def multiplayer_extract_average_scores(states_list: list[list[GameState]]) -> list[list[float]]:
  """Compute per-turn average scores for each player across multiple games.
  - states_list: list of game states (one per game). Games may have
    different lengths; shorter games contribute only to their existing turns.

  Returns a list of lists where element i is a list of average scores
  for each player at turn i+1.
  """
  if not states_list:
    return []
  num_players = len(states_list[0][0].players)

  avg_scores_list = []
  for seat_id in range(num_players):
    scores_list = _extract_scores(states_list, seat_id)
    avg_scores = _average_scores(scores_list)
    avg_scores_list.append(avg_scores)
  return avg_scores_list
