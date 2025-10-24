from matplotlib import pyplot as plt


def plot_scores(score_lists: list[list[int]] | list[list[float]], labels: list[str] | None = None) -> None:
  """Plot score progress for each replay using matplotlib.
  - score_lists: list of score sequences (one per game)
  """
  if not labels:
    labels = [f"Game {i + 1}" for i in range(len(score_lists))]
  if len(labels) != len(score_lists):
    raise ValueError("Length of labels must match length of score_lists")

  # plt.figure(figsize=(8, 4 + len(score_lists) * 0.5))
  for scores, label in zip(score_lists, labels):
    x = list(range(1, len(scores) + 1))
    plt.plot(x, scores, marker=".", label=label)

  plt.xlabel("Turn")
  plt.ylabel("Score")
  plt.title("Score over turn")
  plt.legend(loc="upper left")
  plt.grid(True, linestyle="--", alpha=0.4)
  plt.show()


def plot_rounds(finish_rounds: list[int], label: str | None = None) -> None:
  plt.hist(finish_rounds, bins=range(0, max(finish_rounds) + 2, 1))
  plt.xlabel("Number of rounds to finish")
  plt.ylabel("Number of games")
  label = label or "Distribution of number of rounds to finish games"
  plt.title(label)
  plt.show()
