# %%
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_scores(score_lists: list[list[int]] | list[list[float]], labels: list[str] | None = None) -> Figure:
  """Plot score progress for each replay using matplotlib.
  - score_lists: list of score sequences (one per game)
  """
  if not labels:
    labels = [f"Game {i + 1}" for i in range(len(score_lists))]
  if len(labels) != len(score_lists):
    raise ValueError("Length of labels must match length of score_lists")

  fig = plt.figure()
  ax = fig.add_subplot(111)
  for scores, label in zip(score_lists, labels):
    x = list(range(1, len(scores) + 1))
    ax.plot(x, scores, marker=".", label=label)

  ax.set_xlabel("Round")
  ax.set_ylabel("Score")
  ax.set_title("Average Score over Round")
  ax.legend(loc="upper left")
  ax.grid(True, linestyle="--", alpha=0.4)
  return fig


def plot_rounds(finish_rounds: list[int], label: str) -> Figure:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.hist(finish_rounds, bins=range(0, max(finish_rounds) + 2, 1), label=label)
  ax.set_xlabel("Number of rounds to finish")
  ax.set_ylabel("Number of games")
  title = "Distribution of number of rounds to finish games"
  ax.set_title(title)
  return fig


def plot_winrate(win_counts: dict[int, int], total_games: int, player_labels: list[str] | None = None) -> Figure:
  if player_labels is None:
    player_labels = [f"Player {i}" for i in range(len(win_counts))]
  if len(player_labels) != len(win_counts):
    raise ValueError("Length of labels must match length of win_counts")
  fig = plt.figure()
  ax = fig.add_subplot(111)

  total_counts = sum(win_counts.values())
  seatid_counts = sorted(win_counts.items())
  print("Win counts per seat_id:", seatid_counts)

  counts = [v for _, v in seatid_counts]

  # Prepare labels with counts and percentages (use provided total for percent)
  percents = [(c / total_counts) * 100 for c in counts]
  labels = [f"{lab}\n{c} wins\n{p:.1f}%" for lab, c, p in zip(player_labels, counts, percents)]

  # Draw a donut chart (pie with a hole)
  # Build a color list from the tab20 colormap sized to the number of players
  cmap = plt.get_cmap("tab20")
  # sample the colormap evenly
  if len(counts) == 1:
    colors = [cmap(0.0)]
  else:
    colors = [cmap(i / (len(counts) - 1)) for i in range(len(counts))]
  ax.pie(
      counts,
      labels=labels,
      startangle=90,
      counterclock=False,
      colors=colors,
      wedgeprops=dict(width=0.4, edgecolor="w"),
  )

  ax.set_title(f"Win counts (total games: {total_games})")
  ax.axis("equal")
  fig.tight_layout()
  return fig


# %%
win_counts = {0: 60, 1: 30, 2: 20}
total_games = 100
fig = plot_winrate(win_counts, total_games, player_labels=["Alice", "Bob", "Charlie"])
fig.show()
