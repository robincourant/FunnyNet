import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_segments(
    true_segment_list: List[Tuple[float, float]],
    pred_segment_list: List[Tuple[float, float]],
    t_min: float = None,
    t_max: float = None,
    zoom: bool = True,
    marker_list: List[int] = None,
):
    """Display true and predicted timecodes on a common timeline.

    :param true_segment_list: list of groundtruth timecodes.
    :param pred_segment_list: list of predicted timecodes.
    :param t_min: timecode from which starting the timeline.
    :param t_max: timecode to which ending the timeline.
    :param zoom: wether to display the diagram in a "zoom" fashion or not
        (ie: with details, the timeline should be short).
    :param marker_list: list of markers to add on the diagram (gray lines).
    """
    true_segment_list = sorted(true_segment_list)
    pred_segment_list = sorted(pred_segment_list)

    x_max = max(true_segment_list[-1][-1], pred_segment_list[-1][-1])
    t_min = 0 if not t_min else t_min
    t_max = x_max if not t_max else t_max

    true_segment_list = [
        [t1, t2]
        for t1, t2 in true_segment_list
        if (t1 >= t_min) and (t2 <= t_max)
    ]
    pred_segment_list = [
        [t1, t2]
        for t1, t2 in pred_segment_list
        if (t1 >= t_min) and (t2 <= t_max)
    ]

    plt.figure(figsize=(20, 5))
    prev_x_min = t_min
    for x_min, x_max in true_segment_list:
        if zoom:
            plt.vlines(x_min, 1, 2, color="#e41a1c", linestyles="dashed")
            plt.vlines(x_max, 1, 2, color="#e41a1c", linestyles="dashed")
            plt.fill_between([x_min, x_max], 1, 2, color="#e41a1c", alpha=0.1)
            plt.hlines(
                1,
                prev_x_min,
                x_min,
                color="black",
                linewidth=2,
                linestyles="dashed",
            )
        plt.hlines(1, x_min, x_max, color="#e41a1c", linewidth=4)
        prev_x_min = x_max
    plt.hlines(
        1, x_max, t_max, color="black", linewidth=2, linestyles="dashed"
    )

    prev_x_min = t_min
    for x_min, x_max in pred_segment_list:
        if zoom:
            plt.vlines(x_min, 1, 2, color="#377eb8", linestyles="dashed")
            plt.vlines(x_max, 1, 2, color="#377eb8", linestyles="dashed")
            plt.fill_between([x_min, x_max], 1, 2, color="#377eb8", alpha=0.1)
            plt.hlines(
                2,
                prev_x_min,
                x_min,
                color="black",
                linewidth=2,
                linestyles="dashed",
            )
        plt.hlines(2, x_min, x_max, color="#377eb8", linewidth=4)
        prev_x_min = x_max
    plt.hlines(
        2, x_max, t_max, color="black", linewidth=2, linestyles="dashed"
    )

    if marker_list is not None:
        marker_list = [t for t in marker_list if (t >= t_min) and (t <= t_max)]
        for timecode in marker_list:
            plt.vlines(timecode, 1, 2, color="#000000")

    pred_legend = mpatches.Patch(color="#e41a1c", label="pred")
    true_legend = mpatches.Patch(color="#377eb8", label="true")
    plt.legend(handles=[pred_legend, true_legend], loc=6)
    plt.show()


def load_labels(label_path: str) -> List[Tuple[float, float]]:
    """Load a Friends label file and extract laugther timecodes."""
    labels = pickle.load(open(label_path, "rb"))

    true_timecodes = []
    for segment in labels.values():
        if segment[-1][-10:-2].lower() == "laughter":
            true_timecodes.append(segment[:2])

    return sorted(true_timecodes)


def load_preds(pred_path: str) -> List[Tuple[float, float]]:
    """Load a prediction file with laugther timecodes."""
    preds = pickle.load(open(pred_path, "rb"))
    return sorted(preds)
