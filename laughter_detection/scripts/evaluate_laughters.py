import argparse
from collections import defaultdict
import os
import os.path as osp

import numpy as np
import pandas as pd

from laughter_detection.core.utils import load_labels, load_preds
from laughter_detection.core.evaluation import (
    get_detection_scores,
    get_temporal_scores,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred_dir", type=str, help="Path to the prediction directory (.pickle)"
    )
    parser.add_argument(
        "label_dir", type=str, help="Path to the label directory (.pickle)"
    )
    parser.add_argument("audio_dir", type=str, help="Path to the audio directory (.wav)")
    parser.add_argument(
        "output_dir", type=str, help="Path to the output score directory"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    pred_dir = args.pred_dir
    label_dir = args.label_dir
    audio_dir = args.audio_dir
    output_dir = args.output_dir

    pred_filenames = sorted(os.listdir(pred_dir))
    label_filenames = sorted(os.listdir(label_dir))

    temporal_scores, detect_scores = {}, defaultdict(list)
    for pred_name, label_name in zip(pred_filenames, label_filenames):
        print(pred_name, label_name)
        audio_path = osp.join(audio_dir, pred_name[:-3] + ".wav")

        # Load predicted and true laughter timecodes
        pred_timecodes = load_preds(osp.join(pred_dir, pred_name))
        true_timecodes = load_labels(osp.join(label_dir, label_name))

        # Get frame scaled scores
        episode_framescores = get_temporal_scores(
            pred_timecodes, true_timecodes, audio_path
        )
        temporal_scores[pred_name] = episode_framescores

        # Get detection scaled scores
        episode_detectionscores = get_detection_scores(
            pred_timecodes, true_timecodes, np.arange(0.3, 0.8, 0.1)
        )
        detect_scores["trues"].append(episode_detectionscores["n_trues"])
        detect_scores["predictions"].append(episode_detectionscores["n_preds"])
        detect_scores["TP"].append(episode_detectionscores["true_positives"])
        detect_scores["FP"].append(episode_detectionscores["false_positives"])
        detect_scores["FN"].append(episode_detectionscores["false_negatives"])
        detect_scores["precs"].append(episode_detectionscores["iou_precs"])
        detect_scores["recalls"].append(episode_detectionscores["iou_recalls"])
        detect_scores["f1"].append(episode_detectionscores["iou_f1"])

    temporal_score_df = pd.DataFrame.from_dict(temporal_scores, orient="index")
    detect_score_df = pd.DataFrame.from_dict(
        {
            (i, str(round(float(j), 2))): [
                detect_scores[i][k2][j] for k2 in range(len(detect_scores[i]))
            ]
            for i in detect_scores.keys()
            for k1 in range(len(detect_scores[i]))
            for j in detect_scores[i][k1].keys()
        }
    )
    detect_score_df.index = temporal_score_df.index

    # Add a row with the mean of each column
    temporal_score_df.loc["mean"] = temporal_score_df.mean()
    detect_score_df.loc["mean"] = detect_score_df.mean()

    # Save scores
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    temporal_score_df.to_csv(osp.join(output_dir, "temporal_score.csv"))
    detect_score_df.to_csv(osp.join(output_dir, "detection_score.csv"))
