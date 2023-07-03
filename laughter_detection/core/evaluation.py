from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

import audioread
import numpy as np


def compute_temporal_iou(
    segment_1: Tuple[float, float], segment_2: Tuple[float, float]
) -> float:
    """Compute temporal IoU between two timecodes."""
    start_1, end_1 = segment_1
    start_2, end_2 = segment_2

    temporal_intersection = max(0, min(end_1, end_2) - max(start_1, start_2))
    temporal_union = max(end_1, end_2) - min(start_1, start_2)
    temporal_iou = temporal_intersection / temporal_union

    return temporal_iou


def get_cross_iou(
    true_segment_list: List[Tuple[float, float]],
    pred_segment_list: List[Tuple[float, float]],
) -> np.array:
    """Compute IoU between each pair of true and predicted segments."""
    n_segments_1, n_segments_2 = len(true_segment_list), len(pred_segment_list)
    cross_iou = np.zeros((n_segments_1, n_segments_2))
    for i, j in product(range(n_segments_1), range(n_segments_2)):
        iou = compute_temporal_iou(true_segment_list[i], pred_segment_list[j])
        cross_iou[i, j] = iou

    return cross_iou


def get_detection_confusionmatrix(
    true_segments: List[Tuple[float, float]],
    pred_segments: List[Tuple[float, float]],
    iou_threshold: float = 0.3,
) -> Tuple[int, int, int, int, int]:
    """
    Compute detection confusion matrix elements by comparing true and predicted
    segments with IoU.
    """
    # Compute IoU between all true/predicted segment pairs
    laughter_ious = get_cross_iou(pred_segments, true_segments)
    # Compute confusion matrix elements
    n_true = len(true_segments)
    n_predicted = len(pred_segments)
    # Count the true prositives, false positives and false negatives
    n_TP = ((laughter_ious >= iou_threshold).sum(axis=0) > 0).sum()
    n_FP = n_predicted - n_TP
    n_FN = n_true - n_TP

    return n_true, n_predicted, n_TP, n_FP, n_FN


def compute_detection_metrics(
    n_true_positive: int,
    n_false_positive: int,
    n_false_negative: int,
) -> Tuple[float, float]:
    """Compute precision, recall and F1 given detection TP, FP and FN."""
    precision = n_true_positive / (n_true_positive + n_false_positive)
    recall = n_true_positive / (n_true_positive + n_false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def compute_temporal_metrics(
    n_true_positive: int,
    n_true_negative: int,
    n_false_positive: int,
    n_false_negative: int,
):
    """
    Compute accuracy, precision, recall and F1 given temporal TP, FP and FN.
    """
    accuracy = (n_true_positive + n_true_negative) / (
        n_true_positive + n_true_negative + n_false_positive + n_false_negative
    )
    precision = n_true_positive / (n_true_positive + n_false_positive)
    recall = (n_true_positive) / (n_true_positive + n_false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


def get_audiolength(path: str):
    """Get the number of frames (length) of the audio."""
    audio = audioread.audio_open(path)
    audio_length = audio.duration
    audio.close()
    return audio_length


def get_overlap(start_1: float, end_1: float, start_2: float, end_2: float):
    """Get the number of frames that overlaps between two segments."""

    def check_overlap(
        start_1: float, end_1: float, start_2: float, end_2: float
    ):
        """Check wether segments overlap."""
        if end_1 <= start_2 or end_2 <= start_1:
            return False
        else:
            return True

    if not check_overlap(start_1, end_1, start_2, end_2):
        return 0.0

    # Equal on one side
    elif start_1 == start_2:
        return np.minimum(end_1, end_2) - start_1
    elif end_1 == end_2:
        return end_1 - np.maximum(start_1, start_2)

    # One contained totally within the other
    elif start_2 > start_1 and end_2 < end_1:
        return end_2 - start_2
    elif start_1 > start_2 and end_1 < end_2:
        return end_1 - start_1

    # Overlap on one side
    elif end_1 > start_2 and start_1 < start_2:
        return end_1 - start_2
    elif end_2 > start_1 and start_2 < start_1:
        return end_2 - start_1


def get_totaloverlap(
    true_segments: List[Dict[str, float]],
    predicted_segments: List[Dict[str, float]],
):
    """Get the number of frames that overlaps between predictions and labels"""
    total_overlap = 0
    for ts in true_segments:
        for ps in predicted_segments:
            total_overlap += get_overlap(
                ts["start"], ts["end"], ps["start"], ps["end"]
            )
    return total_overlap


def get_nonlaughterlength(
    laughter_segments: List[Dict[str, float]],
    window_start: float,
    window_length: float,
    avoid_edges: bool = True,
    edge_gap: float = 0.5,
):
    """Get the number of frames (length) of non laughter parts."""
    non_laughter_segments = []

    if avoid_edges:
        non_laughter_start = window_start + edge_gap
    else:
        non_laughter_start = window_start
    for segment in laughter_segments:
        non_laughter_end = segment["start"]
        if non_laughter_end > non_laughter_start:
            non_laughter_segments.append(
                {"start": non_laughter_start, "end": non_laughter_end}
            )
        non_laughter_start = segment["end"]

    if avoid_edges:
        non_laughter_end = window_start + window_length - edge_gap
    else:
        non_laughter_end = window_length

    if non_laughter_end > non_laughter_start:
        non_laughter_segments.append(
            {"start": non_laughter_start, "end": non_laughter_end}
        )
    return non_laughter_segments


def get_frame_confusionmatrix(
    true_laughter_segments: List[Dict[str, float]],
    pred_laughter_segments: List[Dict[str, float]],
    max_time: float,
    min_gap: float = 0.0,
    threshold: float = 0.5,
    use_filter: bool = False,
    min_length: float = 0.0,
    avoid_edges: bool = True,
    edge_gap: float = 0.5,
    expand_channel_dim: bool = False,
):
    """
    Compute frame confusion matrix elements by comparing true and predicted
    segments frame by frame.
    """
    true_nonlaughter_segments = get_nonlaughterlength(
        true_laughter_segments,
        0,
        max_time,
        avoid_edges=avoid_edges,
        edge_gap=edge_gap,
    )
    pred_nonlaughter_segments = get_nonlaughterlength(
        pred_laughter_segments,
        0,
        max_time,
        avoid_edges=avoid_edges,
        edge_gap=edge_gap,
    )

    n_true_positive = get_totaloverlap(
        true_laughter_segments, pred_laughter_segments
    )
    n_true_negative = get_totaloverlap(
        true_nonlaughter_segments, pred_nonlaughter_segments
    )
    n_false_positive = get_totaloverlap(
        true_nonlaughter_segments, pred_laughter_segments
    )
    n_false_negative = get_totaloverlap(
        true_laughter_segments, pred_nonlaughter_segments
    )

    return (
        n_true_positive,
        n_true_negative,
        n_false_positive,
        n_false_negative,
    )


def get_temporal_scores(
    pred_timecodes: List[Tuple[float]],
    true_timecodes: List[Tuple[float]],
    audio_path: str,
):
    """Compute frame scaled scores."""
    pred_laughter_segments = [
        {"start": start, "end": end} for start, end in pred_timecodes
    ]
    true_laughter_segments = [
        {"start": start, "end": end} for start, end in true_timecodes
    ]

    max_time = get_audiolength(audio_path)

    n_TP, n_TN, n_FP, n_FN = get_frame_confusionmatrix(
        true_laughter_segments, pred_laughter_segments, max_time
    )

    accuracy, precision, recall, f1 = compute_temporal_metrics(
        n_TP, n_TN, n_FP, n_FN
    )

    frame_scores = {
        "true_positive_time": n_TP,
        "true_negative_time": n_TN,
        "false_positive_time": n_FP,
        "false_negative_time": n_FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return frame_scores


def get_detection_scores(
    pred_timecodes: List[Tuple[float]],
    true_timecodes: List[Tuple[float]],
    ious: List[float],
):
    """Compute detection scaled scores for different IoUs."""
    detection_scores = defaultdict(lambda: defaultdict(int))
    for iou in ious:
        n_true, n_predicted, n_TP, n_FP, n_FN = get_detection_confusionmatrix(
            true_timecodes, pred_timecodes, iou_threshold=iou
        )
        precision, recall, f1 = compute_detection_metrics(n_TP, n_FP, n_FN)

        detection_scores["n_trues"][str(iou)] = n_true
        detection_scores["n_preds"][str(iou)] = n_predicted
        detection_scores["true_positives"][str(iou)] = n_TP
        detection_scores["false_positives"][str(iou)] = n_FP
        detection_scores["false_negatives"][str(iou)] = n_FN
        detection_scores["iou_precs"][str(iou)] = precision
        detection_scores["iou_recalls"][str(iou)] = recall
        detection_scores["iou_f1"][str(iou)] = f1

    return dict(detection_scores)
