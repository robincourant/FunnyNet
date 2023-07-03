"""
This script detects laughter within all audio files contained in the directory
`root_dir/audio/raw`, and save one pickle file for each audio file with
laughter timecodes in the directory `root_dir/audio/laughter`.
"""


import argparse
import os
import os.path as osp
import pickle

from laughter_detection.core.laughter_detector import LaughterDetector


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="Path to the root of FunnyNet dataset"
    )
    parser.add_argument(
        "--embedding-name",
        "-e",
        type=str,
        help="embedding model to use.",
        default="byola",
    )
    parser.add_argument(
        "--laughter-dir",
        "-l",
        type=str,
        help="Path to the directory to save detected laughters",
        default=None,
    )
    parser.add_argument(
        "--n-clusters", "-n", type=int, help="Number of clusters", default=3
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    embedding_name = args.embedding_name
    root_dir = args.root_dir
    laughter_dir = args.laughter_dir
    n_clusters = args.n_clusters

    if not laughter_dir:
        laughter_dir = osp.join(root_dir, "audio", "laughter", embedding_name)

    if not osp.exists(laughter_dir):
        os.makedirs(laughter_dir)

    raw_dir = osp.join(root_dir, "audio", "raw")
    audio_filenames = sorted(os.listdir(raw_dir))

    laughter_detector = LaughterDetector(
        embedding_name, root_dir, num_workers=6, n_clusters=n_clusters
    )
    pred_timecodes = laughter_detector.detect_laughters()

    for current_filename, current_timecodes in pred_timecodes.items():
        laughter_filename = f"{current_filename[:-4]}.pk"
        laughter_path = osp.join(laughter_dir, laughter_filename)

        # Save laughter timecodes
        with open(laughter_path, "wb") as f:
            pickle.dump(current_timecodes, f)
