"""
This scripts load detected laughter timecodes saved in the directory
`root_dir/audio/laughter`, extract each audio segment from the long audio file
and save each segment in the directory `root_dir/audio/laughter_segment`.
"""

import argparse
import os
import os.path as osp

import soundfile as sf
from tqdm import tqdm
import torchaudio

from laughter_detection.core.utils import load_preds


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root_dir", type=str, help="Path to the root of FunnyNet dataset"
    )
    parser.add_argument(
        "--laughter-dir",
        "-l",
        type=str,
        help="Path to the laughter directory",
        default=None,
    )
    parser.add_argument(
        "--segment-dir",
        "-o",
        type=str,
        help="Path to the output directory",
        default=None,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    root_dir = args.root_dir
    laughter_dir = args.laughter_dir
    segment_dir = args.segment_dir

    raw_dir = osp.join(root_dir, "audio", "raw")
    if not laughter_dir:
        laughter_dir = osp.join(root_dir, "audio", "laughter")
    if not segment_dir:
        segment_dir = osp.join(root_dir, "audio", "laughter_segment")

    audio_filenames = sorted(os.listdir(raw_dir))
    _, sample_rate = torchaudio.load(
        osp.join(raw_dir, audio_filenames[0]), num_frames=1000
    )

    for current_filename in tqdm(audio_filenames):
        laughter_filename = f"{current_filename[:-4]}.pk"

        # Load pre-computed prediction timecodes
        preds = load_preds(osp.join(laughter_dir, laughter_filename))

        episode_segment_dir = osp.join(segment_dir, current_filename[:-4])
        if not osp.exists(episode_segment_dir):
            os.makedirs(episode_segment_dir)

        # Load from full audio track each laughter segment and save it
        for k, (start_timecode, end_timecode) in enumerate(preds):
            # Get segment boundaries
            start_index = int(sample_rate * start_timecode)
            duration = int(sample_rate * (end_timecode - start_timecode))
            # Load segment from full audio track
            laughter_segment, _ = torchaudio.load(
                osp.join(raw_dir, current_filename),
                frame_offset=start_index,
                num_frames=duration,
            )
            # Save segment
            laughter_segment_path = osp.join(
                episode_segment_dir, f"{str(k).zfill(3)}.wav"
            )
            sf.write(laughter_segment_path, laughter_segment.T, sample_rate)
