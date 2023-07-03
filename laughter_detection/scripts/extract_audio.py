"""
This script extracts the audio tracks from all video files within a specified
directory, and saves tracks in a `raw`-named directory.
"""

import argparse
import os
import os.path as osp
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rootdir", type=str, help="Path to the root of FunnyNet dataset"
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    rootdir = args.rootdir
    clip_dir = osp.join(rootdir, "episode")

    # Check if the `audio/raw` directory exists, or create it
    audio_dir = osp.join(rootdir, "audio", "raw")
    if not osp.exists(audio_dir):
        os.makedirs(audio_dir)

    # Iterate over all video files of the specified directory
    filename_list = os.listdir(clip_dir)
    n_files = len(filename_list)
    for k, filename in enumerate(filename_list):
        clip_path = osp.join(clip_dir, filename)
        audio_path = osp.join(audio_dir, filename[:-3] + "wav")
        command_output = subprocess.call(
            [
                "ffmpeg",
                "-loglevel",
                "panic",
                "-i",
                clip_path,
                "-q:a",
                "0",
                "-map",
                "a",
                audio_path,
            ]
        )

        # Check if the command terminates
        if command_output:
            print(f"{k + 1} / {n_files}] [Warning] Something went wrong")
        else:
            print(f"[{k + 1} / {n_files}] File {audio_path} saved")
