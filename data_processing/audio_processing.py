import argparse
import os
import os.path as osp

import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir",
        type=str,
        help="Path to the directory with audio files",
    )
    parser.add_argument(
        "label_path",
        type=str,
        help="Path to the laughter annotation file",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Path to the saving directory",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=8.0,
        help="Lenght of the time window",
    )
    args = parser.parse_args()

    return args.audio_dir, args.label_path, args.save_dir, args.time_step


if __name__ == "__main__":
    audio_dir, label_path, save_dir, time_step = parse_arguments()

    save_dir = osp.join(save_dir, f"{label_path.split('/')[-1][:-3]}_{int(time_step)}s")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    label_files = pd.read_pickle(label_path)
    for file_index, file_data in tqdm(enumerate(label_files), total=len(label_files)):
        season = 3
        episode = int(file_data[1])
        label = int(file_data[2])
        if label == 0:
            frame_end = file_data[4]
            frame_start = file_data[3]
        else:
            frame_end = file_data[3]
            if frame_end < time_step:
                frame_end = time_step
            frame_start = frame_end - time_step

        file_name = "friends.s03e" + str(int(episode)).zfill(2) + ".720p.bluray"
        file_path = osp.join(audio_dir, file_name + ".wav")
        audio = AudioSegment.from_wav(file_path)

        st = int(frame_start) * 1000
        et = int(frame_end) * 1000
        extract = audio[st:et]

        # Save
        out_file = osp.join(save_dir, str(file_index).zfill(5) + ".wav")
        extract.export(out_file, format="wav")
