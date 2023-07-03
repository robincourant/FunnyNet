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

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    audio_file = sorted(os.listdir(audio_dir))
    for file_name in tqdm(audio_file):
        audio_name = os.path.join(audio_dir, file_name)
        file_prefix = file_name[:-4]
        label_name = os.path.join(label_path, file_prefix + ".pk")
        annotation = pd.read_pickle(label_name)

        len_file = len(annotation)
        result = []
        begin = 0
        song = AudioSegment.from_wav(audio_name)
        for j in range(0, len_file, 1):
            frame_end = annotation[j][1]
            frame_start = annotation[j][0]

            st = int(frame_start * 1000)
            et = int(frame_end * 1000)

            s = song[begin:st]
            result.append(s)

            # create silence
            duration = et - st
            s = AudioSegment.silent(duration)
            result.append(s)

            # value for next loop
            begin = et

        result.append(song[begin:])

        # join all parts using standard `sum()` but it need `parts[0]` as start value
        b = sum(result[1:], result[0])
        save_path = os.path.join(save_dir, file_name)
        b.export(save_path, format="wav")
