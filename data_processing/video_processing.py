import argparse
import os
import os.path as osp

import cv2
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_dir",
        type=str,
        help="Path to the directory with subtitle files",
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
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Size of images after resizing",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=23.976023976023978,
        help="Fps",
    )
    args = parser.parse_args()

    return (
        args.video_dir,
        args.label_path,
        args.save_dir,
        args.time_step,
        args.img_size,
        args.fps,
    )


if __name__ == "__main__":
    video_dir, label_path, save_dir, time_step, img_size, fps = parse_arguments()

    new_size_in = tuple([img_size, img_size])
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    save_dir = osp.join(save_dir, f"{label_path.split('/')[-1][:-3]}_{int(time_step)}s")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    label_files = pd.read_pickle(label_path)
    for file_index, file_data in tqdm(enumerate(label_files), total=len(label_files)):
        season = int(file_data[0])
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
        file_path = osp.join(video_dir, file_name + ".mkv")
        clip = VideoFileClip(file_path).subclip(int(frame_start), int(frame_end))

        out_file = osp.join(save_dir, str(file_index).zfill(5) + ".mp4")
        clip.write_videofile(out_file, verbose=False, logger=None)
