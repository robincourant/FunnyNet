import argparse
import os
import os.path as osp
import pickle

import pandas as pd
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sub_dir",
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
    args = parser.parse_args()

    return args.sub_dir, args.label_path, args.save_dir, args.time_step


if __name__ == "__main__":
    sub_dir, label_path, save_dir, time_step = parse_arguments()

    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(
        save_dir, f"sub_{label_path.split('/')[-1][:-3]}_{int(time_step)}s.pk"
    )

    file_name = [f for f in os.listdir(sub_dir) if osp.isfile(osp.join(sub_dir, f))]
    record = []
    label_files = pd.read_pickle(label_path)
    for i in tqdm(range(len(label_files))):
        season = 3
        episode = label_files[i][1]  # 0 for old (plus +1), and 1 for new
        label = label_files[i][2]  # 3 for old, and 2 for new
        st = label_files[i][3]  # 1 for old, 3 for new
        et = label_files[i][4]  # 2 for old, 4 for new
        if label == 0:
            frame_end = et
            frame_start = st
        else:
            frame_end = st
            if frame_end < time_step:
                frame_end = time_step
            frame_start = frame_end - time_step

        file_name = "friends.s03e" + str(episode).zfill(2) + ".720p.bluray"
        sub_file = osp.join(sub_dir, file_name + ".pkl")
        sub_label_files = pd.read_pickle(sub_file)

        tmp = ""
        for j in range(len(sub_label_files)):
            sub_st = sub_label_files[j][0][0]
            sub_et = sub_label_files[j][0][1]
            if (sub_st > frame_start and sub_st < frame_end) or (
                sub_et > frame_start and sub_et < frame_end
            ):
                sub_record = sub_label_files[j][1]
                tmp = tmp + sub_record
        posi = [season, episode, label, tmp]
        record.append(posi)

    with open(save_path, "wb") as f:
        pickle.dump(record, f)
