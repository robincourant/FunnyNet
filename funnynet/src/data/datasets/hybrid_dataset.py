import itertools
import os.path as osp
import re
import warnings

from facenet_pytorch import MTCNN, InceptionResnetV1
from omegaconf import DictConfig
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as TA
import torchvision.transforms as TV
import torchvision
from transformers import BertTokenizer

from ext.byol_a.byol_a.augmentations import PrecomputedNorm

warnings.filterwarnings("ignore")
torchvision.set_video_backend("video_reader")

EPS = torch.finfo(torch.float).eps


class HybridDataset(Dataset):
    def __init__(self, split: str, config: DictConfig):
        super().__init__()
        self.split = split
        self.modalities = config.modalities
        self.faces = False

        # ############# audio parameters #################
        if "a" in self.modalities:
            self.resample = config.audio.resample
            self.num_mels = config.audio.num_mels
            if self.num_mels > 0:
                if config.audio.fbank:
                    self.fbank = True
                    self.spectrogram = False
                    self.target_length = config.audio.target_length
                else:
                    self.fbank = False
                    self.spectrogram = True
                    self.normalizer = PrecomputedNorm(config.audio.stats)
                    self.transform_spectrogram = torchaudio.transforms.MelSpectrogram(
                        sample_rate=self.resample, n_mels=self.num_mels
                    )
            else:
                self.fbank = False
                self.spectrogram = False

        # ############# vision parameters ################
        if "v" in self.modalities:
            self.frame_rate = config.vision.frame_rate
            self.img_size = config.vision.img_size
            self.num_frame = config.vision.num_frame
            self.time_step = config.vision.time_step
            self.mean = config.vision.mean
            self.std = config.vision.std
            self.tchw = config.vision.tchw

            self.resize_frames = TV.Compose(
                [
                    TV.Resize((self.img_size, self.img_size)),
                    TV.Normalize(self.mean, self.std),
                ]
            )
            self.augment_frames = TV.Compose(
                [
                    TV.Resize((self.img_size, self.img_size)),
                    TV.RandomHorizontalFlip(p=0.5),
                    TV.RandomVerticalFlip(p=0.5),
                    TV.Normalize(self.mean, self.std),
                ]
            )

        # ############# sub parameters ################
        if "t" in self.modalities:
            self.tokenizer = BertTokenizer.from_pretrained(config.text.weight_path)

        # ############# face parameters ################
        if "f" in self.modalities:
            self.mtcnn = MTCNN(
                image_size=config.face.face_size,
                margin=config.face.margin,
                min_face_size=config.face.min_face_size,
                keep_all=config.face.keep_all,
                thresholds=config.face.thresholds,
                factor=config.face.factor,
                post_process=config.face.post_process,
            )
            self.face_resnet = InceptionResnetV1(pretrained="vggface2").eval()
            self.faces = True

        # ############# label file #######################
        self.classes_for_all_imgs = []
        self.annotation = []
        self.audio = []
        self.subtitle = []
        self.video = []
        annotation_path = osp.join(config.data_dir, f"laughter/{split}.pk")
        annotation = pd.read_pickle(annotation_path)
        sub_path = osp.join(config.data_dir, f"sub_split/{split}_8s.pk")
        subtitle = pd.read_pickle(sub_path)
        video_dir = osp.join(config.data_dir, f"video_split/{split}_8s")
        audio_dir = osp.join(config.data_dir, f"audio_split/{split}_8s")
        split_index = 1 if split == "train" else 2
        for i in range(len(annotation)):
            self.annotation.append([split_index, i, annotation[i][2]])
            self.audio.append([split_index, i, audio_dir])
            self.video.append([split_index, i, video_dir])

            sub = subtitle[i][-1]
            sub = re.sub("\n", " ", sub)
            sub = re.sub("-", " ", sub)
            self.subtitle.append([split_index, i, sub])

            clas_id = annotation[i][2]  # 3 for old version, 2 for new version
            self.classes_for_all_imgs.append(clas_id)

        self.load_audio = (
            self.load_aug_audio if self.split == "train" else self.load_raw_audio
        )
        self.load_frames = (
            self.load_aug_frames if self.split == "train" else self.load_raw_frames
        )

    def __len__(self):
        return len(self.annotation)

    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs

    def load_aug_audio(self, index):
        waveform_path = self.audio[index][2]
        no = self.annotation[index][1]

        wav_name = osp.join(waveform_path, str(no).zfill(5) + ".wav")

        waveform, sample_rate = torchaudio.load(wav_name)

        resampler = TA.Resample(sample_rate, self.resample)
        waveform = resampler(waveform)
        frame_length = waveform.shape[1] / self.resample

        if frame_length > self.time_step:
            start_time = (
                torch.randint(int(frame_length - self.time_step), (1,)) * self.resample
            )
            end_time = start_time + self.time_step * self.resample
            waveform = waveform[:, start_time:end_time]
        else:
            tmp = torch.zeros(waveform.shape[0], self.resample * self.time_step)
            tmp[:, : waveform.shape[1]] = waveform
            waveform = tmp

        if torch.rand(1) > 0.5:
            waveform = waveform + torch.randn(waveform.shape) * 0.1
            shift_idx = torch.randint(16, (1,)) + 1
            waveform = torch.roll(
                waveform, shifts=self.resample // shift_idx.item(), dims=1
            )

        # BYOL-A
        mean_waveform = torch.mean(waveform, dim=0).unsqueeze(0)
        spectrogram = (self.transform_spectrogram(mean_waveform) + EPS).log()
        audio = self.normalizer(spectrogram)
        sequence = {"audio": audio}

        return sequence

    def load_raw_audio(self, index):
        waveform_path = self.audio[index][2]
        no = self.annotation[index][1]

        wav_name = osp.join(waveform_path, str(no).zfill(5) + ".wav")
        waveform, sample_rate = torchaudio.load(wav_name)

        resampler = TA.Resample(sample_rate, self.resample)
        waveform = resampler(waveform)
        frame_length = waveform.shape[1] / self.resample

        if frame_length > self.time_step:
            start_time = int((frame_length - self.time_step) / 2.0)
            end_time = start_time + self.time_step
            waveform = waveform[:, start_time * self.resample : end_time * self.resample]
        else:
            tmp = torch.zeros(waveform.shape[0], self.resample * self.time_step)
            tmp[:, : waveform.shape[1]] = waveform
            waveform = tmp

        # BYOL-A
        mean_waveform = torch.mean(waveform, dim=0).unsqueeze(0)
        spectrogram = (self.transform_spectrogram(mean_waveform) + EPS).log()
        audio = self.normalizer(spectrogram)
        sequence = {"audio": audio}

        return sequence

    def load_aug_frames(self, index, label):
        # Load video stream
        video_dir = self.video[index][2]
        video_index = self.annotation[index][1]
        video_path = osp.join(video_dir, str(video_index).zfill(5) + ".mp4")
        video_object = torchvision.io.VideoReader(video_path, "video")

        # Get metadata
        len_time = video_object.get_metadata()["video"]["duration"][0]
        frame_rate = video_object.get_metadata()["video"]["fps"][0]
        total_frames = int(len_time * frame_rate) + 1

        # Set starting and ending video times
        if int(len_time) > self.time_step:
            start_index = int(
                torch.randint(int(len_time - self.time_step), (1,)) * frame_rate
            )
            end_index = start_index + int(self.time_step * frame_rate) + 1
        else:
            start_index, end_index = 0, total_frames
        start_time, end_time = start_index / frame_rate, end_index / frame_rate
        num_frames = int(end_index - start_index)

        # Sample frames
        video_object.set_current_stream("video")
        video_object = video_object.seek(start_time)
        frame_sequence = torch.empty(0)
        step = int(num_frames / self.num_frame)
        table = torch.arange(0, num_frames, step)
        frame_count = 0
        frames, face_sequence = [], torch.zeros((self.num_frame, 512))
        for frame_index, frame in enumerate(
            itertools.takewhile(lambda x: x["pts"] <= end_time, video_object)
        ):
            if frame_index not in table:
                continue
            frames.append(frame["data"])

            if self.faces:
                img = frame["data"].permute(1, 2, 0).squeeze(0).numpy().astype("uint8")
                img = Image.fromarray(img)
                x_aligned, prob = self.mtcnn(img, return_prob=True)
                if x_aligned is not None:
                    embeddings = self.face_resnet(x_aligned).detach()
                    face_feat = torch.mean(embeddings[:8, :], dim=0).squeeze(0)
                    face_sequence[frame_index : frame_index + 1, :] = face_feat

            frame_count += 1
            if frame_count >= self.num_frame:
                frame_count = 0
                break

        # Apply augmentations
        frame_sequence = self.augment_frames(torch.stack(frames) / 255)
        if torch.rand(1) > 0.5:
            frame_sequence = frame_sequence + torch.randn(frame_sequence.shape) * 0.1
            shift_idx = torch.randint(7, (1,)) + 1
            frame_sequence = torch.roll(frame_sequence, shifts=shift_idx.item(), dims=1)

        if not self.tchw:
            frame_sequence = frame_sequence.permute(1, 0, 2, 3)

        sequence = {"frames": frame_sequence}
        if self.faces:
            sequence["faces"] = face_sequence

        return sequence

    def load_raw_frames(self, index, label):
        # Load video stream
        video_dir = self.video[index][2]
        video_index = self.annotation[index][1]
        video_path = osp.join(video_dir, str(video_index).zfill(5) + ".mp4")
        video_object = torchvision.io.VideoReader(video_path, "video")

        # Get metadata
        len_time = video_object.get_metadata()["video"]["duration"][0]
        frame_rate = video_object.get_metadata()["video"]["fps"][0]
        total_frames = int(len_time * frame_rate) + 1

        # Set starting and ending video times
        if len_time > self.time_step:
            start_index = int((len_time - self.time_step) / 2.0)
            end_index = start_index + int(self.time_step * frame_rate) + 1
        else:
            start_index, end_index = 0, total_frames
        start_time, end_time = start_index / frame_rate, end_index / frame_rate
        num_frames = int(end_index - start_index)

        # Sample frames
        video_object.set_current_stream("video")
        video_object = video_object.seek(start_time)
        frame_sequence = torch.empty(0)
        step = int(num_frames / self.num_frame)
        table = torch.arange(0, num_frames, step)
        frame_count = 0
        frames, face_sequence = [], torch.zeros((self.num_frame, 512))
        for frame_index, frame in enumerate(
            itertools.takewhile(lambda x: x["pts"] <= end_time, video_object)
        ):
            if frame_index not in table:
                continue
            frames.append(frame["data"])

            if self.faces:
                img = frame["data"].permute(1, 2, 0).squeeze(0).numpy().astype("uint8")
                img = Image.fromarray(img)
                x_aligned, prob = self.mtcnn(img, return_prob=True)
                if x_aligned is not None:
                    embeddings = self.face_resnet(x_aligned).detach()
                    face_feat = torch.mean(embeddings[:8, :], dim=0).squeeze(0)
                    face_sequence[frame_index : frame_index + 1, :] = face_feat

            frame_count += 1
            if frame_count >= self.num_frame:
                frame_count = 0
                break

        # Resize and normalize
        frame_sequence = self.resize_frames(torch.stack(frames) / 255)
        if not self.tchw:
            frame_sequence = frame_sequence.permute(1, 0, 2, 3)
        sequence = {"frames": frame_sequence}
        if self.faces:
            sequence["faces"] = face_sequence

        return sequence

    def load_text(self, index):
        sub = self.subtitle[index][2]

        if sub == "":
            sub += " "
        sub_tokens = self.tokenizer.encode_plus(
            text=sub,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=64,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = sub_tokens["input_ids"].squeeze(0)
        attention_mask = sub_tokens["attention_mask"].squeeze(0)
        return {"text_ids": input_ids, "text_masks": attention_mask}

    def __getitem__(self, index):
        # ########## sample ground truth label #####################
        labels = self.annotation[index][2]  # new version
        outputs = {"labels": labels}

        # ############## sample audio data #########################
        if "a" in self.modalities:
            outputs.update(self.load_audio(index))

        # ############## sample video data #########################
        if "v" in self.modalities:
            outputs.update(self.load_frames(index, labels))

        # ############## sample text data #########################
        if "t" in self.modalities:
            outputs.update(self.load_text(index))

        return outputs
