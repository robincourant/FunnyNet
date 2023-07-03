from collections import Counter, defaultdict
import os
import os.path as osp
from typing import Dict, List, Tuple

import auditok
import numpy as np
from sklearn.cluster import KMeans
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from laughter_detection.core.audio_embedding import AudioEmbedder


class LaughterDetector:
    """Detect all laugthers within an audio track."""

    def __init__(
        self,
        embedding_name: str,
        root_dir: str,
        byol_dir: str = "ext/byol_a",
        n_clusters: int = 3,
        batch_size: int = 20,
        num_workers: int = 1,
        num_gpus: int = 1,
        verbose: bool = False,
    ):
        self.root_dir = root_dir

        # Directory with stereo audio tracks
        self.raw_dir = osp.join(root_dir, "audio", "raw")

        # Directory with difference of stereo audio tracks
        self.diff_dir = osp.join(root_dir, "audio", "diff")
        if not osp.exists(self.diff_dir):
            os.makedirs(self.diff_dir)
        # Directory with left channel of surround audio tracks
        self.left_dir = osp.join(root_dir, "audio", "left")
        if not osp.exists(self.left_dir):
            os.makedirs(self.left_dir)
        # Directory with audi embedding vectors
        self.embedding_dir = osp.join(root_dir, "audio", "embedding", embedding_name)
        if not osp.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)

        # Minimum duration of a valid audio event in seconds
        self.min_dur = 0.8
        # Maximum duration of an event
        self.max_dur = 11
        # Maximum duration of continuous silence in an event
        self.max_silence = 0.1
        # Time offset to add before and after the detected segment
        self.offset = 0.6
        # Detection threshold for stereo audio tracks
        self.stereo_detection_threshold = 57
        # Detection threshold for surround audio tracks
        self.surround_detection_threshold = 45
        # Number of clusters (audio embedding)
        self.n_clusters = n_clusters

        # Initilaize the audio embedder
        self.audio_embedder = AudioEmbedder(
            model_name=embedding_name,
            byol_dir=byol_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            num_gpus=num_gpus,
            verbose=verbose,
        )

    def _save_stereodiff(
        self, raw_track: torch.Tensor, sample_rate: float, diff_path: str
    ):
        """Save the difference between stereo channels."""
        diff_track = raw_track[0] - raw_track[1]
        sf.write(diff_path, diff_track.numpy(), sample_rate)

        self.sample_rate = sample_rate

    def _save_surroundleft(
        self, raw_track: torch.Tensor, sample_rate: float, left_path: str
    ):
        """Save only the left channel containing sound effects."""
        left_channel = raw_track[0]
        sf.write(left_path, left_channel.numpy(), sample_rate)

        self.sample_rate = sample_rate

    def _detect_nonsilent(
        self, diff_path: str, detection_threshold: int
    ) -> List[Tuple[float, float]]:
        """Detect non-silent audio segments within an audio track."""
        nonsilent_segments = auditok.split(
            input=diff_path,
            min_dur=self.min_dur,
            max_dur=self.max_dur,
            max_silence=self.max_silence,
            energy_threshold=detection_threshold,
        )

        # Get each segment's timecodes and enlarge it with an offset
        nonsilent_timecodes = sorted(
            [
                [r.meta.start - self.offset, r.meta.end + self.offset]
                for r in nonsilent_segments
            ]
        )

        return nonsilent_timecodes

    def _get_nonsilent(self, audio_filename: str):
        """Get non-silent segment tiomecodes of the given audio file."""
        # Load raw audio tracks
        raw_path = osp.join(self.raw_dir, audio_filename)
        raw_track, sample_rate = torchaudio.load(raw_path)
        n_channels = raw_track.shape[0]

        # Compute and save the difference between stereo channels
        if n_channels == 2:
            diff_path = osp.join(self.diff_dir, audio_filename[:-4] + ".wav")
            self._save_stereodiff(raw_track, sample_rate, diff_path)
            # Detect non-silent regions
            nonsilent_timecodes = self._detect_nonsilent(
                diff_path, self.stereo_detection_threshold
            )

        # Get the left channel of surround track (5.1) with sound effects
        if n_channels == 6:
            left_path = osp.join(self.left_dir, audio_filename[:-4] + ".wav")
            self._save_surroundleft(raw_track, sample_rate, left_path)
            # Detect non-silent regions
            nonsilent_timecodes = self._detect_nonsilent(
                left_path, self.surround_detection_threshold
            )

        return nonsilent_timecodes

    def _load_segments(
        self,
        segment_timecodes: List[Tuple[float, float]],
        audio_filename: str,
    ) -> List[torch.Tensor]:
        """Load and extract audio segments within given timecodes."""
        raw_segments = []
        for start_timecode, end_timecode in segment_timecodes:
            start_index = max(int(self.sample_rate * start_timecode), 0)
            duration = int(self.sample_rate * (end_timecode - start_timecode))

            raw_segment, _ = torchaudio.load(
                osp.join(self.raw_dir, audio_filename),
                frame_offset=start_index,
                num_frames=duration,
            )
            raw_segments.append(raw_segment[0][None])

        return raw_segments

    def _get_embeddings(
        self, audio_dir
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]], List[str]]:
        """Get embedding vectors for all audio files in `audio_dir`.

        :param audio_dir: path to the director with audio files to compute.
        :return: embedding vector, corresponding non-silent timecodes and
            filenames.
        """
        audio_embeddings, nonsilent_timecodes, episode_filenames = [], [], []
        for audio_filename in tqdm(os.listdir(audio_dir)):
            # Detect non-silent timecodes
            current_nonsilent = self._get_nonsilent(audio_filename)
            current_filenames = [audio_filename for _ in current_nonsilent]
            nonsilent_timecodes.extend(current_nonsilent)
            episode_filenames.extend(current_filenames)

            # Check if embeddings are already computed
            embedding_filename = osp.join(
                self.embedding_dir, audio_filename[:-4] + ".pt"
            )
            if osp.exists(embedding_filename):
                current_embedding = torch.load(embedding_filename)
                audio_embeddings.append(current_embedding)
                continue

            # Load and extract all detected non-silent regions
            raw_segments = self._load_segments(current_nonsilent, audio_filename)

            # Compute and save audio embedding for all detected segments
            current_embedding = self.audio_embedder.get_audioembeddings(
                raw_segments, self.sample_rate, embedding_filename
            )
            audio_embeddings.append(current_embedding)

        audio_embeddings = torch.vstack(audio_embeddings)

        return audio_embeddings, nonsilent_timecodes, episode_filenames

    @staticmethod
    def _merge_segments(
        segments: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Merge segments sharing a common part."""
        index, lenght = 0, len(segments)
        while (lenght > 1) and (lenght - index > 1):
            # Check if the two consecutive segment share a part
            if max(segments[index]) >= min(segments[index + 1]):
                new_segment = [min(segments[index]), max(segments[index + 1])]
                # Add the merged segment and revove the originals
                segments.pop(index)
                segments.insert(index, new_segment)
                segments.pop(index + 1)

            else:
                index += 1
            lenght = len(segments)

        return segments

    def __detect_laughters(self, audio_filename: str) -> List[Tuple[float, float]]:
        """[DEPRECATED] Detect laughters within a given stereo audio track.

        :param audio_filename: name of the audio track in the stereo directory.
        :param detection_threshold: threshold of detection.
        :return: detected laughter timecodes.
        """
        # Load raw audio tracks
        raw_path = osp.join(self.raw_dir, audio_filename)
        raw_track, sample_rate = torchaudio.load(raw_path)
        n_channels = raw_track.shape[0]

        # Compute and save the difference between stereo channels
        if n_channels == 2:
            diff_path = osp.join(self.diff_dir, audio_filename)
            self._save_stereodiff(raw_track, sample_rate, diff_path)
            # Detect non-silent regions
            nonsilent_timecodes = self._detect_nonsilent(
                diff_path, self.stereo_detection_threshold
            )

        # Get the left channel of surround track (5.1) with sound effects
        if n_channels == 6:
            left_path = osp.join(self.left_dir, audio_filename)
            self._save_surroundleft(raw_track, sample_rate, left_path)
            # Detect non-silent regions
            nonsilent_timecodes = self._detect_nonsilent(
                left_path, self.surround_detection_threshold
            )

        # Load and extract all detected non-silent regions
        raw_segments = self._load_segments(nonsilent_timecodes, audio_filename)

        # Compute audio embedding for all detected segments
        audio_embeddings = self.audio_embedder.get_audioembeddings(
            raw_segments, self.sample_rate
        )

        # Cluster embeddings with k-means
        k_means = KMeans(n_clusters=self.n_clusters)
        cluster_results = k_means.fit_predict(audio_embeddings)

        # Music cluster is the less populated one
        cluster_counts = Counter(cluster_results)
        music_cluster = min(cluster_counts, key=cluster_counts.get)

        # Remove music cluster from detected segments
        (music_indices,) = np.where(cluster_results == music_cluster)
        laughter_timecodes = []
        for k, timecode in enumerate(nonsilent_timecodes):
            if k in music_indices:
                continue
            laughter_timecodes.append(timecode)

        # Merge segments sharing a common part
        laughter_timecodes = self._merge_segments(laughter_timecodes)

        return laughter_timecodes

    def detect_laughters(self) -> Dict[str, List[Tuple[float, float]]]:
        """Detect laughters within a given stereo audio track."""
        # Get audio embedding vectors of all files in `self.raw_dir`
        (
            audio_embeddings,
            nonsilent_timecodes,
            episode_filenames,
        ) = self._get_embeddings(self.raw_dir)
        # Cluster embeddings with k-means
        k_means = KMeans(n_clusters=self.n_clusters)
        cluster_results = k_means.fit_predict(audio_embeddings)

        # Music cluster is the less populated one
        cluster_counts = Counter(cluster_results)
        music_cluster = (
            min(cluster_counts, key=cluster_counts.get) if self.n_clusters != 1 else -1
        )

        # Remove music cluster from detected segments
        (music_indices,) = np.where(cluster_results == music_cluster)
        laughter_timecodes = defaultdict(list)
        n_detections = len(nonsilent_timecodes)
        for detection_index in range(n_detections):
            if detection_index in music_indices:
                continue
            timecode = nonsilent_timecodes[detection_index]
            filename = episode_filenames[detection_index]
            laughter_timecodes[filename].append(timecode)

        # Merge segments sharing a common part
        for filename, timecodes in laughter_timecodes.items():
            merged_timecodes = self._merge_segments(timecodes)
            laughter_timecodes[filename] = merged_timecodes

        return dict(laughter_timecodes)
