import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pywt
import tifffile
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from ramces_cnn import SimpleCNN
import pandas as pd


class Ramces:
    def __init__(
        self, channels: list, model_path: Path = None, device: str = "cpu"
    ):
        self.model_path = Path("models/trained_model.h5")

        self.model = SimpleCNN((128, 128))

        try:
            self.device = device
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        except RuntimeError:
            logging.warning(
                f"No {self.device} device detected. Falling back to CPU."
            )
            self.device = "cpu"
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )

        self.model = self.model.eval()

        self.channels = channels
        self.number_channels = len(self.channels)

        self.marker_scores_raw = np.zeros(len(self.channels))
        self.marker_scores = np.zeros(len(self.channels))

        self.number_tiles = 0

    def __repr__(self):
        channel_str = "channels: " + ", ".join(self.channels)
        return (
            f"- - - - - - \nRamces model\n{channel_str},\ndevice={self.device}"
        )

    def preprocess_image(self, im: np.array):
        im = cv2.resize(im, dsize=(1024, 1024))

        im_std = np.std(im)
        im_mean = np.mean(im)

        im = im - im_mean

        if im_std != 0:
            im = im / im_std
            np.clip(im, -3, 3, out=im)

        # Wavelet decomposition and recombination
        coeffs = pywt.dwt2(im, "db2")
        ll, lh, hl, hh = coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]
        im = np.stack([ll, lh, hl, hh], axis=-1)

        # Convert to tensor and extract patches
        im = TF.to_tensor(np.float64(im))
        patches = im.unfold(1, 128, 128).unfold(2, 128, 128).transpose(0, 2)
        im = patches.contiguous().view(-1, 4, 128, 128)

        return im

    def rank_markers(self, im: np.array):
        num_markers = im.shape[-1]
        assert num_markers == self.number_channels

        with torch.inference_mode():
            for i in tqdm(
                range(self.number_channels),
                desc="Ranking proteins",
                bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}",
            ):
                im_proc = self.preprocess_image(im[:, :, i])
                output = self.model(
                    im_proc.view(-1, 4, 128, 128).type("torch.FloatTensor")
                )

                self.marker_scores_raw[i] += output.max()

        self.number_tiles += 1
        self.marker_scores = self.marker_scores_raw / self.number_tiles
        self.top_markers = np.argsort(self.marker_scores)[::-1]

    def create_pseudochannel(self, im: np.array, num_weighted: int = 3):
        top_weights = self.marker_scores[self.top_markers[:num_weighted]]
        top_images = im[:, :, self.top_markers[:num_weighted]]

        top_weights /= top_weights.sum()
        weighted_im = (top_images * top_weights).sum(axis=-1)
        weighted_im = np.asarray(weighted_im, dtype=im.dtype)

        return weighted_im

    def ranking_table(self):
        df = pd.DataFrame(
            {"Markers": self.channels, "Scores": self.marker_scores}
        )

        df = df.sort_values(by="Scores", ascending=False)
        return df


def main():
    directory_path = "demo/data/"
    tif_files = [f for f in os.listdir(directory_path) if f.endswith(".tif")]
    tif_files.sort()

    channels = pd.read_csv("demo/channels.csv", header=None)[0].to_list()

    images = [
        tifffile.imread(os.path.join(directory_path, f)) for f in tif_files
    ]
    img = np.stack(images, axis=-1)

    ram = Ramces(device="mps", channels=channels)
    print(ram)
    # Testing
    ram.rank_markers(img)
    ram.create_pseudochannel(img)


if __name__ == "__main__":
    main()
