import argparse
import os
import re
from pathlib import Path
import logging
import cv2
import numpy as np
import pandas as pd
import pywt
import tifffile
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from sys import exit

from ramces_cnn import SimpleCNN


class Ramces:
    # Todo: Incorporate tifffile.memmap()

    def __init__(self, model_path=None, channels=None, device="cpu"):
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

        if channels is None:
            channels = []

        self.channels = channels

    def __repr__(self):
        return f"Ramces model, channels={self.channels}, devide={self.device}"

    def preprocess_image(self, im):
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

    def rank_markers(self, im):
        num_markers = im.shape[-1]
        tile_marker_scores = np.zeros(num_markers)
        marker_scores = np.zeros((num_markers, 2))

        with torch.inference_mode():
            for i in range(num_markers):
                im_proc = self.preprocess_image(im[:, :, i])
                output = self.model(
                    im_proc.view(-1, 4, 128, 128).type("torch.FloatTensor")
                )

                marker_scores[i, 0] += output.max()
                marker_scores[i, 1] += 1
                tile_marker_scores[i] = output.max()

        # Todo: Replace this by using num_markers
        marker_scores[:, 0] /= marker_scores[:, 1]

        return marker_scores, tile_marker_scores


def main():
    n_channel = 10
    img = np.zeros((1024, 1024, n_channel))
    for i in range(1, n_channel):
        img[:, :, i] = tifffile.imread(
            f"LHCC35_small/data/LHCC35_0001_t00{i}_c001.tif"
        )
    ram = Ramces(device="mps")
    score, tile = ram.rank_markers(img)
    sorted_idx = np.argsort(score[:, 0])[::-1]
    print(score)


if __name__ == "__main__":
    main()
