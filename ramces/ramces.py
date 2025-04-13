import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
import torch
import torchvision.transforms.functional as TF
from network import SimpleCNN
from tifffile import tifffile
from tqdm import tqdm
from utils import split_into_tiles_padding
import matplotlib.pyplot as plt


class Ramces:
    """
    Class for ranking proteins in a given image.

    Parameters
    ----------
    channels : list
        List of channel names.
    device : str, optional
        Device to use for inference. Defaults to "cpu".

    Attributes
    ----------
    model_path : pathlib.Path
        Path to the trained model.
    model : ramces.network.SimpleCNN
        Trained model.
    device : str
        Device used for inference.
    channels : list
        List of channel names.
    number_channels : int
        Number of channels.
    marker_scores_raw : numpy.ndarray
        Raw marker scores.
    marker_scores : numpy.ndarray
        Normalized marker scores.
    top_markers : numpy.ndarray
        Indices of the top markers.
    number_tiles : int
        Number of tiles processed.

    Methods
    -------
    __repr__()
        Returns a string representation of the class.
    preprocess_image(im)
        Preprocesses an image.
    rank_markers(im)
        Ranks the markers in an image.
    create_pseudochannel(im)
        Creates a pseudochannel from the top markers.
    ranking_table()
        Returns a DataFrame with the marker scores.
    """

    def __init__(self, channels: list, device: str = "cpu") -> None:
        """
        Parameters
        :param channels:
        :param device:

        :type channels: list
        :type device: str

        :raises:

        :return: None
        :rtype: None

        :raises:


        Examples
        --------
        >>> from ramces import Ramces
        >>> ramces = Ramces(["DAPI", "CD3", "CD8", "CD20", "CD68"])
        >>> ramces
        - - - - - -
        Ramces model
        channels: DAPI, CD3, CD8, CD20, CD68,
        device=cpu
        """
        self.model_path = Path("../models/trained_model.h5")
        self.model = SimpleCNN((128, 128))

        try:
            self.device = device
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model = self.model.to(device)
        except RuntimeError:
            logging.warning(
                "No %sdevice detected. Falling back to CPU.", self.device
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
        self.top_markers = None

        self.number_tiles = 0

    def __repr__(self):
        """
        Returns a string representation of the class.
        """
        channel_str = "channels: " + ", ".join(self.channels)
        return (
            f"- - - - - - \nRamces model\n{channel_str},\ndevice={self.device}"
        )

    def preprocess_image(self, im: np.array) -> np.array:
        """
        Preprocesses an image.

        :param im:
        :return:

        :type im: np.array
        :rtype: np.array

        :raises:

        Examples
        --------
        >>> from ramces import Ramces
        >>> ramces = Ramces(["DAPI", "CD3", "CD8", "CD20", "CD68"])
        >>> im = np.random.randint(0, 255, (1024, 1024))
        >>> im_proc = ramces.preprocess_image(im)
        >>> im_proc.shape
        (64, 4, 128, 128)
        """
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

    def preprocess_image_batch(self, im: np.array) -> np.array:
        """
        Preprocesses an image.

        :param im:
        :return:

        :type im: np.array
        :rtype: np.array

        :raises:

        Examples
        --------
        >>> from ramces import Ramces
        >>> ramces = Ramces(["DAPI", "CD3", "CD8", "CD20", "CD68"])
        >>> im = np.random.randint(0, 255, (1024, 1024))
        >>> im_proc = ramces.preprocess_image(im)
        >>> im_proc.shape
        (64, 4, 128, 128)
        """
        # im = cv2.resize(im, dsize=(1024, 1024))

        num_images = im.shape[0]
        im_std = np.std(im)
        im_mean = np.mean(im)

        im = im - im_mean

        if im_std != 0:
            im = im / im_std
            np.clip(im, -3, 3, out=im)

        coeffs = pywt.dwt2(im, "db2")
        ll, lh, hl, hh = coeffs[0], coeffs[1][0], coeffs[1][1], coeffs[1][2]
        im = np.stack([ll, lh, hl, hh], axis=-1)
        im = np.transpose(im, (0, 3, 1, 2))
        im = torch.tensor(im, dtype=torch.float32)
        patches = im.unfold(2, 128, 128).unfold(3, 128, 128)
        im = patches.contiguous().view(num_images * 16, 4, 128, 128)

        return im

    def rank_markers_multi_channel(self, im: np.array) -> None:
        """
        Ranks the markers in an image.
        :param im:
        :return:

        :type im: np.array
        :rtype: None

        :raises:

        Examples
        --------
        >>> from ramces import Ramces
        >>> ramces = Ramces(["DAPI", "CD3", "CD8", "CD20", "CD68"])
        >>> im = np.random.randint(0, 255, (1024, 1024, 5))
        >>> ramces.rank_markers_multi_channel(im)
        """
        with torch.inference_mode():
            for i in tqdm(
                range(self.number_channels),
                desc="Ranking proteins",
                bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}",
                disable=True,
            ):
                im_proc = self.preprocess_image(im[:, :, i])
                output = self.model(
                    im_proc.view(-1, 4, 128, 128).type("torch.FloatTensor")
                )

                self.marker_scores_raw[i] += output.max()

        self.number_tiles += 1
        self.marker_scores = self.marker_scores_raw / self.number_tiles
        self.top_markers = np.argsort(self.marker_scores)[::-1]

    def rank_markers_tensor(self, im: np.array) -> None:
        assert len(im.shape) == 4, "Wrong dimensions!"
        num_images, _, _, num_markers = im.shape

        with torch.inference_mode():
            for marker_idx in range(num_markers):
                for image_idx in range(num_images):
                    im_proc = self.preprocess_image(
                        im[image_idx, :, :, marker_idx]
                    )
                    im_proc = im_proc.type("torch.FloatTensor").to(self.device)
                    output = self.model(im_proc)

                    self.marker_scores_raw[marker_idx] += output.max()
                self.marker_scores[marker_idx] = (
                    self.marker_scores_raw[marker_idx] / num_images
                )
        self.top_markers = np.argsort(self.marker_scores)[::-1]

    def rank_markers_batch(self, im: np.array) -> None:
        assert len(im.shape) == 4, "Wrong dimensions!"
        num_images, _, _, num_markers = im.shape

        with torch.inference_mode():
            for marker_idx in range(num_markers):
                im_proc = self.preprocess_image_batch(im[:, :, :, marker_idx])
                im_proc = im_proc.type("torch.FloatTensor").to(self.device)
                output = self.model(im_proc)

                reshaped_tensor = output.view(num_images, 16)
                max_values, _ = torch.max(reshaped_tensor, 1)
                total_sum = max_values.sum()

                self.marker_scores_raw[marker_idx] += total_sum
                self.marker_scores[marker_idx] = (
                    self.marker_scores_raw[marker_idx] / num_images
                )
        self.top_markers = np.argsort(self.marker_scores)[::-1]

    def create_pseudochannel(
        self, im: np.array, num_weighted: int = 3, mode: str = "ramces"
    ) -> np.array:
        """
        Creates a pseudochannel from the top markers.
        :param im:
        :param num_weighted:
        :return:

        :type im: np.array
        :type num_weighted: int
        :rtype: np.array

        :raises:

        Examples
        --------
        >>> from ramces import Ramces
        >>> ramces = Ramces(["DAPI", "CD3", "CD8", "CD20", "CD68"])
        >>> im = np.random.randint(0, 255, (1024, 1024, 5))
        >>> ramces.rank_markers(im)
        >>> ramces.create_pseudochannel(im).shape
        (1024, 1024)
        """
        if mode == "ramces":
            top_weights = self.marker_scores[self.top_markers[:num_weighted]]
            top_weights /= top_weights.sum()
        elif mode == "uniform":
            top_weights = np.ones(num_weighted) / num_weighted
        # Todo: check this
        top_images = im[:, :, :, self.top_markers[:num_weighted]]

        weighted_im = (top_images * top_weights).sum(axis=-1)
        weighted_im = np.asarray(weighted_im, dtype=im.dtype)

        return weighted_im[0]

    def ranking_table(self) -> pd.DataFrame:
        """
        Returns a DataFrame with the marker scores.
        :return:

        :rtype: pd.DataFrame

        :raises:

        Examples
        --------
        >>> from ramces import Ramces
        >>> ramces = Ramces(["DAPI", "CD3", "CD8", "CD20", "CD68"])
        >>> im = np.random.randint(0, 255, (1024, 1024, 5))
        >>> ramces.rank_markers(im)
        >>> ramces.ranking_table()
                Markers    Scores
        0       DAPI  0.999999
        1        CD3  0.999999
        2        CD8  0.999999
        3       CD20  0.999999
        4       CD68  0.999999
        """
        df = pd.DataFrame(
            {"Markers": self.channels, "Scores": self.marker_scores}
        )

        df = df.sort_values(by="Scores", ascending=False)
        return df


def main():
    """
    stacked_tiles = np.empty((19 * 20, 1024, 1024, len(path_lst)))
    for idx, img_path in enumerate(path_lst):
        img = tifffile.imread(img_path)
        tiles = split_into_tiles_padding(img, (1024, 1024), "reflect")
        tiles = tiles.reshape(tiles.shape[0] * tiles.shape[1], 1024, 1024)
        stacked_tiles[:, :, :, idx] = tiles
    sub_tiles = stacked_tiles[:5]
    """

    path_lst = sorted(
        [
            e
            for e in Path(
                "/Users/matthias/github/segError/data/raw_codex/LHCC35/"
            ).iterdir()
            if e.suffix == ".tif"
        ]
    )

    channels = [str(e).split("_")[4][:-4] for e in path_lst]
    ram_tensor = Ramces(channels=channels, device="mps")

    start_x = 5_000
    start_y = 5_000
    offset = 1024

    full_img = np.zeros((1, 1024, 1024, len(channels)))

    for idx, file in enumerate(path_lst):
        img = tifffile.imread(file, key=0)
        img = img[start_x : start_x + offset, start_y : start_y + offset]
        full_img[0, :, :, idx] = img

    ram = Ramces(channels=channels, device="mps")
    ram.rank_markers_batch(full_img)
    table = ram.ranking_table()
    print(table)
    pseudo_channel = ram.create_pseudochannel(full_img, num_weighted=5, mode="uniform")

    comp_markers = table.head(5).index.to_list()

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax = ax.ravel()

    if False:
        for i in range(len(channels)):
            m = full_img[0, :, :, i].max()
            full_img[0, :, :, i] /= m

    for idx, e in enumerate(comp_markers):
        ax[idx].imshow(full_img[0, :, :, e], cmap="gray")
        ax[idx].axis("off")
        ax[idx].set_title(idx)

    ax[5].imshow(pseudo_channel, cmap="gray")
    ax[5].axis("off")
    ax[5].set_title(5)

    plt.suptitle("Marker comparison")
    plt.savefig("LHCC35_comp.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    exit()

    # image = np.moveaxis(image, 0, -1)

    # image = np.pad(
    #    image, pad_width=((12, 12), (12, 12), (0, 0)), mode="reflect"
    # )

    image = np.expand_dims(image, axis=0)
    channels = [str(i) for i in range(image.shape[-1])]
    """
    rng = np.random.default_rng(0)
    number_channels = 4
    channels = [str(i) for i in range(number_channels)]
    number_tiles = 100
    sub_tiles = rng.integers(
        low=0, high=512, size=(number_tiles, 1024, 1024, number_channels)
    )
    """
    ram_tensor = Ramces(channels=channels, device="mps")
    ram_tensor.rank_markers_batch(image)
    print(ram_tensor.ranking_table())

    pseudo_channel = ram_tensor.create_pseudochannel(image, 3)

    fig, ax = plt.subplots(2, 4, figsize=(16, 9))
    ax = ax.ravel()


if __name__ == "__main__":
    main()


"""
[Finished in 83.5s]
"""
