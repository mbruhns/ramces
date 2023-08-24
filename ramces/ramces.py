import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .ramces_cnn import SimpleCNN


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
    model : ramces.ramces_cnn.SimpleCNN
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

    def rank_markers(self, im: np.array) -> None:
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
        >>> ramces.rank_markers(im)
        """
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

    def create_pseudochannel(
        self, im: np.array, num_weighted: int = 3
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
        top_weights = self.marker_scores[self.top_markers[:num_weighted]]
        top_images = im[:, :, self.top_markers[:num_weighted]]

        top_weights /= top_weights.sum()
        weighted_im = (top_images * top_weights).sum(axis=-1)
        weighted_im = np.asarray(weighted_im, dtype=im.dtype)

        return weighted_im

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
