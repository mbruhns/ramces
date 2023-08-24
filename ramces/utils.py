from pathlib import Path

import numpy as np
from tifffile import tifffile


class DataLoader:
    """
    Class for lazily loading and tiling an image.
    """

    def __init__(self, path: Path = None):
        self.tile_len = 1024
        self.path = path

    def __repr__(self):
        return "DataLoader"

    def tile_indicex(self):
        pass

    def generate_tiles(self):
        pass


def main():
    rng = np.random.default_rng(42)
    full_image = rng.integers(low=0, high=1025, size=(2048, 2048))

    dl = DataLoader()


if __name__ == "__main__":
    main()
