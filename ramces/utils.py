from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
from tifffile import tifffile


def split_into_tiles_padding(image, tile_size, padding_mode) -> np.array:
    """
    Function to split an image into tiles of a given size. If the image does not divide evenly into tiles, the image is
    padded. The padding mode can be specified. Options are zeros, reflect and circular.
    :param image:
    :param tile_size:
    :return:
    """

    padding_height = (
        0
        if image.shape[0] % tile_size[0] == 0
        else tile_size[0] - image.shape[0] % tile_size[0]
    )
    padding_width = (
        0
        if image.shape[1] % tile_size[1] == 0
        else tile_size[1] - image.shape[1] % tile_size[1]
    )
    padding_tuple = (
        (padding_height // 2, (padding_height + 1) // 2),
        (padding_width // 2, (padding_width + 1) // 2),
    )
    image = np.pad(image, padding_tuple, padding_mode)

    # Get the shape of the image
    image_shape = image.shape

    # Get the number of tiles in each dimension
    num_tiles = [
        image_shape[0] // tile_size[0],
        image_shape[1] // tile_size[1],
    ]

    # Create an empty array to store the tiles
    tiles = np.zeros((num_tiles[0], num_tiles[1], tile_size[0], tile_size[1]))

    # Loop over the tiles and add them to the array
    for i in range(num_tiles[0]):
        for j in range(num_tiles[1]):
            tiles[i, j, :, :] = image[
                i * tile_size[0] : (i + 1) * tile_size[0],
                j * tile_size[1] : (j + 1) * tile_size[1],
            ]

    return tiles


def load_tif(file_path):
    try:
        return tifffile.imread(file_path)
    except RuntimeError:
        return tifffile.imread(file_path, key=0)


def load_tif_to_array(array, file_path, index):
    array[index] = tifffile.imread(file_path, key=0)


def main():
    file_paths = sorted(
        [
            e
            for e in Path(
                "/Users/matthias/github/segError/data/raw_codex/LHCC35"
            ).iterdir()
            if e.suffix == ".tif"
        ]
    )

    file_paths = file_paths[:8]

    image_shape = tifffile.imread(file_paths[0], key=0).shape
    result_array = np.empty((len(file_paths),) + image_shape)

    images = []
    parallel = False

    if parallel:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    load_tif_to_array, result_array, file_path, index
                )
                for index, file_path in enumerate(file_paths)
            ]
            for future in futures:
                future.result()

    else:
        for idx, img_path in enumerate(file_paths):
            obj = load_tif(img_path)
            print("Single size", obj.nbytes / 1024**2)
            images.append(obj)

    print(result_array.nbytes)


if __name__ == "__main__":
    main()
