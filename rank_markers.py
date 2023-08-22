import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
import tifffile
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

from ramces_cnn import SimpleCNN


def preprocessImage(im):
    im = cv2.resize(im, dsize=(1024, 1024))
    # im = np.array(Image.fromarray(im).resize((1024, 1024)))

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


def main():
    parser = argparse.ArgumentParser()

    # Create argument groups
    ranking_args = parser.add_argument_group("Ranking Arguments")
    image_args = parser.add_argument_group("Weighted Image Arguments")

    # Ranking arguments
    ranking_args.add_argument(
        "-m",
        "--model-path",
        help="Path to trained model to rank markers. Does not need to be specified if --no-ranking is set.",
        default="./models/trained_model.h5",
    )
    ranking_args.add_argument(
        "-d",
        "--data-dir",
        help="Path to directory with image data. Each image file should represent a SINGLE channel, with the file name in the required format as detailed on github.com/mdayao/ramces.",
    )
    ranking_args.add_argument(
        "--channels",
        help="Path to file indicating which channels to use and their names. See github.com/mdayao/ramces for details on how to format this file.",
        required=True,
    )
    ranking_args.add_argument(
        "--num-cycles",
        type=int,
        help="The number of cycles present in the data.",
        required=True,
    )
    ranking_args.add_argument(
        "--num-channels-per-cycle",
        type=int,
        help="The number of channels per cycle. For example, if there are 10 cycles with 3 channels each, then we would expect a total of 30 channels.",
        required=True,
    )
    ranking_args.add_argument(
        "--no-ranking",
        help="Set this flag if ranking has already been calculated. If this is set, then --model-path argument does not need to be specified.",
        action="store_true",
    )
    ranking_args.add_argument(
        "-r",
        "--rank-path",
        help="Path to file where marker ranking is to be saved/where marker ranking is saved (if ranking has already been performed). IMPORTANT: the file extension should be '.csv'.",
        required=True,
    )
    ranking_args.add_argument(
        "--gpu",
        action="store_true",
        help="Set if you want to use the GPU.",
    )
    ranking_args.add_argument(
        "--indiv-tiles",
        help="Use this argument if you want to save the scores for each individual tile (this ONLY works with multi-tiff files). Provide the directory where these will be saved. Files will be saved as .npy.",
    )

    # Weighted image arguments
    image_args.add_argument(
        "--create-images",
        help="Set this flag to create weighted images based on the top --num_weighted markers. If this flag is set, the --num-weighted argument must be specified.",
        action="store_true",
    )
    image_args.add_argument(
        "--num-weighted",
        type=int,
        help="Number of top-ranked markers to use to create weighted images.",
        default=3,
    )
    image_args.add_argument(
        "--output-weighted",
        help="Path to directory to output weighted images. Must be specified if the --create-images flag is set.",
    )
    image_args.add_argument(
        "--exclude",
        nargs="*",
        type=int,
        help="Rank of any markers that you wish to exclude from the combined weighted images.",
    )

    args = parser.parse_args()

    if args.create_images and (
        not args.num_weighted or not args.output_weighted
    ):
        parser.error(
            "The --create-images argument requires the --num-weighted and the --output-weighted arguments to be specified."
        )

    # We expect each row of the channel csv file to have Name, Boolean
    ch_boolean = np.loadtxt(args.channels, delimiter=",", dtype=str)
    marker_indices = np.array(
        [i for i, item in enumerate(ch_boolean[:, 1]) if "True" in item]
    )  # which indices out of all the channels to use
    num_markers = len(marker_indices)

    if not args.no_ranking:  # Ranking has not been calculated
        if not args.model_path or not args.data_dir:
            parser.error(
                "Unless the --no-ranking flag is set, we require the --model-path and --data-dir arguments to be specified."
            )

        device = torch.device("cpu")

        if args.gpu:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("mps")
                if torch.backends.mps.is_available()
                else (
                    print("No GPU library available, falling back to CPU."),
                    torch.device("cpu"),
                )[1]
            )

        image_shape = (128, 128)
        model = SimpleCNN(image_shape)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()

        # The score for each marker of interest
        marker_scores = np.zeros((num_markers, 2))

        # image_list = sorted(os.listdir(args.data_dir))
        image_list = sorted(
            [file.name for file in Path(args.data_dir).iterdir()]
        )

        pat_t = re.compile("(?:t)(...)")
        pat_c = re.compile("(?:c)(...)")

        # with tifffile.TiffFile(os.path.join(args.data_dir, image_list[0])) as tif:
        with tifffile.TiffFile(Path(args.data_dir) / image_list[0]) as tif:
            tif_shape = tif.pages[0].shape

        # Each individual tif file is a separate marker image
        if len(tif_shape) == 2:
            # Ranking proteins
            for i, image_file in enumerate(
                tqdm(image_list, desc="Ranking proteins", unit="proteins")
            ):
                t = int(
                    re.findall(pat_t, image_file)[0]
                )  # cycle number, starts from 1
                c = int(
                    re.findall(pat_c, image_file)[0]
                )  # channel number, starts from 1
                marker_idx = (t - 1) * args.num_channels_per_cycle + (
                    c - 1
                )  # which marker index this image refers to
                if marker_idx not in marker_indices:
                    continue
                score_idx = list(marker_indices).index(
                    marker_idx
                )  # which index we need to use for the marker_scores array

                im = tifffile.imread(os.path.join(args.data_dir, image_file))
                im = preprocessImage(im)

                with torch.inference_mode():
                    output = model(
                        im.view(-1, 4, 128, 128).type("torch.FloatTensor")
                    )

                marker_scores[score_idx, 0] += output.max()
                marker_scores[score_idx, 1] += 1

        elif (
            len(tif_shape) == 4
        ):  # Each tif file contains all the cycles and channels for the specific tile
            for i, image_file in enumerate(image_list):
                if args.indiv_tiles is not None:
                    tile_marker_scores = np.zeros(num_markers)
                c_iloc = tif_shape.index(args.num_channels_per_cycle)
                t_iloc = tif_shape.index(args.num_cycles)
                full_im = tifffile.imread(
                    os.path.join(args.data_dir, image_file)
                )
                for t in range(args.num_cycles):
                    for c in range(args.num_channels_per_cycle):
                        marker_idx = (t) * args.num_channels_per_cycle + (c)
                        # which marker index this image refers to
                        if marker_idx not in marker_indices:
                            continue

                        score_idx = list(marker_indices).index(
                            marker_idx
                        )  # which index we need to use for the marker_scores array

                        # Initialize the slice_idx with ellipsis for all dimensions
                        slice_idx = [slice(None)] * len(tif_shape)

                        # Replace specific slices with c and t values
                        slice_idx[c_iloc] = c
                        slice_idx[t_iloc] = t

                        # Convert to tuple and get the desired slice from full_im
                        im = full_im[tuple(slice_idx)]
                        im = preprocessImage(im)

                        with torch.inference_mode():
                            output = model(
                                im.view(-1, 4, 128, 128).type(
                                    "torch.FloatTensor"
                                )
                            )

                        marker_scores[score_idx, 0] += output.max()
                        marker_scores[score_idx, 1] += 1
                        tile_marker_scores[score_idx] = output.max()

                if args.indiv_tiles is not None:
                    output_path = (
                        Path(args.indiv_tiles)
                        / f"{image_file.rsplit('.', 1)[0]}_scores.npy"
                    )
                    np.save(output_path, tile_marker_scores)

        marker_scores[:, 0] /= marker_scores[:, 1]  # averaging over all tiles

        # Output rank and scores
        # sorted indices of the marker_scores, len = num_markers
        sorted_idx = np.argsort(marker_scores[:, 0])[::-1]
        score_dict = {
            "Marker": ch_boolean[marker_indices[sorted_idx], 0],
            "Score": marker_scores[sorted_idx, 0],
            "Cycle": marker_indices[sorted_idx] // args.num_channels_per_cycle
            + 1,
            "Channel": marker_indices[sorted_idx] % args.num_channels_per_cycle
            + 1,
        }
        score_df = pd.DataFrame(data=score_dict)
        score_df.to_csv(args.rank_path, index=False)

    else:
        # Ranking is already calculated
        score_df = pd.read_csv(args.rank_path)
        reverse_indices = np.array(
            [
                list(ch_boolean[:, 0]).index(i)
                for i in score_df["Marker"].values
            ]
        )
        sorted_idx = np.argsort(np.argsort(reverse_indices))

    print(score_df)

    if args.create_images:
        print(
            f"\nCreating weighted images based on the top {args.num_weighted} markers"
        )

        top_weights = np.zeros((args.num_weighted, 3))

        if args.exclude is not None:
            sorted_idx = np.delete(sorted_idx, np.array(args.exclude) - 1)
        for i, idx in enumerate(
            marker_indices[sorted_idx[: args.num_weighted]]
        ):
            cyc_num = (idx // args.num_channels_per_cycle) + 1
            ch_num = (idx % args.num_channels_per_cycle) + 1
            top_weights[i, :] = np.array(
                [score_df.iloc[i, 1], cyc_num, ch_num]
            )

        image_list = sorted(
            [file.name for file in Path(args.data_dir).iterdir()]
        )

        pat_t = re.compile("(?:t)(...)")
        pat_c = re.compile("(?:c)(...)")

        with tifffile.TiffFile(
            os.path.join(args.data_dir, image_list[0])
        ) as tif:
            tif_shape = tif.pages[0].shape

        # Each individual tif file is a separate marker image
        if len(tif_shape) == 2:
            req_pat = re.compile(
                "(t\d{3}.c\d{3}|c\d{3}.t\d{3})", flags=re.IGNORECASE
            )
            tc_pat = re.findall(req_pat, image_list[0])[0]
            # _, tif_ext = os.path.splitext(image_list[0])

            # Todo: Why are we doing this instead of just hardcoding ".tif"?
            tif_ext = Path(image_list[0]).suffix
            tile_ids = []

            for i, image_file in enumerate(image_list):
                tile_ids.append(
                    "".join(re.sub(req_pat, "", image_file).split(".")[:-1])
                )

            output_weighted_path = Path(args.output_weighted)
            if not output_weighted_path.exists():
                output_weighted_path.mkdir(parents=True, exist_ok=True)

            tile_ids = list(set(tile_ids))
            for tile_id in tqdm(
                tile_ids, desc="Saving weighted images", unit="images"
            ):
                paths = []
                for marker_i in range(args.num_weighted):
                    t = int(top_weights[marker_i, 1])
                    c = int(top_weights[marker_i, 2])
                    tc_pat = re.sub(r"(?:t)(\d{3})", f"t{t:03d}", tc_pat)
                    tc_pat = re.sub(r"(?:c)(\d{3})", f"c{c:03d}", tc_pat)

                    paths.append(f"{tile_id}{tc_pat}{tif_ext}")

                ims = [
                    tifffile.imread(os.path.join(args.data_dir, path))
                    for path in paths
                ]

                """
                weighted_num = np.sum(
                    np.array(
                        [
                            top_weights[i, 0] * ims[i]
                            for i in range(args.num_weighted)
                        ]
                    ),
                    0,
                )
                weighted_norm = np.sum(top_weights[:, 0])

                weighted_im = weighted_num / weighted_norm
                weighted_im = np.asarray(weighted_im, dtype=ims[0].dtype)
                """

                weighted_num = (
                    top_weights[: args.num_weighted, 0, None, None]
                    * ims[: args.num_weighted]
                ).sum(axis=0)
                weighted_norm = top_weights[: args.num_weighted, 0].sum()
                weighted_im = weighted_num / weighted_norm
                weighted_im = weighted_im.astype(ims[0].dtype)

                weighted_path = os.path.join(
                    args.output_weighted, f"{tile_id}weighted{tif_ext}"
                )

                with tifffile.TiffWriter(weighted_path) as tif:
                    tif.write(weighted_im)

        elif (
            len(tif_shape) == 4
        ):  # Each tif file contains all the cycles and channels for the specific tile
            c_iloc = tif_shape.index(args.num_channels_per_cycle)
            t_iloc = tif_shape.index(args.num_cycles)
            tif_ext = Path(image_list[0]).suffix
            for i, image_file in enumerate(
                tqdm(image_list, desc="Ranking proteins", unit="proteins")
            ):
                tile_id = image_file.split(".")[0]
                full_im = tifffile.imread(
                    os.path.join(args.data_dir, image_file)
                )
                indiv_ims = [None] * args.num_weighted
                for marker_i in range(args.num_weighted):
                    t = int(top_weights[marker_i, 1])
                    c = int(top_weights[marker_i, 2])
                    slice_idx = [..., ..., ..., ...]
                    slice_idx[c_iloc] = c - 1
                    slice_idx[t_iloc] = t - 1
                    slice_idx.remove(...)
                    slice_idx = tuple(slice_idx)
                    indiv_ims[marker_i] = full_im[slice_idx]

                weighted_num = np.sum(
                    np.array(
                        [
                            top_weights[i, 0] * indiv_ims[i]
                            for i in range(args.num_weighted)
                        ]
                    ),
                    0,
                )
                weighted_norm = np.sum(top_weights[:, 0])

                weighted_im = weighted_num / weighted_norm
                weighted_im = np.asarray(weighted_im, dtype=indiv_ims[0].dtype)

                weighted_path = os.path.join(
                    args.output_weighted, f"{tile_id}weighted{tif_ext}"
                )

                with tifffile.TiffWriter(weighted_path) as tif:
                    tif.write(weighted_im)

        print(f"\nWeighted images saved to {args.output_weighted}")


if __name__ == "__main__":
    main()
