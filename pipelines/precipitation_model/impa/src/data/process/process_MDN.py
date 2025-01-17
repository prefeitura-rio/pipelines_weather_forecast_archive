import argparse
import datetime
import gzip
import pathlib
from functools import partial
from multiprocessing.pool import Pool

import h5py
import numpy as np
from tqdm import tqdm

from src.data.process.RadarData import RadarData
from src.utils.data_utils import NRAYS, VARIABLES_DICT
from src.utils.general_utils import print_error, print_warning


def prj0(x, y):
    return x


def prj1(x, y):
    return y


def partition_consecutive_mod_n(l1: list, n: int):
    if n <= 0:
        raise ValueError("Integer n must be positive.")

    if len(l1) == 0:
        raise ValueError("List is empty.")

    l2 = sorted(l1)

    partition_dict = {l2[0]: [l2[0]]}

    last_partition_representative = l2[0]
    for k in l2[1:]:
        if k >= n:
            raise ValueError("There is a number in list that is greater or equal than n.")

        if (k - 1) in partition_dict[last_partition_representative]:
            partition_dict[last_partition_representative].append(k)
        else:
            partition_dict[k] = [k]
            last_partition_representative = k

    if l2[0] == 0 and n - 1 in partition_dict[last_partition_representative]:
        partition_dict[last_partition_representative].extend(partition_dict[l2[0]])
        del partition_dict[l2[0]]

    return partition_dict


def get_azimuth_indices(sweep_info: dict) -> dict:
    indices_dict = dict()

    for primary_key in sweep_info.keys():
        startazA = sweep_info[primary_key]["startazA"]
        stopazA = sweep_info[primary_key]["stopazA"]

        # remove nan values
        nan_idx = np.isnan(startazA)
        assert np.all(
            nan_idx == np.isnan(stopazA)
        ), "nan values in startazA and stopazA do not match."
        startazA = startazA[~nan_idx]
        stopazA = stopazA[~nan_idx]

        pos_dist = ((stopazA - startazA) % 360) / 2
        neg_dist = 180 - pos_dist

        dist = np.minimum(pos_dist, neg_dist)

        avgazA = (startazA + (2 * (pos_dist < neg_dist) - 1) * dist) % 360
        hist, bins = np.histogram(avgazA, bins=np.arange(0, NRAYS + 1, 1))
        indices_dict[primary_key] = dict()
        indices_dict[primary_key]["repeated_indices"] = dict()

        repeated_indices = np.asarray(hist > 1).nonzero()[0]
        for i in repeated_indices:
            indices_dict[primary_key]["repeated_indices"][i] = []
            for j, az in enumerate(avgazA):
                if bins[i] <= az < bins[i + 1]:
                    indices_dict[primary_key]["repeated_indices"][i].append(j)

        indices_dict[primary_key]["missing_indices"] = np.asarray(hist == 0).nonzero()[0]
        indices_dict[primary_key]["indices"] = np.digitize(avgazA, bins) - 1

    return indices_dict


def process_data(full_matrix: np.array, operator) -> dict:
    processed_data, index_matrix = operator(full_matrix)

    return processed_data, index_matrix


# flake8: noqa
def task_filepath(filepath, verbose=False, overwrite=False, feature="DBZH", process_type="CMAX"):
    print(f"Processing {filepath}")
    is_gz = filepath.suffix == ".gz"
    if is_gz:
        f = gzip.open(filepath, "rb")
        hdf = h5py.File(f, "r")

        lat = hdf["where"].attrs["lat"]
        lon = hdf["where"].attrs["lon"]

        file_date = hdf["what"].attrs["date"].decode("UTF-8")
        date = datetime.datetime.strptime(file_date, "%Y%m%d").strftime("%Y-%m-%d")
        time = hdf["what"].attrs["time"].decode("UTF-8")
    else:
        raise NotImplementedError("Only gzipped files are supported.")

    file_stem = f"MDN-{process_type}-{feature}-{file_date}-{time}"
    output_filepath = pathlib.Path(
        f"pipelines/precipitation_model/impa/data/processed/processed_PPI_MDN/{process_type}/{feature}/{date}/{file_stem}.hdf"
    )
    pathlib.Path(output_filepath).parents[0].mkdir(parents=True, exist_ok=True)
    if output_filepath.is_file():
        if overwrite:
            print_warning(f"Warning: overwriting existing file {output_filepath}", verbose=verbose)
        else:
            print_warning(f"Warning: skipping existing file {output_filepath}", verbose=verbose)
            return output_filepath

    gains = []
    offsets = []

    secondary_key = VARIABLES_DICT[feature]

    primary_keys = [key for key in hdf.keys() if key not in ["how", "what", "where"]]
    sweep_info = dict([(primary_key, dict()) for primary_key in primary_keys])

    rscales = []
    rstarts = []
    for primary_key in primary_keys:
        sweep_info[primary_key]["nrays"] = hdf[primary_key]["where"].attrs["nrays"]
        sweep_info[primary_key]["nbins"] = hdf[primary_key]["where"].attrs["nbins"]

        rscale = hdf[primary_key]["where"].attrs["rscale"]
        rstart = hdf[primary_key]["where"].attrs["rstart"]

        rscales.append(rscale)
        rstarts.append(rstart)

        sweep_info[primary_key]["startazA"] = np.array(hdf[primary_key]["how"].attrs["startazA"])
        sweep_info[primary_key]["stopazA"] = np.array(hdf[primary_key]["how"].attrs["stopazA"])
        gains.append(hdf[primary_key][secondary_key]["what"].attrs["gain"])
        offsets.append(hdf[primary_key][secondary_key]["what"].attrs["offset"])

    try:
        assert rscales.count(rscales[0]) == len(rscales), "Not all rscales are identical."
        assert rstarts.count(rstarts[0]) == len(rstarts), "Not all rstarts are identical."
    except AssertionError as e:
        print_error(str(e) + "\nSkipping datetime.", verbose)
        return None
    assert gains.count(gains[0]) == len(gains), "Not all gains are identical."
    assert offsets.count(offsets[0]) == len(offsets), "Not all offsets are identical."
    gain = gains[0]
    offset = offsets[0]

    assert gain > 0, "gain variable less or equal than zero."
    del gains
    del offsets

    nrays_max = max([sweep_info[primary_key]["nrays"] for primary_key in sweep_info.keys()])
    nbins_max = max([sweep_info[primary_key]["nbins"] for primary_key in sweep_info.keys()])
    try:
        assert nrays_max == NRAYS, f"Maximum value of nrays over elevations is not {NRAYS}."
    except AssertionError as e:
        print_error(str(e) + "\nSkipping datetime.", verbose)
        return None

    try:
        assert (
            nbins_max == 1000
        ), f"Maximum value of nbins over elevations is not 1000 in file {filepath}."
    except AssertionError as e:
        print_warning(e, verbose)
    indices_dict = get_azimuth_indices(sweep_info)

    full_matrix = -np.ones((nrays_max, nbins_max, len(primary_keys)))

    full_startazA_matrix = -np.ones((nrays_max, nbins_max, len(primary_keys)))
    full_stopazA_matrix = -np.ones((nrays_max, nbins_max, len(primary_keys)))
    full_startazT_matrix = -np.ones((nrays_max, nbins_max, len(primary_keys)))
    full_stopazT_matrix = -np.ones((nrays_max, nbins_max, len(primary_keys)))
    full_elevation_matrix = np.empty((nrays_max, nbins_max, len(primary_keys)))

    for el_index, primary_key in enumerate(primary_keys):
        if len(indices_dict[primary_key]["missing_indices"]) > 0:
            print_warning(f"Warning: missing indices in {filepath} at {primary_key}.", verbose)
        if len(indices_dict[primary_key]["repeated_indices"].keys()) > 0:
            print_warning(f"Warning: repeated indices in {filepath} at {primary_key}.", verbose)

        data_array = np.array(hdf[primary_key][secondary_key]["data"])

        for i in indices_dict[primary_key]["repeated_indices"].keys():
            repeated_list = indices_dict[primary_key]["repeated_indices"][i]

            nonzero_repeated_list = repeated_list

            zero_line_indices = np.where(np.all(data_array[repeated_list, :] == 0, axis=1))[0]

            succeding_ray_index = (i + 1) % NRAYS
            while succeding_ray_index in indices_dict[primary_key]["missing_indices"]:
                succeding_ray_index = (succeding_ray_index + 1) % NRAYS
            preceding_ray_index = (i - 1) % NRAYS
            while preceding_ray_index in indices_dict[primary_key]["missing_indices"]:
                preceding_ray_index = (preceding_ray_index - 1) % NRAYS

            # Warn when a whole repeated line is zero. If this is the case, take the mean of non-zero repeated lines
            if len(zero_line_indices):
                print_warning(
                    f"Warning: Data array at one of the repeated indices in {filepath} at {primary_key} is all zeros.",
                    verbose,
                )
                nonzero_repeated_list = list(
                    set(repeated_list) - set([repeated_list[ind] for ind in zero_line_indices])
                )

            nonzero_repeated_list += [
                list(indices_dict[primary_key]["indices"]).index(preceding_ray_index),
                list(indices_dict[primary_key]["indices"]).index(succeding_ray_index),
            ]

            data_array[repeated_list, :] = np.median(
                data_array[list(nonzero_repeated_list), :], axis=0
            ).reshape((1, data_array.shape[1]))

        startazA = np.array(hdf[primary_key]["how"].attrs["startazA"])
        stopazA = np.array(hdf[primary_key]["how"].attrs["stopazA"])
        startazT = np.array(hdf[primary_key]["how"].attrs["startazT"])
        stopazT = np.array(hdf[primary_key]["how"].attrs["stopazT"])
        assert np.all(np.isnan(startazA) == np.isnan(startazT)) and np.all(
            np.isnan(stopazT) == np.isnan(stopazA)
        ), "startazA, stopazA, startazT and stopazT do not all match."
        nan_az_idx = np.isnan(startazA)
        startazA = startazA[~nan_az_idx]
        stopazA = stopazA[~nan_az_idx]
        startazT = startazT[~nan_az_idx]
        stopazT = stopazT[~nan_az_idx]
        full_matrix[
            indices_dict[primary_key]["indices"],
            : hdf[primary_key]["where"].attrs["nbins"],
            el_index,
        ] = data_array[~nan_az_idx]

        full_startazA_matrix[indices_dict[primary_key]["indices"], :, el_index] = startazA.reshape(
            startazA.shape[0], 1
        )
        full_stopazA_matrix[indices_dict[primary_key]["indices"], :, el_index] = stopazA.reshape(
            stopazA.shape[0], 1
        )
        full_startazT_matrix[indices_dict[primary_key]["indices"], :, el_index] = startazT.reshape(
            startazT.shape[0], 1
        )
        full_stopazT_matrix[indices_dict[primary_key]["indices"], :, el_index] = stopazT.reshape(
            stopazT.shape[0], 1
        )

        full_elevation_matrix[:, :, el_index] = hdf[primary_key]["where"].attrs["elangle"]

        missing_indices_partition = None
        try:
            missing_indices_partition = partition_consecutive_mod_n(
                indices_dict[primary_key]["missing_indices"], NRAYS
            )
        except ValueError:
            missing_indices_partition = {}

        for class_representative in missing_indices_partition.keys():
            m = len(missing_indices_partition[class_representative])
            preceding_ray_index = (class_representative - 1) % NRAYS
            succeding_ray_index = (missing_indices_partition[class_representative][-1] + 1) % NRAYS

            for i, ind in enumerate(missing_indices_partition[class_representative]):
                full_matrix[ind, :, el_index] = np.average(
                    full_matrix[[preceding_ray_index, succeding_ray_index], :, el_index],
                    axis=0,
                    weights=[(i + 1) / (m + 1), 1 - (i + 1) / (m + 1)],
                ).reshape((1, nbins_max))

    ind0 = np.fromfunction(prj0, (nrays_max, nbins_max)).astype(int)
    ind1 = np.fromfunction(prj1, (nrays_max, nbins_max)).astype(int)

    operator = None
    if process_type == "CMAX":

        def operator(full_matrix):
            index_matrix = np.argmax(full_matrix, axis=2)

            def to_list(x):
                return [x]

            return full_matrix[ind0, ind1, index_matrix], np.vectorize(to_list, otypes=["object"])(
                index_matrix
            )

    else:
        raise NotImplementedError("process_type not implemented yet.")

    processed_data, index_matrix = process_data(full_matrix, operator)
    index_permutations = [(n % nrays_max, int(n / nrays_max)) for n in range(nrays_max * nbins_max)]

    def get_from_indices(data, indices):
        new_data = np.empty((nrays_max, nbins_max), dtype="object")
        for i, j in index_permutations:
            new_data[i, j] = data[i, j, indices[i, j]]
        return new_data

    startazA_matrix = get_from_indices(full_startazA_matrix, index_matrix)
    stopazA_matrix = get_from_indices(full_stopazA_matrix, index_matrix)
    startazT_matrix = get_from_indices(full_startazT_matrix, index_matrix)
    stopazT_matrix = get_from_indices(full_stopazT_matrix, index_matrix)
    elevation_matrix = get_from_indices(full_elevation_matrix, index_matrix)

    radar_data = RadarData(
        processed_data,
        process_type,
        gain,
        offset,
        feature,
        nrays_max,
        nbins_max,
        index_matrix,
        startazA_matrix,
        stopazA_matrix,
        startazT_matrix,
        stopazT_matrix,
        elevation_matrix,
        file_date,
        time,
        lat,
        lon,
        rscale=rscale,
        rstart=rstart,
    )
    try:
        radar_data.save_hdf(output_filepath)
    except OSError as e:
        if output_filepath.is_file():
            return None
        else:
            raise e

    f.close()
    hdf.close()

    return output_filepath


def process_radar(
    verbose=False, overwrite=False, feature="DBZH", process_type="CMAX", num_workers=1
):
    if feature not in VARIABLES_DICT.keys():
        print_error(f"Error: Specified feature {feature} not allowed.")
        exit()

    possible_process_types = ["CMAX"]

    if process_type not in possible_process_types:
        print_error(f"Error: Specified process type {process_type} not allowed.")
        exit()

    with open("pipelines/precipitation_model/impa/data/raw/radar_PPI_MDN/downloaded_files.log", "r") as f:
        filepaths = [pathlib.Path(line.strip()) for line in f]

    task_filepath_partial = partial(
        task_filepath,
        verbose=verbose,
        overwrite=overwrite,
        feature=feature,
        process_type=process_type,
    )

    with Pool(num_workers) as pool:
        output_filepaths = list(
            tqdm(pool.imap(task_filepath_partial, filepaths), total=len(filepaths))
        )

    # log output filepaths
    with open("pipelines/precipitation_model/impa/data/processed/processed_PPI_MDN/processed_files.log", "w") as f:
        for output_filepath in output_filepaths:
            if output_filepath is not None:
                f.write(f"{output_filepath}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="If true, overwrites output; otherwise, skips existing files.",
    )
    parser.add_argument(
        "--feature",
        "-f",
        default="DBZH",
        help="Feature to be processed (DBZH, DBZV, ...)",
        type=str,
    )
    parser.add_argument(
        "--process_type",
        "-t",
        default="CMAX",
        help="Type of processing (CMAX, PSEUDO-CAPPI, ...)",
        type=str,
    )
    parser.add_argument(
        "--num_workers",
        "-n",
        type=int,
        default=1,
        help="Number of processes for parallelization.",
    )
    args = parser.parse_args()
    process_radar(**vars(args))
