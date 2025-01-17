import pathlib
from multiprocessing import Pool

import tqdm

from src.data.process.RadarData import RadarData


def despeckle(filepath):
    new_filepath = pathlib.Path(str(filepath).replace("CMAX", "d2CMAX"))
    new_filepath.parents[0].mkdir(parents=True, exist_ok=True)
    if new_filepath.is_file():
        return
    rd = RadarData.load_hdf(filepath)

    d_rd = rd.despeckle(1, 10, 0.7, 100)
    d_rd.save_hdf(new_filepath)


def despeckle_radar_data(radar_filepaths: list, num_workers: int = 1):
    with Pool(num_workers) as pool:
        list(tqdm.tqdm(pool.imap(despeckle, radar_filepaths), total=len(radar_filepaths)))
