import pathlib

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
import scipy.ndimage
from scipy import interpolate
from scipy.signal import convolve2d

from src.utils.data_utils import NRAYS

RSCALE = 250.0
RSTART = 0.0

MAP_CENTER = {"lat": -22.9932804107666, "lon": -43.26795928955078}
REF_LATLON = (-22.9932804107666, -43.58795928955078)
LOG_SCALE_VARIABLES = ["TH", "TV", "DBZH", "DBZV", "ZDR"]


def flat_concat(v: np.array):
    if v is None:
        return h5py.Empty("f")
    flat_v = v.flatten()
    flat_concat_v = []
    for i in range(len(flat_v)):
        try:
            flat_concat_v += list(flat_v[i])
        except TypeError:
            flat_concat_v += [flat_v[i]]
    return flat_concat_v


def invert_flat_concat(v: np.array, list_lengths_cumsum: np.array, nrays: int, nbins: int):
    if v.shape is None:
        return None
    v_split = np.array(np.split(v, list_lengths_cumsum), dtype="object")
    return v_split.reshape(nrays, nbins)


class RadarData:
    def __init__(
        self,
        data: np.array,
        process_type: str = None,
        gain: float = None,
        offset: float = None,
        feature: str = None,
        nrays: int = 360,
        nbins: int = 1000,
        indices: np.array = None,
        startazA: np.array = None,
        stopazA: np.array = None,
        startazT: np.array = None,
        stopazT: np.array = None,
        elevations: np.array = None,
        date: str = None,
        time: str = None,
        lat: float = REF_LATLON[0],
        lon: float = REF_LATLON[1],
        compressed: bool = True,
        rscale: float = RSCALE,
        rstart: float = RSTART,
    ):
        self.data = data
        self.process_type = process_type
        self.gain = gain
        self.offset = offset
        self.feature = feature

        assert nrays == NRAYS, f"nrays should be {NRAYS}"
        self.nrays = nrays
        self.nbins = nbins
        self.indices = indices
        self.startazA = startazA
        self.stopazA = stopazA
        self.startazT = startazT
        self.stopazT = stopazT
        self.elevations = elevations
        self.date = date
        self.time = time
        self.lat = lat
        self.lon = lon
        self.compressed = compressed
        self.rstart = rstart
        self.rscale = rscale

    @classmethod
    def load_hdf(cls, input_filepath: pathlib.Path):
        with h5py.File(input_filepath, "r") as f:
            if f["list_lengths"].shape is None:
                list_lengths_cumsum = None
            else:
                list_lengths_cumsum = np.cumsum(f["list_lengths"])[:-1]
            nrays = f.attrs["nrays"]
            nbins = f.attrs["nbins"]
            try:
                rscale = f.attrs["rscale"]
            except KeyError:
                rscale = RSCALE
            try:
                rstart = f.attrs["rstart"]
            except KeyError:
                rstart = RSTART
            indices = invert_flat_concat(f["indices"], list_lengths_cumsum, nrays, nbins)
            startazA = invert_flat_concat(f["startazA"], list_lengths_cumsum, nrays, nbins)
            stopazA = invert_flat_concat(f["stopazA"], list_lengths_cumsum, nrays, nbins)
            startazT = invert_flat_concat(f["startazT"], list_lengths_cumsum, nrays, nbins)
            stopazT = invert_flat_concat(f["stopazT"], list_lengths_cumsum, nrays, nbins)
            elevations = invert_flat_concat(f["elevations"], list_lengths_cumsum, nrays, nbins)

            if "compressed" in f.attrs.keys():
                compressed = f.attrs["compressed"]
            else:
                compressed = True

            return cls(
                np.array(f["dataset"]),
                f.attrs["process_type"],
                f.attrs["gain"],
                f.attrs["offset"],
                f.attrs["feature"],
                nrays,
                nbins,
                indices,
                startazA,
                stopazA,
                startazT,
                stopazT,
                elevations,
                f.attrs["date"],
                f.attrs["time"],
                f.attrs["lat"],
                f.attrs["lon"],
                compressed,
                rscale,
                rstart,
            )

    def save_hdf(self, output_filepath: pathlib.Path):
        flat_indices = flat_concat(self.indices)
        flat_startazA = flat_concat(self.startazA)
        flat_stopazA = flat_concat(self.stopazA)
        flat_startazT = flat_concat(self.startazT)
        flat_stopazT = flat_concat(self.stopazT)
        flat_elevations = flat_concat(self.elevations)
        if self.indices is None:
            list_lengths = h5py.Empty("f")
        else:
            try:
                list_lengths = np.vectorize(len)(self.indices.flatten())
            except TypeError:
                list_lengths = np.ones(self.indices.flatten().shape, dtype=np.int8)

        with h5py.File(output_filepath, "w") as f:
            f.create_dataset("dataset", data=self.data, compression="gzip")
            try:
                f.create_dataset("indices", data=flat_indices, compression="gzip")
                f.create_dataset("startazA", data=flat_startazA, compression="gzip")
                f.create_dataset("stopazA", data=flat_stopazA, compression="gzip")
                f.create_dataset("startazT", data=flat_startazT, compression="gzip")
                f.create_dataset("stopazT", data=flat_stopazT, compression="gzip")
                f.create_dataset("elevations", data=flat_elevations, compression="gzip")
                f.create_dataset("list_lengths", data=list_lengths, compression="gzip")
            except TypeError:
                f.create_dataset("indices", data=flat_indices)
                f.create_dataset("startazA", data=flat_startazA)
                f.create_dataset("stopazA", data=flat_stopazA)
                f.create_dataset("startazT", data=flat_startazT)
                f.create_dataset("stopazT", data=flat_stopazT)
                f.create_dataset("elevations", data=flat_elevations)
                f.create_dataset("list_lengths", data=list_lengths)
            f.attrs["process_type"] = self.process_type
            f.attrs["gain"] = self.gain
            f.attrs["offset"] = self.offset
            f.attrs["feature"] = self.feature
            f.attrs["nrays"] = self.nrays
            f.attrs["nbins"] = self.nbins
            f.attrs["date"] = self.date
            f.attrs["time"] = self.time
            f.attrs["lat"] = self.lat
            f.attrs["lon"] = self.lon
            f.attrs["compressed"] = self.compressed
            f.attrs["rscale"] = self.rscale
            f.attrs["rstart"] = self.rstart

    def unwrap_data(self):
        if not self.compressed:
            raise ValueError("Radar data is already unwrapped.")

        real_data = np.array(self.data)
        if self.offset != 0 and not self.feature == "VRAD":
            real_data[real_data <= 0] = np.nan
        real_data = real_data * self.gain + self.offset

        if self.feature in LOG_SCALE_VARIABLES:
            real_data = 10 ** (real_data / 10)
            real_data = np.nan_to_num(real_data)

        return RadarData(
            real_data,
            self.process_type,
            self.gain,
            self.offset,
            self.feature,
            self.nrays,
            self.nbins,
            self.indices,
            self.startazA,
            self.stopazA,
            self.startazT,
            self.stopazT,
            self.elevations,
            self.date,
            self.time,
            self.lat,
            self.lon,
            False,
            self.rscale,
            self.rstart,
        )

    def compress_data(self):
        if self.compressed:
            raise ValueError("Radar data is already compressed.")

        compressed_data = np.array(self.data)

        if self.feature in LOG_SCALE_VARIABLES:
            compressed_data = 10 * np.log10(compressed_data)
            compressed_data = np.where(np.isinf(compressed_data), np.nan, compressed_data)
        compressed_data[np.isnan(compressed_data)] = self.offset
        compressed_data = (compressed_data - self.offset) / self.gain

        return RadarData(
            compressed_data,
            self.process_type,
            self.gain,
            self.offset,
            self.feature,
            self.nrays,
            self.nbins,
            self.indices,
            self.startazA,
            self.stopazA,
            self.startazT,
            self.stopazT,
            self.elevations,
            self.date,
            self.time,
            self.lat,
            self.lon,
            True,
            self.rscale,
            self.rstart,
        )

    def write_image(
        self,
        output_filepath: pathlib.Path = None,
        lower_bound: float = 0,
        zoom: float = 6.3,
        return_trace: bool = False,
        map_center: dict[str, float] = MAP_CENTER,
        interactive: bool = False,
        marker_dict: dict[str, any] | None = None,
    ):
        if marker_dict is None:
            marker_dict = dict(
                cmin=0, cmax=60, colorscale="rainbow", opacity=0.3, colorbar={"title": self.feature}
            )
        # Geod is used to transform initial position, angle and radius into a new lat/lon position
        geodesic = pyproj.Geod(ellps="WGS84")

        # RSTART and RSCALE are information given by the original hdf
        radii = (self.rstart + self.rscale / 2) + np.arange(0, self.nbins) * self.rscale
        azimuths = range(NRAYS)

        data = np.empty((self.nrays, self.nbins))
        # Original data is compressed in integer format, we have to convert to the real measurements using gain and offset
        if self.compressed:
            data = self.unwrap_data().data
        else:
            data = self.data
        if self.feature in LOG_SCALE_VARIABLES:
            data = 10 * np.log10(data)
        df = []

        for i, azimuth in enumerate(azimuths):
            for j, radius in enumerate(radii):
                data_point = data[i, j]
                # Check if values are correct and plot only points above lower_bound
                if np.isnan(data_point) or data_point < lower_bound:
                    continue
                # Calculate end position using initial lat/lon position, angle and radius values.
                lon, lat = geodesic.fwd(self.lon, self.lat, azimuth, radius)[:2]
                df.append((lat, lon, data_point))
        df = pd.DataFrame(df, columns=["latitude", "longitude", self.feature])

        trace = go.Scattermapbox(
            lat=df.latitude,
            lon=df.longitude,
            mode="markers",
            marker={**marker_dict, "color": df[self.feature]},
            hovertemplate="Latitude: %{lat:.3f}<br>"
            + "Longitude: %{lon:.3f}<br>"
            + f"{self.feature}: %{{marker.color:,}}"
            + "<extra></extra>",
        )
        if return_trace:
            return trace

        fig = go.Figure(trace)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=600, hovermode="x unified")
        fig.update_mapboxes(center=map_center, zoom=zoom, style="open-street-map")

        if output_filepath is not None:
            if interactive:
                fig.write_html(output_filepath)
            else:
                fig.write_image(output_filepath)
        return fig

    def find_objects(self, lower_threshold: float, smooth_window: int = None):
        if self.compressed:
            data = self.unwrap_data().data
        else:
            data = self.data
        if smooth_window is not None:
            data = convolve2d(
                data,
                np.ones((1, smooth_window)) / float(smooth_window),
                mode="same",
                boundary="symm",
            )
        cond = data < lower_threshold
        binary_data = np.where(cond, 0, 1)

        binary_data = np.append(binary_data, binary_data, axis=0)
        adj = np.ones((3, 3), dtype="int16")
        labels, _ = scipy.ndimage.label(binary_data, structure=adj)
        i1 = 0
        i2 = labels.shape[0] // 2
        old_labs = np.unique(labels[i2][labels[i2] > 0])
        for lab in old_labs:
            indices = np.where(labels[i2] == lab)
            new_lab = np.unique(labels[i1][indices[0]])[0]
            labels[labels == lab] = new_lab
        labels = labels[0:i2]

        return labels

    def despeckle(
        self,
        lower_threshold: float = 10.0,
        min_size=10,
        min_area_per_ratio: float = 0.5,
        max_variation=0.1,
        smooth_window: int = None,
        fillvalue: float = 0.0,
    ):
        new_rd = self.unwrap_data()
        labels = new_rd.find_objects(lower_threshold, smooth_window)
        new_data = np.where(new_rd.data < lower_threshold, fillvalue, new_rd.data)

        variation_matrix = np.array([[0, -1 / 4, 0], [-1 / 4, 1, -1 / 4], [0, -1 / 4, 0]])
        variation = np.abs(convolve2d(new_data, variation_matrix, mode="same", boundary="fill"))
        relative_variation = variation / new_data
        unique_labels = np.unique(labels[labels > 0])
        for lab in unique_labels:
            cond = labels == lab
            filtered_labels = np.where(cond, 1, 0)

            # Calculate size of block
            obj_size = np.size(labels[cond])

            # Calculate area/perimeter ratio of block
            adj_matrix = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            neighbors = convolve2d(filtered_labels, adj_matrix, mode="same", boundary="fill")
            n_neighbors = np.sum(np.where(cond, 0, neighbors))
            ratio = obj_size / n_neighbors

            # Calculate average variation over block
            average_variation = np.sum(np.where(cond, relative_variation, 0)) / obj_size

            if (
                obj_size < min_size
                or ratio < min_area_per_ratio
                or average_variation > max_variation
            ):
                new_data[cond] = fillvalue

        return RadarData(
            new_data,
            self.process_type,
            self.gain,
            self.offset,
            self.feature,
            self.nrays,
            self.nbins,
            self.indices,
            self.startazA,
            self.stopazA,
            self.startazT,
            self.stopazT,
            self.elevations,
            self.date,
            self.time,
            self.lat,
            self.lon,
            False,
            self.rscale,
            self.rstart,
        )

    def despeckle_time(
        self,
        rds: list,
        lower_threshold: float = 10.0,
        # inflate_radius: int = 1,
        min_intersections: int = 1,
        smooth_window: int = None,
        fillvalue: float = 0.0,
    ):
        new_rd = self.unwrap_data()
        labels = new_rd.find_objects(lower_threshold, smooth_window)
        for rd in rds:
            assert rd.nbins == self.nbins
            assert rd.nrays == self.nrays
            assert rd.rscale == self.rscale
            assert rd.rstart == self.rstart
            assert rd.lat == self.lat
            assert rd.lon == self.lon

        other_labels = [rd.unwrap_data().find_objects(lower_threshold) for rd in rds]

        intersections_per_pixel = np.zeros_like(labels)
        for other_label in other_labels:
            other_label = np.where(other_label > 0, 1, 0)
            intersections_per_pixel += labels * other_label
        intersections_per_pixel = np.floor_divide(intersections_per_pixel, labels)

        new_data = np.where(new_rd.data < lower_threshold, 0.0, new_rd.data)

        unique_labels = np.unique(labels[labels > 0])
        for lab in unique_labels:
            cond = labels == lab
            intersections = intersections_per_pixel[cond]
            if np.max(intersections) < min_intersections:
                new_data[cond] = fillvalue

        return RadarData(
            new_data,
            self.process_type,
            self.gain,
            self.offset,
            self.feature,
            self.nrays,
            self.nbins,
            self.indices,
            self.startazA,
            self.stopazA,
            self.startazT,
            self.stopazT,
            self.elevations,
            self.date,
            self.time,
            self.lat,
            self.lon,
            False,
            self.rscale,
            self.rstart,
        )

    def get_radar_grid(self):
        geodesic = pyproj.Geod(ellps="WGS84")

        radii = (self.rstart + self.rscale / 2) + np.arange(0, self.nbins) * self.rscale
        azimuths = np.array(range(NRAYS))

        azimuths_mat = np.ones((len(azimuths), len(radii))) * azimuths.reshape(-1, 1)
        radii_mat = np.ones((len(azimuths), len(radii))) * radii.reshape(1, -1)

        lat_mat = np.ones((len(azimuths), len(radii))) * self.lat
        lon_mat = np.ones((len(azimuths), len(radii))) * self.lon

        latlons = geodesic.fwd(lon_mat, lat_mat, azimuths_mat, radii_mat)[:2]

        grid = np.dstack([latlons[1], latlons[0]])

        np.save(
            f"pipelines/precipitation_model/data/dataframe_grids/radar_grid-rstart={self.rstart}-rscale={self.rscale}-nbins={self.nbins}.npy",
            grid,
        )
        return np.dstack([latlons[1], latlons[0]])

    def interp_at_grid(self, target_grid):
        assert not self.compressed
        if self.rscale == RSCALE and self.rstart == RSTART and self.nbins >= 500:
            polar_grid = np.load("data/dataframe_grids/radar_grid.npy")
        else:
            try:
                polar_grid = np.load(
                    f"data/dataframe_grids/radar_grid-rstart={self.rstart}-rscale={self.rscale}-nbins={self.nbins}.npy"
                )
            except FileNotFoundError:
                polar_grid = self.get_radar_grid()
        x = polar_grid[:, :, 1].flatten()
        y = polar_grid[:, :, 0].flatten()
        points = np.stack((x, y)).T
        values = self.data[:, : polar_grid.shape[1]].flatten()
        shape = target_grid.shape[:2]
        return interpolate.griddata(
            points,
            values,
            (target_grid[:, :, 1].flatten(), target_grid[:, :, 0].flatten()),
            method="linear",
        ).reshape(shape)
