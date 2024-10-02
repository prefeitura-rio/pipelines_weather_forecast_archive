# -*- coding: utf-8 -*-

"""
rodar local:
python integrator.py --sources InmetWS AlertaRioRG --period "2024-02-02" "2024-02-03"
"""
import argparse
import os
from datetime import datetime

import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# from utils.datalake import DataLakeWrapper
from utils.date_time import is_valid_date, today

# from dotenv import load_dotenv
from utils.logging import Logger

INSTRUMENTS = ["AlertaRioRG", "AlertaRioWS", "InmetWS", "IneaRadar"]
FEATURE_HANDLERS = ["NoneHandler", "ZeroHandler", "ZeroWithFlagHandler"]


# class DataLakeHandler:
#     """Class responsible for collecting and storing processed data in the data lake
#     """
#     def __init__(self) -> None:
#         self._dl = DataLakeWrapper()


#     def read_parquet(self, remote_path, columns=None, filters=None):
#         ddf = self._dl.read_parquet_from_lake(
#             remote_path,
#             columns=columns,
#             filters=filters
#         )
#         return ddf


#     def store_data_in_lake(self, table:pa.Table, path):
#         pq.write_table(
#             table,
#             where=f's3://{path}',
#             filesystem=self._dl.get_filesystem(),
#         )


class Instrument:

    # def __init__(self, data_lake_handler: DataLakeHandler) -> None:
    def __init__(
        self,
    ) -> None:
        self._logger = Logger.get_logger()
        self._data = None
        self._remote_path = None
        self._institution = None
        self._instrument = None
        # self._dl = data_lake_handler
        self._features = []

    def prepare_data(self):
        self._load_data()
        self._add_source_columns()

    def get_data(self):
        self._logger.info(f"Inside get_data")
        # self._logger.info(f"\n{self._data.iloc[0]}")
        return self._data

    def get_features(self):
        return self._features

    def get_non_features(self):
        return [col for col in self._data.columns if col not in self._features]

    def _load_data(self):
        start_date, end_date = self._compute_period()
        # df = self._dl.read_parquet(
        #     self._remote_path,
        #     filters=[
        #         ('year', '>=', start_date['year']),
        #         ('year', '<=', end_date['year']),
        #     ],
        # ).compute()
        self._logger.info(f"Starting loading data")
        print(f"Starting loading data")
        df = pd.read_csv(self._remote_path)

        df = df[df["year"] >= int(start_date.split("-")[0])]
        df = df[df["year"] <= int(end_date.split("-")[0])]

        start_date = pd.to_datetime(start_date).tz_localize("UTC")
        end_date = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1)
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Filtrar o DataFrame pelo intervalo de datas
        df = df[(df["datetime"] >= start_date) & (df["datetime"] < end_date)]
        self._logger.info(f"Finished loading data")
        print(f"Finished loading data")
        # self._logger.info(f"pós filtro\n{df.iloc[0]}")
        self._data = df.drop(columns=["year", "month"])

    # def _compute_period(self):
    #     period = args.period
    #     year, month, day = split_date(parse_to_date(period[0]))
    #     start_date = {"year": year, "month": month, "day": day}
    #     if len(period) == 2:
    #         year, month, day = split_date(parse_to_date(period[1]))
    #     else:
    #         year, month, day = split_date(today())
    #     end_date = {"year": year, "month": month, "day": day}

    #     print(f"period {start_date, end_date}")
    #     return (start_date, end_date)
    def _compute_period(self):
        period = args.period
        start_date = period[0]
        if len(period) == 2:
            end_date = period[1]
        else:
            end_date = today()

        print(f"period {start_date, end_date}")
        return (start_date, end_date)

    def _add_source_columns(self):
        self._logger.info(f"Start adding source columns")
        self._data["institution"] = "_".join(self._institution.lower().split())
        self._data["instrument"] = "_".join(self._instrument.lower().split())
        self._logger.info(f"Finished adding source columns")


class AlertaRioRG(Instrument):

    # def __init__(self, data_lake_handler: DataLakeHandler) -> None:
    #     super().__init__(data_lake_handler)
    def __init__(
        self,
    ) -> None:
        super().__init__()
        # self._remote_path = os.path.join('curated', 'rain_gauge', 'alertario')
        self._remote_path = os.path.join("curated", "rain_gauge_optimized.csv")
        self._institution = "Alerta Rio"
        self._instrument = "Rain Gauge"
        self.name = f"{self._institution} - {self._instrument}"
        self._features = ["precipitation", "hour_sin", "hour_cos", "month_sin", "month_cos"]

    def prepare_data(self):
        super().prepare_data()
        if "InmetWS" in args.sources or args.sources == ["all"]:
            self._aggregate_data()

    def _aggregate_data(self):
        df = self._data.copy()
        df["datetime"] = df["datetime"].dt.ceil("H")
        df["precipitation"] = df["precipitation"].fillna(0)
        df = (
            df.groupby(
                ["institution", "instrument", "station", "datetime"]
            )  # TODO: change transform to save station
            .agg(
                {
                    "precipitation": "sum",
                    "hour_sin": "mean",
                    "hour_cos": "mean",
                    "month_sin": "mean",
                    "month_cos": "mean",
                    "latitude": "first",
                    "longitude": "first",
                }
            )
            .reset_index()
        )
        self._data = df


# class AlertaRioWS(Instrument):

#     def __init__(self, data_lake_handler: DataLakeHandler) -> None:
#         super().__init__(data_lake_handler)
#         self._remote_path = os.path.join('curated', 'weather_station', 'alertario')
#         self._institution = 'Alerta Rio'
#         self._instrument = 'Weather Station'
#         self.name = f'{self._institution} - {self._instrument}'
#         self._features = [
#             'precipitation',
#             'wind_dir',
#             'wind_speed',
#             'temperature',
#             'pressure',
#             'humidity',
#             'wind_u',
#             'wind_v',
#             'hour_sin',
#             'hour_cos',
#             'month_sin',
#             'month_cos'
#         ]


#     def prepare_data(self):
#         super().prepare_data()
#         if 'InmetWS' in args.sources or args.sources == ['all']:
#             self._aggregate_data()


#     def _aggregate_data(self):
#         df = self._data.copy()
#         df['datetime'] = df['datetime'].dt.ceil('H')
#         df = df.groupby(['institution', 'instrument', 'station', 'datetime']) \
#             .agg({
#                 'precipitation': 'sum',
#                 'wind_dir': 'mean',
#                 'wind_speed': 'mean',
#                 'temperature': 'mean',
#                 'pressure': 'mean',
#                 'humidity': 'mean',
#                 'wind_u': 'mean',
#                 'wind_v': 'mean',
#                 'hour_sin': 'mean',
#                 'hour_cos': 'mean',
#                 'month_sin': 'mean',
#                 'month_cos': 'mean',
#                 'latitude': 'first',
#                 'longitude': 'first',
#             }) \
#             .reset_index()
#         self._data = df


class InmetWS(Instrument):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._remote_path = os.path.join("curated", "weather_station_optimized.csv")
        self._institution = "INMET"
        self._instrument = "Weather Station"
        self.name = f"{self._institution} - {self._instrument}"
        self._features = [
            "precipitation",
            "pressure",
            "temperature",
            "dew_point",
            "humidity",
            "wind_dir",
            "wind_speed",
            "wind_u",
            "wind_v",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
        ]


# class IneaRadar(Instrument):

#     def __init__(self, data_lake_handler: DataLakeHandler) -> None:
#         super().__init__(data_lake_handler)
#         self._remote_path = os.path.join('curated', 'radar', 'inea')
#         self._institution = 'INEA'
#         self._instrument = 'Radar'
#         self.name = f'{self._institution} - {self._instrument}'
#         self._features = [
#             'horizontal_reflectivity_mean',
#             'hour_sin',
#             'hour_cos',
#             'month_sin',
#             'month_cos',
#         ]


#     def _load_data(self):
#         super()._load_data()
#         self._data = self._data.drop(columns=['day'])


class ConstantHandler:
    @staticmethod
    def handle_features(df: pd.DataFrame, all_features: set, constant_value=None):
        columns = set(df.columns)
        features_to_handle = sorted(all_features.difference(columns))
        for feature in features_to_handle:
            df[feature] = constant_value
        return df


class NoneHandlerStrategy:
    @staticmethod
    def handle_features(df: pd.DataFrame, all_features: set):
        return ConstantHandler.handle_features(df, all_features, constant_value=None)


class ZeroHandlerStrategy:
    @staticmethod
    def handle_features(df: pd.DataFrame, all_features: set):
        return ConstantHandler.handle_features(df, all_features, constant_value=0)


class ZeroWithFlagHandlerStrategy:
    @staticmethod
    def handle_features(df: pd.DataFrame, all_features: set):
        columns = set(df.columns)
        existing_features = sorted(all_features.intersection(columns))
        for feature in existing_features:
            df[f"{feature}_flag"] = 1
        features_to_handle = sorted(all_features.difference(columns))
        for feature in features_to_handle:
            df[feature] = 0
            df[f"{feature}_flag"] = 0
        return df


class DataIntegrator:
    def __init__(self) -> None:
        self._logger = Logger.get_logger()
        # self._dl = DataLakeHandler()
        self._feature_handler = self._get_feature_handler(args.non_shared_feature_handler)
        self._datasets = []
        self._all_features = list()
        self._all_non_features = list()
        self._integrated_data = None
        self._all_metadata = self._build_all_metadata()
        self._sources = INSTRUMENTS if args.sources == ["all"] else args.sources

    def run(self):
        self._logger.info("Script initialized")
        self._prepare_data()
        self._integrate_data()
        self._save_data()
        self._logger.info("Script finished")

    def _prepare_data(self):
        for string_source in self._sources:
            source = eval(string_source)()
            # source = eval(string_source)(self._dl) ########
            try:
                self._logger.info(f"Preparing data from source {source.name}...")
                source.prepare_data()
                self._logger.info(f"\n{source.get_data().iloc[0]}")
                self._datasets.append(source.get_data())
                self._all_features.append(source.get_features())
                self._all_non_features.append(source.get_non_features())
                message = f"Finished preparing data from source {source.name}."
                self._logger.info(message)
            except Exception as error:
                message = f"Error while preparing data from source {source.name}. \
Source will be ignored and data integration will continue."
                self._logger.error(message)
                self._logger.error(error)
                continue

    def _integrate_data(self):
        self._logger.info("Integrating data sources...")
        if len(self._datasets) > 1:
            non_shared_features = self._get_non_shared_elements(self._all_features)
            self._logger.debug(f"Non shared features: {non_shared_features}")
            non_shared_non_features = self._get_non_shared_elements(self._all_non_features)
            self._logger.debug(f"Non shared non features: {non_shared_non_features}")
            columns_order = None
            self._integrated_data = pd.DataFrame(data=None)
            for _ in range(len(self._datasets)):
                df = self._datasets.pop(0)
                df = self._feature_handler.handle_features(df, non_shared_features)
                df = self._handle_non_features(df, non_shared_non_features)
                if columns_order is None:
                    columns_order = df.columns
                else:
                    df = df[columns_order]
                self._integrated_data = pd.concat([self._integrated_data, df], ignore_index=True)
                del df
            self._integrated_data = self._integrated_data.reset_index(drop=True)
        else:
            self._integrated_data = self._datasets[0]
            self._datasets = None

    def _get_non_shared_elements(self, all_elements: list):
        """Returns a set with all elements that are not common to all lists

        Args:
            all_elements (list): A list of lists

        Returns:
            set: a set with all elements that are not common to all lists
        """
        all_elements = list(map(set, all_elements))
        shared_elements = all_elements[0].intersection(*all_elements)
        non_shared_elements = set()
        for s in all_elements:
            non_shared_elements.update(s.difference(shared_elements))
        return non_shared_elements

    def _get_feature_handler(self, strategy):
        try:
            if not args.non_shared_feature_handler:
                return NoneHandlerStrategy
            handler_strategy = eval(f"{strategy}Strategy")
            return handler_strategy
        except NameError:
            message = f"Class {strategy}Strategy not implemented."
            self._logger.error(message)
            exit()
        except Exception as e:
            message = f"Error while instantiating handler strategy"
            self.logger.error(message)
            self._logger.error(e)
            exit()

    def _handle_non_features(self, df: pd.DataFrame, non_shared_non_features: set):
        columns = set(df.columns)
        non_features_to_handle = sorted(non_shared_non_features.difference(columns))
        for non_feature in non_features_to_handle:
            df[non_feature] = None
        return df

    def _build_all_metadata(self):
        non_features = {
            "institution": "institution (name of the authority owning the data)",
            "instrument": "instrument (type of meteorological instrument)",
            "station": "station (station name)",
            "station_id": "station_id (station UID)",
            "datetime": "datetime (UTC datetime of the measurement)",
            "latitude": "latitude (degrees)",
            "longitude": "longitude (degrees)",
            "altitude": "altitude (kilometers)",
        }
        features = {
            "horizontal_reflectivity_mean": "horizontal_reflectivity_mean (mean reflectivity in dBZ)",
            "dew_point": "dew_point (dew point temperature in Celsius)",
            "humidity": "humidity (instant relative humidity in %)",
            "pressure": "pressure (instant atmospheric pressure in mB)",
            "temperature": "temperature (instant temperature in Celsius)",
            "wind_dir": "wind_dir (clockwise wind direction in degrees)",
            "wind_speed": "wind_speed (hourly wind speed in m/s)",
            "wind_u": "wind_u (cyclic U component from the wind)",
            "wind_v": "wind_v (cyclic V component from the wind)",
            "precipitation": "precipitation (hourly precipitation in mm)",
            "hour_sin": "hour_sin (sine encoding of the time of day)",
            "hour_cos": "hour_cos (cosine encoding of the time of day)",
            "month_sin": "month_sin (sine encoding of the month of year)",
            "month_cos": "month_cos (cosine encoding of the month of year)",
        }
        return {"features": features, "non_features": non_features}

    def _save_data(self):
        self._logger.info("Preparing data to be stored...")
        created_at = "output"  # TODO: coloca timestamp
        path = "curated"
        if not os.path.exists(path):
            os.makedirs(path)

        filepath = f"{path}/data_{created_at}.csv"
        df = self._integrated_data
        df.to_csv(filepath, index=False)
        mlflow.log_artifact(filepath)
        # table = self._dataframe_to_arrow(self._integrated_data)
        # created_at = table.schema.metadata[b"created_at"].decode("utf-8").replace(" ", "_")
        # filepath = f"curated/datasets/data_{created_at}.parquet"
        # self._dl.store_data_in_lake(table, filepath)
        ###### adicionar
        self._logger.info(f"Data stored in {filepath}")

    def _dataframe_to_arrow(self, df: pd.DataFrame):
        self._logger.debug(f"Converting dataframe to arrow table...")
        attrs = self._build_metadata()
        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.cast(pa.schema(table.schema, metadata=attrs))
        return table

    def _build_metadata(self):
        attrs = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "param_1": f"sources: {self._sources}",  # TODO: filter only sources that were successfully integrated
            "param_2": f"period: {args.period}",
            "param_3": f"non_shared_feature_handler: {args.non_shared_feature_handler}",
        }
        attrs.update(self._build_variables("non_features"))
        attrs.update(self._build_variables("features"))
        return attrs

    def _build_variables(self, string_category):
        columns = [
            col
            for col in self._integrated_data.columns
            if col in self._all_metadata[string_category].keys()
        ]
        values = [self._all_metadata[string_category][col] for col in columns]
        keys = [f"{string_category}_{i}" for i in range(1, len(values) + 1)]
        return dict(zip(keys, values))


def assert_arguments():
    if args.sources != ["all"]:
        for source in args.sources:
            assert (
                source in INSTRUMENTS
            ), f"Argument --sources must be <all> or a subset of [{', '.join(INSTRUMENTS)}]. Got '{args.sources}'"
    assert len(args.period) in [
        1,
        2,
    ], f"Argument --period must have one or two values. Got {len(args.period)}"
    for date in args.period:
        assert is_valid_date(
            date
        ), f'Argument --period is not a valid date in format YYYY-MM-DD. Got "{date}"'
    if args.non_shared_feature_handler:
        assert (
            args.non_shared_feature_handler in FEATURE_HANDLERS
        ), f'Argument --non_shared_feature_handler must be one of [{", ".join(FEATURE_HANDLERS)}]. Got "{args.non_shared_feature_handler}"'


def parameter_parser():
    description = "Script to perform weather sources data integration."

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    parser = Logger.add_log_parameters(parser, os.path.basename(__file__))

    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help=f"REQUIRED. Sources to be integrated. Accepts <all> or a subset of {'{'+ ', '.join(INSTRUMENTS) + '}'}",
    ),
    parser.add_argument(
        "--period",
        nargs="+",
        required=True,
        help="REQUIRED. Period (<start_date>, <end_date>) to be integrated. Accepts one or two values in format YYYY-MM-DD. \
If only one value is passed, it will be considered as <start_date> and <end_date> will be set to today.",
    ),
    parser.add_argument(
        "--non_shared_feature_handler",
        nargs="?",
        default=None,
        help=f"OPTIONAL. Method to be applied for weather feature integration. Accepts one value in {'{'+ ', '.join(FEATURE_HANDLERS) + '}'}. \
Defines weather feature integration method for weather features not common to all source/instruments. If not passed, None will by applied.",
    )
    return parser.parse_args()


def main():
    assert_arguments()
    DataIntegrator().run()


if __name__ == "__main__":
    # load_dotenv('config/.env')
    args = parameter_parser()
    Logger.init(filename=args.logfile, level=args.loglevel, verbose=args.verbose)
    main()
