# -*- coding: utf-8 -*-
# pylint: disable= C0207
"""
Tasks
"""
import datetime
import os
from pathlib import Path
from time import sleep
from typing import List

import matplotlib.colors as mcolors  # pylint: disable=E0611, E0401
import matplotlib.pyplot as plt  # pylint: disable=E0611, E0401
import numpy as np
import pandas as pd
import pendulum  # pylint: disable=E0611, E0401

# import seaborn as sns
from basedosdados import Base  # pylint: disable=E0611, E0401
from google.cloud import bigquery  # pylint: disable=E0611, E0401
from PIL import Image
from prefect import task  # pylint: disable=E0611, E0401
from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401

from pipelines.precipitation_model.rionowcast.utils import calculate_opacity
from pipelines.utils.utils_wf import convert_dtypes

# @task()
# def get_billing_project_id(
#     bd_project_mode: str = "prod",
#     billing_project_id: str = None,
# ) -> str:
#     """
#     Get billing project id.
#     OBS: not workin in this basedosdados version
#     """
#     if not billing_project_id:
#         log("Billing project ID was not provided, trying to get it from environment variable")
#     try:
#         bd_base = Base()
#         billing_project_id = bd_base.config["gcloud-projects"][bd_project_mode]["name"]
#         log(f"Billing project ID was inferred from environment variables: {billing_project_id}")
#     except KeyError:
#         pass
#     if not billing_project_id:
#         raise ValueError(
#             "billing_project_id must be either provided or inferred from environment variables"
#         )
#     log(f"Billing project ID: {billing_project_id}")
#     return billing_project_id


def download_data_from_bigquery(query: str, billing_project_id: str) -> pd.DataFrame:
    """ADD"""
    # pylint: disable=E1124, protected-access
    # client = google_client(billing_project_id, from_file=True, reauth=False)
    # job_config = bigquery.QueryJobConfig()
    # # job_config.dry_run = True

    # # Get data
    log("Querying data from BigQuery")
    # job = client["bigquery"].query(query, job_config=job_config)
    # https://github.com/prefeitura-rio/pipelines_rj_iplanrio/blob/ecd21c727b6f99346ef84575608e560e5825dd38/pipelines/painel_obras/dump_data/tasks.py#L39

    bq_client = bigquery.Client(
        credentials=Base(bucket_name="rj-cor")._load_credentials(mode="prod"),
        project=billing_project_id,
    )
    job = bq_client.query(query)
    while not job.done():
        sleep(1)
    log("Getting result from query")
    results = job.result()
    log("Converting result to pandas dataframe")
    dfr = results.to_dataframe()
    log("End download data from bigquery")

    # Get data
    # log("Querying data from BigQuery")
    # job = client["bigquery"].query(query)
    # while not job.done():
    #     sleep(1)
    return dfr


@task(nout=3)
def calculate_start_and_end_date(
    hours_from_past: int = 6,
    end_historical_datetime: str = None,
) -> tuple[str, str]:
    """
    Calculate the start and end datetime based on a specified number of hours in the past.

    The function computes two datetime values: the start and end times.
    The `end_historical_datetime_date` serves as the reference endpoint for calculations, and
    the `hours_from_past` parameter determines how far back in time the start date will be.
    Both values are rounded to the nearest full hour, and the result is returned as strings in
    the format 'yyyy-mm-dd hh:mm:ss' (UTC).

    Parameters:
    ----------
    hours_from_past : int
        Number of hours to subtract from `end_historical_datetime_date` to determine the start
        datetime.
    end_historical_datetime_date : str, optional
        The ending datetime as a string in the format 'yyyy-mm-dd hh:mm:ss' in UTC timezone.
        If not provided, the current UTC time (`datetime.datetime.utcnow()`) is used as the default.

    Returns:
    -------
    tuple[str, str]
        A tuple containing two strings:
        - The start datetime (earlier timestamp).
        - The end datetime (reference timestamp).

    Raises:
    ------
    ValueError
        If `end_historical_datetime_date` is provided but does not follow the expected format.

    Example:
    -------
    >>> calculate_start_and_end_date(6, "2024-11-13 12:00:00")
    ('2024-11-13 06:00:00', '2024-11-13 12:00:00')
    """
    hours_from_past = int(hours_from_past)
    if not end_historical_datetime:
        end_historical_datetime = datetime.datetime.utcnow()
    else:
        try:
            end_historical_datetime = datetime.datetime.strptime(
                end_historical_datetime, "%Y-%m-%d %H:%M:%S"
            )
        except ValueError as error:
            raise ValueError(f"Invalid date format: {error}")

    end_datetime = pendulum.instance(end_historical_datetime).replace(
        minute=0, second=0, microsecond=0
    )
    start_datetime = end_datetime.subtract(hours=hours_from_past)

    # Ajustar para o fuso horário de Brasília
    end_datetime_brasilia = end_datetime.in_timezone("America/Sao_Paulo")

    print(f"Start datetime in UTC: {start_datetime}, End datetime in UTC: {end_datetime}")

    return (
        start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        end_datetime_brasilia.strftime("%Y-%m-%d %H:%M:%S"),
    )


@task()
def query_data_from_gcp(  # pylint: disable=too-many-arguments, too-many-locals
    dataset_id: str,
    table_id: str,
    billing_project_id: str,
    start_datetime: str = None,
    end_datetime: str = None,
    filename: str = "data",
    save_format: str = "csv",
    renormalization: bool = True,
) -> Path:
    """
    Download historical data from source.
    format: csv or parquet
    """
    log(f"Start downloading {dataset_id}.{table_id} data")

    directory_path = Path(f"{dataset_id}_{table_id}")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    savepath = directory_path / f"{filename}.{save_format}"
    start_date, end_date = start_datetime[:10], end_datetime[:10]
    # pylint: disable=consider-using-f-string
    # noqa E262
    query = """
        SELECT
            * EXCEPT (update_time, model_version, ano_particao, mes_particao, data_particao)
        FROM rj-cor.{}.{}
        WHERE data_particao BETWEEN '{}' AND '{}'
        AND datetime >= '{}' AND datetime < '{}'
        """.format(
        dataset_id, table_id, start_date, end_date, start_datetime, end_datetime
    )

    log(f"Query used to download data:\n{query}")

    dfr = download_data_from_bigquery(query=query, billing_project_id=billing_project_id)
    dtype_mapping = {
        "station_id": "int64",
        "datetime": "datetime64[ns, UTC]",
        "precipitation": "float64",
        "latitude": "float64",
        "longitude": "float64",
        "altitude": "int64",
        "horizontal_reflectivity_mean": "float64",
    }
    dfr = convert_dtypes(dfr, dtype_mapping)
    log(f"Shape of dataset before renormalization: {dfr.shape}")
    if "horizontal_reflectivity_mean" in dfr.columns and renormalization:
        min_val = dfr["horizontal_reflectivity_mean"].min()
        max_val = dfr["horizontal_reflectivity_mean"].max()
        log(f"Min and max values before renormalization {min_val}, {max_val}")
        dfr["horizontal_reflectivity_mean"] = (dfr["horizontal_reflectivity_mean"] - min_val) / (
            max_val - min_val
        )
    # TODO: remove normalization after rionowcast finish preprocessing fixes

    log(f"df from {table_id}: {dfr.iloc[0]}")
    log(f"dtypes from {table_id}: {dfr.dtypes}")

    if save_format == "csv":
        dfr.to_csv(savepath, index=False)
    elif save_format == "parquet":
        dfr.to_parquet(savepath, index=False)
    # bd.download(savepath=savepath, query=query, billing_project_id=billing_project_id)

    log(f"{table_id} data saved on {savepath}")
    return savepath


@task
def geolocalize_data(
    denormalized_prediction_dataset: np.array,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> pd.DataFrame:
    """
    Geolocalize the denormalized prediction data by mapping array indices to lat/lon coordinates.
    Any value less then 0.02 will be set to zero. (confirm this)

    Parameters
    ----------
    denormalized_prediction_dataset : np.array
        3D numpy array containing denormalized prediction data with shape (3, height, width)
        representing predictions for 1h, 2h and 3h ahead
    min_lon : float
        Minimum longitude value for the prediction grid
    min_lat : float
        Minimum latitude value for the prediction grid
    max_lon : float
        Maximum longitude value for the prediction grid
    max_lat : float
        Maximum latitude value for the prediction grid

    Returns
    -------
    pd.DataFrame
        DataFrame containing the geolocalized predictions with columns:
        - latitude: Latitude coordinate for each prediction point
        - longitude: Longitude coordinate for each prediction point
        - 1h_prediction: Predicted value 1 hour ahead
        - 2h_prediction: Predicted value 2 hours ahead
        - 3h_prediction: Predicted value 3 hours ahead
    """
    denormalized_prediction_dataset[denormalized_prediction_dataset < 0.2] = 0

    column_names = ["latitude", "longitude", "1h_prediction", "2h_prediction", "3h_prediction"]
    geolocalized_df = pd.DataFrame(columns=column_names)

    dataset_shape = denormalized_prediction_dataset.shape
    log(f"dataset_shape in geolocalize data: {dataset_shape}")

    lat_scale = (max_lat - min_lat) / dataset_shape[1]
    lon_scale = (max_lon - min_lon) / dataset_shape[2]
    for i, j in np.ndindex(
        denormalized_prediction_dataset.shape[1], denormalized_prediction_dataset.shape[2]
    ):

        row = {
            "latitude": -(i + 1) * lat_scale + max_lat,
            "longitude": (j + 1) * lon_scale + min_lon,
            "1h_prediction": denormalized_prediction_dataset[0, i, j],
            "2h_prediction": denormalized_prediction_dataset[1, i, j],
            "3h_prediction": denormalized_prediction_dataset[2, i, j],
        }

        geolocalized_df = pd.concat([geolocalized_df, pd.DataFrame([row])], ignore_index=True)

    return geolocalized_df


@task
def add_caracterization_columns_on_dfr(
    geolocalized_df: pd.DataFrame, model_version, reference_datetime: str
) -> pd.DataFrame:
    """
    Add model version and reference datetime columns to the geolocalized dataframe.

    Parameters
    ----------
    geolocalized_df : pd.DataFrame
        DataFrame containing the geolocalized prediction data
    model_version : int or str
        Version number of the model used for predictions
    reference_datetime : str
        Reference datetime for the predictions in format 'YYYY-MM-DD HH:mm:ss'

    Returns
    -------
    pd.DataFrame
        Input dataframe with added model_version and reference_datetime columns
    """
    geolocalized_df["model_version"] = model_version
    geolocalized_df["reference_datetime"] = reference_datetime
    return geolocalized_df


class CustomNormalize(mcolors.Normalize):
    """Função personalizada de colormap que retorna transparência para valores < 0.02"""

    def __call__(self, value, clip=False):
        # Valores menores que 0.02 serão mapeados como NaN (transparentes)
        if np.isscalar(value):
            if value < 0.02:
                return np.nan
        else:
            value = np.array(value)
            value[value < 0.02] = np.nan
        return super().__call__(value, clip)


# pylint: disable=too-many-locals
@task
def create_image(dataframe: pd.DataFrame, filename: str) -> List:
    """
    Create heatmap visualizations of precipitation predictions.

    Takes a DataFrame containing geolocalized precipitation predictions and generates heatmap
    visualizations for 1-hour, 2-hour and 3-hour predictions using a custom color scheme.
    The heatmaps are saved as image files in prediction-specific subdirectories.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing the geolocalized prediction data with columns:
        - latitude: Latitude coordinates
        - longitude: Longitude coordinates
        - 1h_prediction: 1-hour precipitation predictions
        - 2h_prediction: 2-hour precipitation predictions
        - 3h_prediction: 3-hour precipitation predictions
    filename : str
        Base filename to use when saving the generated images

    Returns
    -------
    List[str]
        List of paths to the saved heatmap image files

    Notes
    -----
    Uses a custom color scheme based on precipitation intensity levels from AlertaRio.
    Creates separate subdirectories for 1h, 2h and 3h prediction images.
    """

    # alertario_precipitation_colors = [
    #     # {"value": 0, "color": "#eeeee4"},  # Nenhuma cor para o valor 0
    #     {"value": 0.02, "color": "#63bbff"},
    #     {"value": 5, "color": "#91ccab"},
    #     {"value": 10, "color": "#bfdd56"},
    #     {"value": 15, "color": "#eeee01"},
    #     {"value": 20, "color": "#ffd163"},
    #     {"value": 25, "color": "#ffb421"},
    #     {"value": 30, "color": "#ff9700"},
    #     {"value": 35, "color": "#f57000"},
    #     {"value": 40, "color": "#ee5500"},
    #     {"value": 45, "color": "#ee2a00"},
    #     {"value": 50, "color": "#ED0000"},
    #     {"value": 55, "color": "#d40000"},
    #     {"value": 60, "color": "#bc0000"},
    #     {"value": 65, "color": "#a30000"},
    #     {"value": 70, "color": "#8A0000"},
    #     {"value": 75, "color": "#6e0000"},
    #     {"value": 80, "color": "#530000"},
    #     {"value": 85, "color": "#380000"},
    #     {"value": 90, "color": "#1C0000"},
    # ]
    alertario_precipitation_colors = [
        {"value": 0.02, "color": "#66BCFB"},
        {"value": 10, "color": "#BEDD58"},
        {"value": 20, "color": "#F6D001"},
        {"value": 30, "color": "#FD9201"},
        {"value": 40, "color": "#F64801"},
        {"value": 50, "color": "#EF0101"},
        {"value": 60, "color": "#BC0101"},
        {"value": 70, "color": "#870101"},
        {"value": 80, "color": "#550101"},
        {"value": 90, "color": "#200101"},
    ]

    filtered_colors = [
        (entry["value"], entry["color"])
        for entry in alertario_precipitation_colors
        if entry["color"] is not None
    ]

    values, colors = zip(*filtered_colors)
    vmin, vmax = min(values), max(values)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    cmap.set_under("#FFFFFF")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # norm = CustomNormalize(vmin=vmin, vmax=vmax)

    dataframe = dataframe.sort_values(by=["latitude", "longitude"], ascending=[False, True])

    predictions = ["1h_prediction", "2h_prediction", "3h_prediction"]
    dataframe[predictions] = dataframe[predictions].astype(float)
    dataframe[predictions] = dataframe[predictions].replace(np.nan, 0)

    image_path_list = []
    for prediction in predictions:
        heatmap_data = dataframe.pivot(
            index="latitude", columns="longitude", values=prediction
        ).sort_index(ascending=False)
        log(f"Heatmap before changing values less than 0.2 to nan:\n{heatmap_data.iloc[:5, :5]}")
        heatmap_data[heatmap_data < 0.2] = 0
        log(f"Heatmap after changing values less than 0.2 to nan:\n{heatmap_data.iloc[:5, :5]}")

        # nan_count = np.isnan(heatmap_data).sum().sum()
        log(f"Min value: {np.min(heatmap_data)}")
        log(f"Max value: {np.max(heatmap_data)}")

        interpolation = "catrom"  # "spline36", "bicubic", "gaussian", "bilinear", "catrom"
        plt.figure(figsize=(10, 10))
        plt.imshow(
            heatmap_data,
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
            interpolation_stage="rgba",
        )
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(ticks=[], labels=[])
        plt.yticks(ticks=[], labels=[])
        plt.axis("off")

        # não pode ter nada no final do nome, para ter no começo tem que
        # alterar a busca no bucket na api
        directory_path = "prediction_images/" + prediction

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        image_path = f"{directory_path}/{filename}.png"
        plt.savefig(image_path, pad_inches=0, dpi=200, bbox_inches="tight", transparent=True)
        # plt.show()
        image_path_list.append(image_path)

    image_path_list.sort()
    return image_path_list


@task
def add_transparency_on_image_whites(img_paths: List[str]) -> List[str]:
    """
    Adiciona transparência a uma imagem com base na proximidade dos pixels à cor branca.

    A função processa uma imagem no formato RGBA e ajusta o canal alfa (transparência)
    para cada pixel. Pixels cuja intensidade mínima (entre os canais R, G e B) seja menor
    ou igual a 200 são definidos como totalmente opacos (opacidade máxima de 255). Para
    os demais pixels, a opacidade é calculada com base na distância em relação à cor branca
    (255, 255, 255).

    A transparência é aplicada de forma que:
        - 0 significa completamente transparente.
        - 255 significa completamente opaco.

    Args:
        img_paths (list): Lista com os caminho para as imagens de entrada no formato PNG.

    Returns:
        list: Lista com os caminho para as novas imagens com transparência aplicada.
    """

    # saved_img_paths = []
    for img_path in img_paths:
        image = Image.open(img_path).convert(
            "RGBA"
        )  # Convertendo para RGBA para permitir a transparência

        data = np.array(image)

        # Atualizando os valores de alfa (transparência) para cada pixel
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                pixel = data[i, j]
                if pixel.min() <= 200:
                    opacity = 255  # total opacity
                    data[i, j] = (pixel[0], pixel[1], pixel[2], opacity)
                else:
                    opacity = calculate_opacity(pixel)
                    data[i, j] = (pixel[0], pixel[1], pixel[2], opacity)

        # Converter o array NumPy de volta para uma imagem
        new_image = Image.fromarray(data, "RGBA")

        # saved_img_path = "data/test_add_image_transparency.png"
        new_image.save(img_path)
    # new_image.show()

    return img_paths
