# -*- coding: utf-8 -*-
# flake8: noqa: E501

import datetime
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from pipelines.precipitation_model.impa.src.eval.metrics.metrics import metrics_dict
from pipelines.precipitation_model.impa.src.models.mamba.predict import main as mamba_predict
from pipelines.precipitation_model.impa.src.models.predict import main as general_predict
from pipelines.precipitation_model.impa.src.models.pysteps_LK.predict import main as pysteps_predict

MAP_CENTER = {"lat": -22.914550816555533, "lon": -43.502443050594596}
ZOOM = 8
eval_window = range(96, 160)

predict_dict = {
    "PySTEPS": pysteps_predict,
    "UNET": general_predict,
    "EVONET": general_predict,
    "NowcastNet": general_predict,
    "MetNet3": general_predict,
    "MetNet_lead_time": general_predict,
    "Mamba": mamba_predict,
}


def fetch_pred_keys(keys, nlags, timestep):
    datetimes = pd.to_datetime(keys)
    relevant_datetimes = [
        [
            f"{dt.strftime('%Y%m%d/%H%m')}/{(dt + datetime.timedelta(minutes=timestep * lag)).strftime('%Y%m%d-%H%M')}"
            for dt in datetimes
        ]
        for lag in range(1, nlags + 1)
    ]
    return list(zip(*relevant_datetimes))


def get_img(
    values,
    latlons,
    model_name,
    feature,
    delta,
    metric_names=None,
    ground_truth=None,
    zoom=ZOOM,
    bg_color="rgba(255,255,255,255)",
    height=500,
    width=1000,
    no_colorbar=False,
):
    if metric_names is not None and ground_truth is not None:
        metrics = [
            metrics_dict[metric_name](
                values[eval_window, eval_window], ground_truth[eval_window, eval_window]
            )
            for metric_name in metric_names
        ]
    values = values.flatten()
    if feature == "DBZH":
        values = 10 * np.log10(values)
        exclude_idx = np.logical_or(values < 10, np.isnan(values))
        marker_dict = dict(
            cmin=0,
            cmax=60,
            colorscale="rainbow",
            opacity=0.3,
            colorbar={"title": feature} if not no_colorbar else None,
        )
    elif feature in ["ABI-L2-RRQPEF", "corrected_ABI-L2-RRQPEF"]:
        exclude_idx = np.logical_or(values < 0.5, np.isnan(values))
        marker_dict = dict(
            cmin=0,
            cmax=20,
            colorscale="rainbow",
            opacity=0.5,
            colorbar={"title": feature} if not no_colorbar else None,
            sizemin=7 * ((zoom / ZOOM) ** 2),
            size=np.ones(values[~exclude_idx].shape),
        )
    else:
        raise NotImplementedError(f"{feature} not implemented yet.")
    trace = go.Scattermapbox(
        lat=np.concatenate([latlons[:, :, 0].flatten()[~exclude_idx], np.array([0])]),
        lon=np.concatenate([latlons[:, :, 1].flatten()[~exclude_idx], np.array([0])]),
        mode="markers",
        marker={**marker_dict, "color": np.concatenate([values[~exclude_idx], np.array([0])])},
    )
    fig = go.Figure(trace)
    if metric_names is not None:
        metrics_text = "<br>".join(
            [f"{metric_name}: {metric:.2E}" for metric_name, metric in zip(metric_names, metrics)]
        )
        fig.add_annotation(
            x=0.9975,
            y=0.0075,
            text=metrics_text,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            opacity=1,
            font=dict(
                size=20,
                color="black",
                family="Courier New, monospace",
            ),
        )
    fig.update_layout(
        margin=dict(l=5, r=5, t=40, b=20),
        height=height,
        width=width,
        title_text=(
            f"{model_name} T{'+'*(delta >= 0)}{delta} minutes" if delta is not None else model_name
        ),
        title_x=0.5,
        title_font_size=25,
        title_font_family="Open Sans",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
    )
    fig.update(layout_showlegend=False)
    fig.update_mapboxes(center=MAP_CENTER, zoom=zoom, style="open-street-map")
    bytes = fig.to_image(format="png")
    buf = io.BytesIO(bytes)
    img = Image.open(buf)
    img = np.asarray(img)
    buf.close()
    return img
