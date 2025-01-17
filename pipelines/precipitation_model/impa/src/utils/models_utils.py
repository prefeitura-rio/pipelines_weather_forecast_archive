# -*- coding: utf-8 -*-
# flake8: noqa: E501

from pipelines.precipitation_model.impa.src.data.HDFDataset2 import HDFDataset2
from pipelines.precipitation_model.impa.src.data.HDFDatasetLocations import HDFDatasetLocations
from pipelines.precipitation_model.impa.src.data.HDFDatasetMerged import HDFDatasetMerged
from pipelines.precipitation_model.impa.src.data.PredHDFDataset2 import PredHDFDataset2
from pipelines.precipitation_model.impa.src.data.PredHDFDatasetLocations import PredHDFDatasetLocations

options_pretrained = {
    1: {
        "SAT-corrected_ABI-L2-RRQPEF-heavy_rain": [
            "UNET/20897c1c0624963957bc961eb6eba9e7",
            "UNET/279de3ee5bf9374409d38ebe1cbafade",
        ],
        "SAT-ABI-L2-RRQPEF-rain_events-sat-thr=10-radius=1h": [
            # n_after = 6
            "UNET/30e5b2de4117c6b6c27ab1d5dcf5a927",
            # n_after = 10
            "UNET/f2768cd156f48152041f4a791c155d48",
        ],
        "square-RADAR-d2CMAX-DBZH-heavy_rain": ["UNET/f572a1c4d4db45e33eb04a5be15db840"],
        "SAT-ABI-L2-RRQPEF-{location}-file=thr=0": [],
    },
    2: {
        "SAT-corrected_ABI-L2-RRQPEF-heavy_rain": [
            "Evolution_Network/1e4883625ea0ad73153141928b0bb552",
            "Evolution_Network/295aa5a022bf8862d9c44fe103e4e57f",
            # n_after = 6
            "Evolution_Network/e4956f4b7d5bae420dd696c27109bbb7",
            # n_after = 18
            # Dataset novo
        ],
        "SAT-ABI-L2-RRQPEF-rain_events-sat-thr=10-radius=1h": [
            "Evolution_Network/f754fc3287b30783a79e4795da67d113",
            "Evolution_Network/ee9af027c77c5657c94ded8aecbb4d47",
            # New dataset
            # n_after = 6
            "Evolution_Network/68ad4908e04740a4c1efc0251708be3c",
            # n_after = 18
            "Evolution_Network/86dde2f78480d7a7051632b8e9a42ffd",
        ],
        "square-RADAR-d2CMAX-DBZH-heavy_rain": [
            "Evolution_Network/ab3c1cd84810b8479d7ede1097a86e75"
        ],
        "RADAR-d2CMAX-DBZH-file=thr=0_split_radar": [
            "Evolution_Network/3eacfec287a8a63979ea52740fd26e47"
        ],
        "RADAR-d2CMAX-DBZH-large_split_radar": [
            "Evolution_Network/86dde2f78480d7a7051632b8e9a42ffd"
        ],
        "SAT-ABI-L2-RRQPEF-{location}-file=thr=0": [
            # n_after = 6
            "Evolution_Network/a0b0b56d49855f2d95a3ca53c88db807"
        ],
        "SAT-ABI-L2-RRQPEF-{location}-file=thr=0_split2": [
            "Evolution_Network/8c04e5f4acbc775e2d07f468888205d4",
            "Evolution_Network/2d1f5e86aa89e1d7354b7fbba6289473",
        ],
    },
    3: {
        "SAT-ABI-L2-RRQPEF-{location}-file=thr=0_split2": [
            "Metnet3/233006ddcc8cb44df47190c7a38590bb"
        ]
    },
}


def get_ds(
    dataframe_filepath,
    n_before,
    n_after,
    new_dataset,
    merge,
    lead_time,
    predict_sat,
    locations,
    needs_prediction,
    args_dict,
):
    if new_dataset:
        if not needs_prediction:
            ds = HDFDatasetLocations(
                dataframe_filepath,
                locations,
                n_after=n_after,
                n_before=n_before,
                leadtime_conditioning=lead_time,
            )
        else:
            if args_dict["predictions"] == 1 or args_dict["predictions"] == 2:
                n_predictions = n_after

                ds = PredHDFDatasetLocations(
                    dataframe_filepath,
                    n_predictions=n_predictions,
                    locations=locations,
                    dataset=args_dict["predict_dataframe"],
                    n_after=n_after,
                    n_before=n_before,
                )
            else:
                raise ValueError("NowcastNet needs predictions to train.")

        assert args_dict["normalized"] >= 3
        val = 0.5

    else:
        if not merge:
            if not needs_prediction:
                ds = HDFDataset2(
                    dataframe_filepath,
                    n_after=n_after,
                    n_before=n_before,
                    leadtime_conditioning=lead_time,
                    get_item_output=["X", "Y", "index"],
                    use_datetime_keys=True,
                )

            else:
                if args_dict["predictions"] == 1 or args_dict["predictions"] == 2:
                    n_predictions = n_after

                    ds = PredHDFDataset2(
                        dataframe_filepath,
                        n_predictions=n_predictions,
                        dataset=args_dict["predict_dataframe"],
                        n_after=n_after,
                        n_before=n_before,
                        get_item_output=["X", "Y", "index"],
                        use_datetime_keys=True,
                    )
                else:
                    raise ValueError("NowcastNet needs predictions to train.")
            assert args_dict["normalized"] <= 2
            val = 1.0

        else:
            if not needs_prediction:
                sat_dataframe_filepath = dataframe_filepath.parents[1].joinpath(
                    "SAT-ABI-L2-RRQPEF-rio_de_janeiro-file=thr=0_split_radar/train.hdf"
                )
                ds = HDFDatasetMerged(
                    sat_dataframe_filepath,
                    dataframe_filepath,
                    n_before=n_before,
                    n_after=n_after,
                    leadtime_conditioning=lead_time,
                    predict_sat=predict_sat,
                )
                val = 0.5
    print(f"Use dataset: {ds}")
    return ds, val
