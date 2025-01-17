# -*- coding: utf-8 -*-
# flake8: noqa: E501

import json
import pathlib
from argparse import ArgumentParser

from prefeitura_rio.pipelines_utils.logging import log

from pipelines.precipitation_model.impa.src.utils.eval_utils import predict_dict
from pipelines.precipitation_model.impa.src.utils.general_utils import print_warning

dataframe_dict = {
    "SAT": {
        "dataframe_filepath": "data/dataframes/SAT-CORRECTED-ABI-L2-RRQPEF-real_time-{location}/test.hdf",
        "dataframe": "SAT-ABI-L2-RRQPEF-{location}-file=thr=0_split2",
        "data_type": "SATELLITE",
        "locations": ["rio_de_janeiro"],
    },
    "MDN": {
        "dataframe_filepath": "data/dataframes/MDN-d2CMAX-DBZH-real_time/test.hdf",
        "dataframe": "RADAR-d2CMAX-DBZH-large_split_radar",
        "data_type": "RADAR",
        "locations": None,
    },
}


def predict(dataframe_key, num_workers=8, cuda=False):
    accelerator = "gpu" if cuda else "cpu"

    config = pathlib.Path(f"pipelines/precipitation_model/impa/src/eval/real_time_config_{dataframe_key}.json")
    with open(config, "r") as json_file:
        specs_dict = json.load(json_file)

    for model_name, info in specs_dict["models"].items():
        if "needs_cuda" in info and info["needs_cuda"] and not cuda:
            print_warning(f"Skipping {model_name} because it needs CUDA...")
            continue

        predict_func = predict_dict[model_name]
        if "args" in info:
            args = info["args"]
            if "locations" not in args:
                args["locations"] = None
            args["num_workers"] = num_workers
            args["accelerator"] = accelerator
            args |= dataframe_dict[dataframe_key]

            if "params_filepath" in info:
                with open(info["params_filepath"], "r") as json_file:
                    params = json.load(json_file)
                args |= params
                if model_name in [
                    "UNET",
                    "EVONET",
                    "NowcastNet",
                    "MetNet3",
                    "MetNet_lead_time",
                ]:
                    args["model"] = args["model_name"]
                del params["model_name"]
                predict_func(args, params)
            else:
                predict_func(args)
            continue

        model_path = pathlib.Path(f"pipelines/precipitation_model/impa/models_{dataframe_key}/{model_name}/")
        model_file = model_path / info["model_file"]

        output_predict_filepaths = [f"pipelines/precipitation_model/impa/predictions_{dataframe_key}/{model_name}.hdf"]

        # Standard arguments
        args = {
            "overwrite": True,
            "accelerator": accelerator,
            "output_predict_filepaths": output_predict_filepaths,
            "num_workers": num_workers,
            "input_model_filepath": model_file,
            "compile": False,
            "batch_to_predict": 8,
        }
        args |= dataframe_dict[dataframe_key]
        # Model params
        with open(model_path / "params.json", "r") as json_file:
            params = json.load(json_file)
        args |= params
        ##################################
        if model_name in [
            "UNET",
            "EVONET",
            "NowcastNet",
            "MetNet3",
            "MetNet_lead_time",
        ]:
            args["model"] = args["model_name"]
        ##################################
        del args["model_name"]
        predict_func(args, params)
    return output_predict_filepaths


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    predict(**vars(args))
