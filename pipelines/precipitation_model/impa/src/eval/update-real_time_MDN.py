import datetime
import subprocess
from argparse import ArgumentParser

from src.data.process.build_dataframe_from_radar import build_dataframe_from_radar
from src.data.process.despeckle_radar_data import despeckle_radar_data
from src.data.process.process_MDN import process_radar
from src.eval.predict_real_time import predict

# Consider 8 hours of data
N_HOURS = 8
DATA_UPLOAD_LAG = 10  # minutes

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--datetime", type=str, default=None, help="Datetime in ISO format, UTC timezone"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use for parallel processing",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for prediction")
    args = parser.parse_args()

    if args.datetime is None:
        dt = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(
            minutes=DATA_UPLOAD_LAG
        )
    else:
        dt = datetime.datetime.fromisoformat(args.datetime)

    print(f"Running predictions on datetime [{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC]")

    last_dt = dt.strftime("%Y%m%dT%H%M%SZ")
    first_dt = (dt - datetime.timedelta(hours=N_HOURS)).strftime("%Y%m%dT%H%M%SZ")

    # Download the latest data
    result = subprocess.call(["src/data/download_MDN.sh", first_dt, last_dt])

    # process data
    print("Processing radar data...")
    process_radar(num_workers=args.num_workers)

    # despeckle radar data
    with open("data/processed/processed_PPI_MDN/processed_files.log", "r") as f:
        processed_files = [line.strip() for line in f.readlines()]
    despeckle_radar_data(processed_files, num_workers=args.num_workers)

    build_dataframe_from_radar(overwrite=True, num_workers=args.num_workers, dt=dt)

    predict("MDN", num_workers=args.num_workers, cuda=args.cuda)
