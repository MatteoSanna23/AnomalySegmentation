import os
import sys
from typing import Literal
import zipfile
import subprocess
import pkg_resources
import argparse

WEIGHTS = "../epoch_106-step_19902_eomt.ckpt"

DATASETS = {
    "road_anomaly": {
        "name": "ROAD ANOMALY",
        "input": "../Validation_Dataset/RoadAnomaly/images/*.jpg",
    },
    "smyic_ra21": {
        "name": "SMIYC RA-21",
        "input": "../Validation_Dataset/RoadAnomaly21/images/*.png",
    },
    "road_obstacle": {
        "name": "ROAD OBSTACLE",
        "input": "../Validation_Dataset/RoadObsticle21/images/*.webp",
    },
    "fs_lostfound": {
        "name": "FS L&F",
        "input": "../Validation_Dataset/FS_LostFound_full/images/*.png",
    },
    "fs_static": {
        "name": "FS STATIC",
        "input": "../Validation_Dataset/fs_static/images/*.jpg",
    },
}

def check_missing_requirements(
    path="/teamspace/studios/this_studio/AnomalySegmentation/eomt/requirements.txt",
):
    with open(path) as f:
        requirements = [r.strip() for r in f if r.strip() and not r.startswith("#")]

    missing = []
    for req in requirements:
        try:
            pkg_resources.require(req)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing.append(req)

    return missing


def install_packages(packages):
    if not packages:
        return

    cmd = [sys.executable, "-m", "pip", "install"] + packages
    subprocess.check_call(cmd)
    print("\nInstallation complete")


def setup():
    # ── Controllo dataset
    dataset_path = (
        "/teamspace/studios/this_studio/Validation_Dataset"
    )
    if not os.path.isdir(dataset_path):
        print("-Dataset not present, unzipping...")
        with zipfile.ZipFile(
            "/teamspace/studios/this_studio/Validation_Dataset.zip", "r"
        ) as zip_ref:
            zip_ref.extractall("/teamspace/studios/this_studio")
    else:
        print("-Dataset already present")

    print("-Installing requirements...")
    os.system(
        "cd /teamspace/studios/this_studio; "
        "pip install -r requirements.txt > /dev/null 2>&1"
    )

    missing = check_missing_requirements()
    if missing:
        print(f"Missing modules: {missing}")
        install_packages(missing)
    else:
        print("Requirements already satisfied")


def run_dataset(key):
    info = DATASETS[key]
    print(f"\n>>> Running on {info['name']}...")
    cmd = (
        f"python ./eval/evalAnomaly_temperature.py "
        f"--input \"{info['input']}\" "
        f'--loadWeights "{WEIGHTS}"'
    )
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",type=type["Literal"], help="Dataset to evalueate"
    )
    args = parser.parse_args()

    setup()

    if args.dataset == "all":
        print(">>> Running on all datasets...")
        for key in DATASETS:
            run_dataset(key)
    elif args.dataset in DATASETS:
        run_dataset(args.dataset)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Valid values:", ", ".join(list(DATASETS.keys()) + ["all"]))
