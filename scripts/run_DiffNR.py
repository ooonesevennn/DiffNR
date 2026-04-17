#!/usr/bin/env python3
"""Unified launcher for DiffNR training workflows."""

import argparse
import os
import subprocess


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Unified DiffNR runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run single-case training")
    train_parser.add_argument("--source", "-s", required=True, help="Path to case folder")
    train_parser.add_argument("--output", "-m", required=True, help="Output experiment directory")
    train_parser.add_argument("--config", default="configs/luna16.yaml")
    train_parser.add_argument("--slicefixer-model-path", required=True)
    train_parser.add_argument("--sd-turbo-path", default=os.environ.get("SD_TURBO_PATH", "stabilityai/sd-turbo"))
    train_parser.add_argument("--organ-type", default="Chest", choices=["Chest", "Tooth"])
    train_parser.add_argument("--gpu", default="0", help="CUDA device id")

    batch_parser = subparsers.add_parser("batch", help="Run multi-case training")
    batch_parser.add_argument("--source", required=True, help="Directory containing multiple cases")
    batch_parser.add_argument("--output", required=True, help="Output root")
    batch_parser.add_argument("--config", default="configs/luna16.yaml")
    batch_parser.add_argument("--slicefixer-model-path", required=True)
    batch_parser.add_argument("--sd-turbo-path", default=os.environ.get("SD_TURBO_PATH", "stabilityai/sd-turbo"))
    batch_parser.add_argument("--organ-type", default="Chest", choices=["Chest", "Tooth"])
    batch_parser.add_argument("--gpu", default="0", help="CUDA device id")

    args = parser.parse_args()

    if args.command == "train":
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        cmd = [
            "python",
            "train_DiffNR.py",
            "-s",
            args.source,
            "-m",
            args.output,
            "--config",
            args.config,
            "--slicefixer_model_path",
            args.slicefixer_model_path,
            "--sd_turbo_path",
            args.sd_turbo_path,
            "--organ_type",
            args.organ_type,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
        return

    cmd = [
        "python",
        "scripts/train_all_DiffNR.py",
        "--source",
        args.source,
        "--output",
        args.output,
        "--config",
        args.config,
        "--ckpt",
        args.slicefixer_model_path,
        "--sd_turbo_path",
        args.sd_turbo_path,
        "--organ_type",
        args.organ_type,
        "--device",
        str(args.gpu),
    ]
    run_cmd(cmd)


if __name__ == "__main__":
    main()
