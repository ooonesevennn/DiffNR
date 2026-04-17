# Script to train all cases.

import os
import os.path as osp
import glob
import subprocess
import argparse


def main(args):
    source_path = args.source
    output_path = args.output
    device = args.device
    config_path = args.config
    ckpt_path = args.ckpt
    organ_type = args.organ_type
    sd_turbo_path = args.sd_turbo_path

    case_paths = sorted(glob.glob(osp.join(source_path, "*")))

    if len(case_paths) == 0:
        raise ValueError("{} find no folder!".format(case_paths))

    for case_path in case_paths:
        case_name = osp.basename(case_path)
        case_output_path = f"{output_path}/{case_name}"
        if not osp.exists(case_output_path):
            cmd = (
                f"CUDA_VISIBLE_DEVICES={device} python train_DiffNR.py "
                f"-s {case_path} -m {case_output_path} "
                f"--slicefixer_model_path {ckpt_path} "
                f"--organ_type {organ_type} "
                f"--sd_turbo_path {sd_turbo_path}"
            )
            if config_path:
                cmd += f" --config {config_path}"
            os.system(cmd)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/cases", type=str, help="Path to CT dataset root.")
    parser.add_argument("--output", default="output/synthetic_dataset/cone_ntrain_50_angle_360", type=str, help="Path to output.")
    parser.add_argument("--config", default=None, type=str, help="Path to config.")
    parser.add_argument("--device", default=0, type=int, help="GPU device.")
    parser.add_argument("--ckpt", default="checkpoints/slicefixer/model.pkl", type=str, help="Path to SliceFixer checkpoint.")
    parser.add_argument("--sd_turbo_path", default="stabilityai/sd-turbo", type=str, help="Path or HF model id for SD-Turbo.")
    parser.add_argument("--organ_type", default="Chest", type=str, help="CT organ type")
    #parser.add_argument("--weight", default=1, type=int, help="Weight for diffusion ssim loss.")
    # fmt: on

    args = parser.parse_args()
    main(args)