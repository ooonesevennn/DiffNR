#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train_DiffNR.py \
    -s "${SOURCE_PATH:-data/example_case}" \
    -m "${MODEL_OUTPUT_PATH:-output/example_case}" \
    --config "${CONFIG_PATH:-configs/luna16.yaml}" \
    --slicefixer_model_path "${SLICEFIXER_MODEL_PATH:-checkpoints/slicefixer/model.pkl}" \
    --sd_turbo_path "${SD_TURBO_PATH:-stabilityai/sd-turbo}" \
    --organ_type "${ORGAN_TYPE:-Chest}"