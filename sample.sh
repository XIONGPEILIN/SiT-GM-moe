#!/bin/bash

echo "Running MIXED sampling with STOCHASTIC jump..."
accelerate launch \
  --num_processes 8 \
  --mixed_precision no \
  sample_ddp.py MIXED \
  --model "SiT-XL/2" \
  --ckpt "results_a100/3090-jump-flow/001-SiT-XL-2-Linear-velocity-None/checkpoints/0005000/model.pt" \
  --sample-dir "samples_trained_only" \
  --per-proc-batch-size 8 \
  --num-fid-samples 64 \
  --num-sampling-steps 1000 \
  --cfg-scale 1.0 \
  --tf32 \
  --stochastic-jump \
  --test-trained-only \
  --train-labels-json "results_a100/3090-jump-flow/001-SiT-XL-2-Linear-velocity-None/used_labels.json"

echo "Running MIXED sampling with DETERMINISTIC jump..."
accelerate launch \
  --num_processes 8 \
  --mixed_precision no \
  sample_ddp.py MIXED \
  --model "SiT-XL/2" \
  --ckpt "results_a100/3090-jump-flow/001-SiT-XL-2-Linear-velocity-None/checkpoints/0005000/model.pt" \
  --sample-dir "samples_trained_only" \
  --per-proc-batch-size 8 \
  --num-fid-samples 64 \
  --num-sampling-steps 1000 \
  --cfg-scale 1.0 \
  --tf32 \
  --no-stochastic-jump \
  --test-trained-only \
  --train-labels-json "results_a100/3090-jump-flow/001-SiT-XL-2-Linear-velocity-None/used_labels.json"