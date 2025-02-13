#!/bin/bash
# Simple script to run the training

python train.py \
  --model_name gpt2 \
  --dataset_path data/example_dataset.csv \
  --output_dir output \
  --num_train_steps 10 \
  --batch_size 2 \
  --completions_per_prompt 2
