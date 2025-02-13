#!/usr/bin/env python

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.trainer_integration.custom_trainer import CustomGRPOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train product title RL model with custom rewards.")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_path", type=str, default="data/example_dataset.csv")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_train_steps", type=int, default=50)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset from CSV
    dataset = load_dataset("csv", data_files={"train": args.dataset_path})["train"]

    # Optionally, you can do dataset = dataset.map(your_data_parsing_fn) here

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Initialize custom trainer
    trainer = CustomGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_func=None,  # We'll use default compute_total_reward inside custom trainer, or pass it here
        output_dir=args.output_dir,
        total_train_steps=args.num_train_steps,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
