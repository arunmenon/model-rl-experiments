#!/usr/bin/env python

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.trainer_integration.custom_grpo_trainer import CustomGRPOTrainer
from src.trainer_integration.training_config import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser("Train product title enhancement with GRPO")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_path", type=str, default="data/example_dataset.csv")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_train_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--completions_per_prompt", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()

    # Prepare config
    config = TrainingConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size
    )

    # Load dataset
    dataset = load_dataset("csv", data_files={"train": config.dataset_path})["train"]
    # dataset = dataset.map(...) if you want additional parsing
    
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    # Initialize custom trainer
    trainer = CustomGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=config.output_dir,
        total_train_steps=config.num_train_steps,
        batch_size=config.batch_size,
        completions_per_prompt=args.completions_per_prompt,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
