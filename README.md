# Product Title Enhancement with RL

This repository implements a custom reward framework and GRPO-based training to enhance product titles for e-commerce platforms. 

## Project Overview

- **src/reward_functions**: Individual modules implementing sub-rewards (semantic similarity, SEO, grammar, etc.).
- **src/trainer_integration**: Integration code with a custom GRPO trainer.
- **train.py**: High-level script to run RL training on product titles.

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
