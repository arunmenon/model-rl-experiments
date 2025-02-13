# src/trainer_integration/custom_trainer.py

import random
from datasets import Dataset
from src.reward_functions.combined_reward import compute_total_reward
from typing import List, Dict

class CustomGRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        reward_func=compute_total_reward,
        output_dir="output",
        total_train_steps=1000,
        # ... other configs ...
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset  # expecting a huggingface Dataset or similar
        self.reward_func = reward_func
        self.output_dir = output_dir
        self.total_train_steps = total_train_steps
        # placeholders for config
        self.global_step = 0
    
    def generate_title(self, prompt: str) -> str:
        """
        Basic stub for generating a title from the model.
        Replace with your actual generate logic (transformers, top-k, etc.).
        """
        # naive approach: pretend we produce a random 'improved' string
        # in reality, call model.generate(...) or similar
        return prompt + " Enhanced Title"
    
    def compute_reward_batch(
        self, 
        prompts: List[str], 
        completions: List[str],
        reference_titles: List[str],
        category_keywords_list: List[List[str]],
        product_info_list: List[Dict]
    ) -> List[float]:
        """
        Compute rewards for a batch of prompts & completions by calling self.reward_func.
        The reward_func should handle or delegate to sub-rewards.
        """
        rewards = []
        for i, completion in enumerate(completions):
            ref_title = reference_titles[i] if reference_titles else None
            cat_keywords = category_keywords_list[i] if category_keywords_list else []
            product_info = product_info_list[i] if product_info_list else {}
            r = self.reward_func(
                generated_title=completion,
                reference_title=ref_title,
                category_keywords=cat_keywords,
                product_info=product_info,
                # optionally pass custom weights here
            )
            rewards.append(r)
        return rewards
    
    def train(self):
        """
        Simplified loop for demonstration.
        A real GRPO loop would have more steps: sampling multiple completions,
        computing advantages, updating the policy, etc.
        """
        dataset_len = len(self.dataset)
        for step in range(self.total_train_steps):
            # pick a random example from dataset
            idx = random.randint(0, dataset_len - 1)
            example = self.dataset[idx]

            prompt = example["prompt"]
            reference_title = example.get("reference_title", None)
            # parse category keywords (comma separated in CSV?)
            raw_keywords = example.get("category_keywords", "")
            if isinstance(raw_keywords, str):
                category_keywords = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
            else:
                category_keywords = raw_keywords or []
            
            product_info = {
                "brand": example.get("brand", ""),
                "product_type": example.get("product_type", ""),
                "material": example.get("material", ""),
                "color": example.get("color", ""),
                "size": example.get("size", "")
            }

            # Generate a 'completion' from the model
            generated = self.generate_title(prompt)

            # compute reward for this single sample 
            # (in a real RL setup, you'd do batch or multiple completions)
            reward = self.reward_func(
                generated_title=generated,
                reference_title=reference_title,
                category_keywords=category_keywords,
                product_info=product_info,
            )

            # This is where you'd do your policy gradient step, 
            # computing advantage, etc. We'll just log for demonstration.
            print(f"Step {step}, Prompt: {prompt}\nGenerated: {generated}\nReward: {reward:.3f}\n---")

            self.global_step += 1

            # Stopping or checkpoint logic can go here
        print("Training complete!")
