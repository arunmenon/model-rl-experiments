# src/trainer_integration/custom_grpo_trainer.py
import random
import torch
from typing import List, Dict
from datasets import Dataset
from src.reward_functions.combined_reward import compute_total_reward
from transformers import PreTrainedModel, PreTrainedTokenizer

class CustomGRPOTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dataset,
        reward_func=compute_total_reward,
        output_dir="output",
        total_train_steps=1000,
        batch_size=2,
        completions_per_prompt=2,
        # ...
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_func = reward_func
        self.output_dir = output_dir
        self.total_train_steps = total_train_steps
        self.batch_size = batch_size
        self.completions_per_prompt = completions_per_prompt

        # For real RL, you'd store optimizers, policy config, etc.
        # self.optimizer = ...
        self.global_step = 0
    
    def generate_titles(self, prompt: str, num_completions: int = 2) -> List[str]:
        """
        A placeholder for multi-completion generation from the model.
        In a real scenario, you'd do something like:
        model.generate(..., num_return_sequences=num_completions, do_sample=True)
        """
        # For demonstration, we'll just produce random strings
        outputs = []
        for i in range(num_completions):
            # This is a naive placeholder
            outputs.append(f"{prompt} - Enhanced Variation {i}")
        return outputs
    
    def compute_rewards_for_batch(
        self, 
        prompts: List[str],
        reference_titles: List[str],
        category_keywords_list: List[List[str]],
        product_info_list: List[Dict],
    ) -> List[List[float]]:
        """
        For each prompt, we generate multiple completions and compute their rewards.
        Returns a 2D list of shape [batch_size, completions_per_prompt].
        """
        batch_rewards = []
        for i, prompt in enumerate(prompts):
            ref = reference_titles[i] if reference_titles else None
            ckeywords = category_keywords_list[i] if category_keywords_list else []
            pinfo = product_info_list[i] if product_info_list else {}

            completions = self.generate_titles(prompt, self.completions_per_prompt)
            # compute reward for each completion
            comp_rewards = []
            for comp in completions:
                r = self.reward_func(
                    generated_title=comp,
                    reference_title=ref,
                    category_keywords=ckeywords,
                    product_info=pinfo,
                )
                comp_rewards.append(r)
            batch_rewards.append(comp_rewards)
        return batch_rewards

    def train(self):
        """
        Simplified training loop demonstrating multiple completions per prompt,
        then computing rewards. Real GRPO would update policy based on relative rewards.
        """
        data_len = len(self.dataset)
        steps = 0
        while steps < self.total_train_steps:
            # We'll sample a batch
            batch_indices = random.sample(range(data_len), min(self.batch_size, data_len))
            prompts, refs, cat_keywords_list, pinfo_list = [], [], [], []
            
            for idx in batch_indices:
                example = self.dataset[idx]
                prompts.append(example["prompt"])
                refs.append(example.get("reference_title", None))
                # handle category_keywords as list
                raw_kw = example.get("category_keywords", "")
                if isinstance(raw_kw, str):
                    cat_keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
                else:
                    cat_keywords = raw_kw or []
                cat_keywords_list.append(cat_keywords)

                product_info = {
                    "brand": example.get("brand", ""),
                    "product_type": example.get("product_type", ""),
                    "material": example.get("material", ""),
                    "color": example.get("color", ""),
                    "size": example.get("size", ""),
                }
                pinfo_list.append(product_info)
            
            # Generate completions, compute rewards
            batch_rewards = self.compute_rewards_for_batch(prompts, refs, cat_keywords_list, pinfo_list)
            # batch_rewards is shape [batch_size, completions_per_prompt]
            # e.g., batch_rewards[i][j] => reward for j-th completion of i-th sample
            
            # In real GRPO: compute advantage for each completion relative to others,
            # update the policy. We'll just log them for demonstration.
            for i, rew_list in enumerate(batch_rewards):
                print(f"Prompt: {prompts[i]}")
                comps = self.generate_titles(prompts[i], self.completions_per_prompt)
                for c_i, r in enumerate(rew_list):
                    print(f"   Completion: {comps[c_i]} => Reward: {r:.3f}")
                best_c_i = max(range(len(rew_list)), key=lambda x: rew_list[x])
                print(f"   Best completion: {comps[best_c_i]} => {rew_list[best_c_i]:.3f}\n---")

            steps += 1
        print("Training finished with a demonstration of multi-completion reward calculation!")
