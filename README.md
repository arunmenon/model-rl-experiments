# **Product Title Enhancement with a Custom Reward Framework**

This project demonstrates how to use **Group Relative Policy Optimization (GRPO)** (or a PPO-like reinforcement learning approach) to improve product titles automatically, guided by a **custom multi-component reward function**.

## **Table of Contents**

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Usage](#usage)
- [Reward Framework](#reward-framework)
  - [Semantic Similarity](#1-semantic-similarity)
  - [SEO Keywords](#2-seo-keywords)
  - [Grammar & Fluency](#3-grammar--fluency)
  - [Structure & Attribute Inclusion](#4-structure--attribute-inclusion)
  - [Length Optimization](#5-length-optimization)
  - [Combined Reward](#6-combined-reward)
- [How the Combined Reward Works](#how-the-combined-reward-works)
- [GRPO Training Integration](#grpo-training-integration)
- [Next Steps](#next-steps)
- [License](#license)

---

## **Overview**

Many online platforms struggle with **concise yet informative** product titles that attract customers and adhere to platform guidelines. This repository uses **Reinforcement Learning** (RL) to automatically generate enriched product titles from minimal inputs (like original titles, brand info, etc.). The **RL agent** learns to maximize a **custom reward** that measures key aspects of a “good” title:

1. **Maintaining Semantic Meaning** (so the title accurately reflects the product).  
2. **Including SEO Keywords** (to optimize search performance).  
3. **Preserving Grammar & Fluency** (for professionalism and readability).  
4. **Using Key Attributes** (brand, product type, color, size, etc.).  
5. **Staying within Optimal Length** (so it’s not truncated or too short).  

A **Group Relative Policy Optimization (GRPO)**-style RL algorithm is applied, generating multiple candidate titles per product and improving the policy by comparing their relative quality. 

---

## **Repository Structure**

```
product-title-enhancement-rl/
├── .gitignore
├── LICENSE
├── README.md                # You're here!
├── requirements.txt         # Python dependencies
├── data/
│   └── example_dataset.csv  # Sample dataset
├── docs/                    # Additional documentation
├── notebooks/
│   └── Demo_Exploration.ipynb # For experimentation
├── scripts/
│   └── run_training.sh      # Shell script to run training
├── src/
│   ├── reward_functions/
│   │   ├── semantic_similarity.py
│   │   ├── seo_keywords.py
│   │   ├── grammar_fluency.py
│   │   ├── structure_inclusion.py
│   │   ├── length_optimization.py
│   │   └── combined_reward.py # Aggregates all sub-rewards
│   ├── trainer_integration/
│   │   ├── custom_grpo_trainer.py
│   │   └── training_config.py
│   └── utils/
│       └── data_utils.py    # Optional data utilities
├── tests/
│   ├── test_semantic_similarity.py
│   ├── test_seo_keywords.py
│   ├── test_grammar_fluency.py
│   ├── test_structure_inclusion.py
│   ├── test_length_optimization.py
│   └── test_combined_reward.py
└── train.py                 # Main script to start training
```

- **`src/reward_functions/`**: Contains individual reward metrics + combined aggregator.
- **`src/trainer_integration/`**: Houses the **CustomGRPOTrainer** stub and training configs.
- **`train.py`**: Entrypoint script that loads data, initializes the trainer, and starts training.
- **`tests/`**: Unit tests for verifying each reward component and combined scoring.

---

## **Getting Started**

### **Installation**

1. **Clone** this repo:
   ```bash
   git clone https://github.com/YourOrg/product-title-enhancement-rl.git
   cd product-title-enhancement-rl
   ```
2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   - This includes libraries like `transformers`, `sentence_transformers`, `language_tool_python`, and more.

### **Data Preparation**

- The example dataset (`example_dataset.csv`) has columns:
  - `prompt`: An original or minimal product title.
  - `reference_title`: A high-quality or human-curated title (optional).
  - `brand`, `product_type`, `material`, `color`, `size`: Key attributes.
  - `category_keywords`: Comma-separated SEO keywords.
- You can **replace** or **extend** this CSV with your own data. Just ensure columns match what the reward functions expect (`brand`, `reference_title`, etc.).

### **Usage**

1. **Run** training via `train.py`:
   ```bash
   python train.py \
       --model_name gpt2 \
       --dataset_path data/example_dataset.csv \
       --output_dir output \
       --num_train_steps 50
   ```
   or **use** the provided script:
   ```bash
   bash scripts/run_training.sh
   ```
2. The trainer logs reward values to stdout. You can observe how the average reward changes over iterations.

---

## **Reward Framework**

Below is a deep dive into each **individual reward function**. Each sub-reward outputs a **float in [0,1]**, with **1** = fully optimal, **0** = poor or missing. The final aggregated reward is also **clamped** to [0,1].

### **1. Semantic Similarity**

**File**: `semantic_similarity.py`  
We measure how close the generated title is in meaning to the `reference_title`, using **sentence embeddings** (e.g., `all-mpnet-base-v2`). A **cosine similarity** is computed, then normalized from `[-1,1]` to `[0,1]`. This ensures the model **retains core meaning** of the product.  
- **If no reference** is available, we use a fallback (like 0.5) so the model isn’t overly penalized.

### **2. SEO Keywords**

**File**: `seo_keywords.py`  
We check if the generated title includes **important category keywords** (e.g., "running", "women", "cotton") that boost search ranking.  
- A base score counts how many unique keywords appear.  
- A small bonus if at least one keyword is found.  
- **Penalty** for **keyword stuffing** (repeated usage of the same term).

### **3. Grammar & Fluency**

**File**: `grammar_fluency.py`  
We run a **LanguageTool** grammar/spelling check. Titles are penalized for each detected error and for excessive punctuation or bizarre punctuation usage (e.g., multiple exclamation marks). This encourages **professional, easy-to-read** titles.

### **4. Structure & Attribute Inclusion**

**File**: `structure_inclusion.py`  
We reward the presence of **Brand**, **Product Type**, and essential attributes (like color/size). We may add a small bonus if brand appears **before** product type. Missing key elements leads to negative scoring. This ensures titles follow a structured **“Brand + Product Type + Key Attributes”** pattern.

### **5. Length Optimization**

**File**: `length_optimization.py`  
Product titles should be **concise** but **informative**. We encourage lengths roughly in the **50-100 character range**:
- Titles below ~30 or above ~120 get a **0**. 
- Perfect length (50-100) gets **1**. 
- Gradual scaling for those in-between.

### **6. Combined Reward**

**File**: `combined_reward.py`  
We call each sub-reward function in turn, then combine them with a **configurable weighted sum**:
```python
total_reward = (
  w['semantic'] * semantic_score +
  w['seo']      * seo_score +
  w['grammar']  * grammar_score +
  w['structure']* struct_score +
  w['length']   * length_score
)
```
By default, each component might have a weight around **(semantic=0.25, seo=0.15, grammar=0.20, structure=0.25, length=0.15)**, but these can be **tuned** as you see fit. The combined reward is **clamped** to `[0,1]`. 

---

## **How the Combined Reward Works**

When the model (RL agent) generates a candidate product title:

1. **Compute Sub-Scores**:  
   - **Semantic**: Embedding-based alignment with a reference.  
   - **SEO**: Checks for relevant keywords.  
   - **Grammar**: LanguageTool errors and punctuation usage.  
   - **Structure**: Looks for brand, product type, color, etc.  
   - **Length**: Evaluates if it falls in the sweet spot.

2. **Weighted Sum**:  
   Each sub-score is multiplied by its **weight**. Summing them yields a final reward. For example, if the model nails meaning, structure, and length but misses grammar, the final reward might be around `0.8`. If it misses brand entirely, the structure portion might be penalized, dropping the final reward significantly.

3. **RL Update**:  
   In **GRPO** or **PPO**, the policy sees a **higher reward** for better titles, guiding the model to replicate those strategies. Over many steps, the model learns to produce titles that **balance** all these objectives.

---

## **GRPO Training Integration**

We provide a **`custom_grpo_trainer.py`** that demonstrates how to:

- **Generate multiple completions** per prompt (`completions_per_prompt`).
- **Evaluate** each completion with `compute_total_reward`.
- **Compare** their relative performance. (In a real GRPO approach, we’d compute policy gradients from these comparisons.)

A typical training loop:

1. **Sample a batch** of data (prompts + product attributes).  
2. **Generate** multiple candidate titles for each prompt.  
3. **Compute Reward** for each candidate using `compute_total_reward`.  
4. **Rank** them or compute advantages (the best titles have higher reward).  
5. **Update** the model parameters to **increase probability** of higher-reward outputs.  

During training, you will see logs like:

```
Prompt: Basic Shoe
   Completion: Basic Shoe - Enhanced Variation 0 => Reward: 0.65
   Completion: Basic Shoe - Enhanced Variation 1 => Reward: 0.72
   Best completion: Basic Shoe - Enhanced Variation 1 => 0.72
```

The **best** completion’s strategy is then reinforced. Over time, the model converges towards titles that rank well on our multi-component reward.

---

## **Next Steps**

- **Tune Weights**: 
  - Adjust the sub-reward weights in `combined_reward.py` to emphasize certain criteria (e.g., more weight on grammar if you see frequent mistakes). 
- **Add More Metrics**: 
  - For instance, brand correctness vs. brand guess, or advanced readability checks. 
---

