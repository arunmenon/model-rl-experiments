# tests/test_combined_reward.py
import pytest
from src.reward_functions.combined_reward import compute_total_reward

def test_combined_reward_basic():
    gen_title = "Nike Men's Running Shoe, Black, Size 10"
    ref_title = "Nike Men's Running Shoe, Black, Size 10"
    category_keywords = ["nike", "shoe", "running", "black"]
    product_info = {
        "brand": "Nike",
        "product_type": "Shoe",
        "color": "Black",
        "material": "Mesh",
        "size": "10"
    }
    score = compute_total_reward(
        generated_title=gen_title,
        reference_title=ref_title,
        category_keywords=category_keywords,
        product_info=product_info
    )
    assert 0.9 <= score <= 1.0, f"Expected a high reward, got {score}"
