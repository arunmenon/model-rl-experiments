import pytest
from src.reward_functions.semantic_similarity import reward_semantic_similarity

def test_semantic_similarity_identical():
    gen = "Nike Men's Running Shoe, Black, Size 10"
    ref = "Nike Men's Running Shoe, Black, Size 10"
    score = reward_semantic_similarity(gen, ref)
    assert score >= 0.95

def test_semantic_similarity_different():
    gen = "Apple iPhone 13, 128GB"
    ref = "Men's Running Shoe"
    score = reward_semantic_similarity(gen, ref)
    assert score < 0.5
