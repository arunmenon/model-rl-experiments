# src/reward_functions/combined_reward.py

from .semantic_similarity import reward_semantic_similarity
from .seo_keywords import reward_seo_keywords
from .grammar_fluency import reward_grammar_fluency
from .structure_inclusion import reward_title_structure
from .length_optimization import reward_length

def compute_total_reward(
    generated_title: str,
    reference_title: str = None,
    category_keywords=None,
    product_info=None,
    weights=None
) -> float:
    """
    Combined reward aggregator for product title enhancement.
    Returns a final score in [0,1].
    """
    default_weights = {
        'semantic': 0.25,
        'seo': 0.15,
        'grammar': 0.20,
        'structure': 0.25,
        'length': 0.15
    }
    if weights is None:
        weights = default_weights

    if category_keywords is None:
        category_keywords = []
    if product_info is None:
        product_info = {}

    r_sem = reward_semantic_similarity(generated_title, reference_title)
    r_seo = reward_seo_keywords(generated_title, category_keywords)
    r_gram = reward_grammar_fluency(generated_title)
    r_struct = reward_title_structure(generated_title, product_info)
    r_len = reward_length(generated_title)

    total = (
        weights['semantic'] * r_sem +
        weights['seo'] * r_seo +
        weights['grammar'] * r_gram +
        weights['structure'] * r_struct +
        weights['length'] * r_len
    )
    return max(0.0, min(1.0, total))
