# src/reward_functions/semantic_similarity.py
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load the embedding model once at module level
EMBED_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def reward_semantic_similarity(generated_title: str, reference_title: str) -> float:
    """
    Returns a score in [0,1] measuring how semantically close generated_title is to reference_title.
    If reference_title is missing, returns 0.5 by default (fallback).
    """
    if not reference_title:
        return 0.5  # fallback score if no reference
    
    embeddings = EMBED_MODEL.encode([generated_title, reference_title])
    cos_sim = util.cos_sim(embeddings[0], embeddings[1])[0]
    # Normalize from [-1,1] to [0,1]
    score = float((cos_sim + 1.0) / 2.0)
    # Bound score in [0,1]
    score = max(0.0, min(1.0, score))
    return score
