# src/reward_functions/seo_keywords.py

def reward_seo_keywords(generated_title: str, category_keywords: list[str]) -> float:
    """
    Reward presence of relevant keywords, penalize overuse.
    Returns a float in [0,1].
    """
    if not category_keywords:
        return 0.5  # fallback score
    
    title_lower = generated_title.lower()
    found_keywords = []
    for kw in category_keywords:
        kw_l = kw.lower().strip()
        if kw_l in title_lower:
            found_keywords.append(kw_l)
    
    unique_count = len(set(found_keywords))
    base_score = unique_count / len(category_keywords)
    
    # Small bonus if at least 1 keyword found
    score = base_score
    if unique_count > 0:
        score += 0.1
    
    # Penalty for repeated usage of the same keyword
    for kw_l in set(found_keywords):
        occurrences = title_lower.count(kw_l)
        if occurrences > 1:
            score -= 0.1 * (occurrences - 1)
    
    return max(0.0, min(1.0, score))
