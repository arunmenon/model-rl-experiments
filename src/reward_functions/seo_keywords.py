# src/reward_functions/seo_keywords.py

def reward_seo_keywords(generated_title: str, category_keywords: list[str]) -> float:
    """
    Rewards the presence of relevant category keywords (passed via category_keywords),
    penalizes excessive repetition.
    """
    if not category_keywords:
        return 0.5  # fallback if no keywords
    
    title_lower = generated_title.lower()
    keywords_found = []
    for kw in category_keywords:
        kw_lower = kw.strip().lower()
        if kw_lower in title_lower:
            keywords_found.append(kw_lower)

    unique_count = len(set(keywords_found))
    # Base score: ratio of found keywords over total available
    base_score = unique_count / len(category_keywords)
    score = base_score

    # Bonus for using at least one
    if unique_count > 0:
        score += 0.1

    # Penalty for repetition of the same keyword
    for kw_l in set(keywords_found):
        occurrences = title_lower.count(kw_l)
        # If keyword repeated more than once, penalize
        if occurrences > 1:
            score -= 0.1 * (occurrences - 1)

    # Bound final score
    score = max(0.0, min(1.0, score))
    return score
