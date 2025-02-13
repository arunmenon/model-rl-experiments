# src/reward_functions/length_optimization.py

def reward_length(generated_title: str) -> float:
    """
    Encourages titles within ~50..100 chars, penalizes outside that range.
    """
    stripped = generated_title.strip()
    length = len(stripped)
    if length == 0:
        return 0.0
    if length < 30 or length > 120:
        return 0.0
    if 50 <= length <= 100:
        return 1.0
    if length < 50:
        return (length - 30) / 20 * 0.9
    # else length in 101..120
    return (120 - length) / 20 * 0.9
