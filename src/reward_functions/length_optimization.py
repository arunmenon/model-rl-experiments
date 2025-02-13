# src/reward_functions/length_optimization.py

def reward_length(generated_title: str) -> float:
    """
    Encourage length in ~[50..100] chars, penalize too short/long.
    """
    length = len(generated_title.strip())
    if length == 0:
        return 0.0

    # <30 => 0.0, >120 => 0.0
    if length < 30 or length > 120:
        return 0.0
    # Optimal: 50..100 => 1.0
    if 50 <= length <= 100:
        return 1.0
    # If in 30..49 => scale from 0..1
    if length < 50:
        return (length - 30) / (50 - 30) * 0.9
    # If in 101..120 => scale
    return (120 - length) / (120 - 100) * 0.9
