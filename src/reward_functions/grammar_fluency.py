# src/reward_functions/grammar_fluency.py
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def reward_grammar_fluency(generated_title: str) -> float:
    """
    Rewards grammatical correctness and penalizes excessive punctuation or errors.
    """
    # Start perfect, deduct for issues
    score = 1.0
    if not generated_title.strip():
        return 0.0  # empty or whitespace only => 0

    # Grammar/spelling check via LanguageTool
    matches = tool.check(generated_title)
    # Deduct 0.1 for each match found (capped at some point)
    penalty = 0.1 * len(matches)
    score -= penalty

    # Check for repeated punctuation
    if "!!" in generated_title or "??" in generated_title:
        score -= 0.2

    # Check total punctuation usage
    punct_chars = ".,;:!?|-"
    punct_count = sum(ch in punct_chars for ch in generated_title)
    if punct_count > 5:
        # subtract 0.05 for each punctuation above 5
        score -= 0.05 * (punct_count - 5)

    score = max(0.0, min(1.0, score))
    return score
