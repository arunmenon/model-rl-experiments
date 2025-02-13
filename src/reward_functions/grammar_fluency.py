# src/reward_functions/grammar_fluency.py
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def reward_grammar_fluency(generated_title: str) -> float:
    """
    Returns ~1.0 for grammatically clean titles, penalizes errors or excessive punctuation.
    """
    if not generated_title.strip():
        return 0.0  # empty
    
    score = 1.0
    matches = tool.check(generated_title)
    score -= 0.1 * len(matches)  # penalty per detected error
    
    # Check punctuation or special characters
    if "!!" in generated_title or "??" in generated_title:
        score -= 0.2

    punct_chars = ".,;:!?|-"
    punct_count = sum(ch in punct_chars for ch in generated_title)
    if punct_count > 5:
        score -= 0.05 * (punct_count - 5)
    
    return max(0.0, min(1.0, score))
