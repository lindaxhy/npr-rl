import re
from typing import List


def _parse_answer(text: str) -> List[List[str]]:
    """
    Same parsing as the official verbal reasoning challenge:
    - ";" separates alternative acceptable answers
    - "," and "-->" separate phrases within an alternative (any order)
    """
    text = text.lower()
    alternatives = re.split(r";", text)
    result = []
    for alternative in alternatives:
        groups = re.split(r"–?-?-?>|,", alternative)
        result.append([" ".join(re.findall(r"\b\w+\b", group)) for group in groups])
    return result


def _answer_without_thoughts(completion: str) -> str:
    completion = re.sub(r"(<think>)?[^<]*<\/think>", "", completion).strip()
    return completion


def _check_answer(completion: str, answer: str) -> bool:
    """
    Phrase-based matching: all phrases in one alternative must appear in the
    completion (any order). Ignores thoughts, case, and punctuation.
    """
    completion = _answer_without_thoughts(completion).lower()
    completion = re.sub(r"[^\w\s]", " ", completion)
    completion = re.sub(r"\s+", " ", completion)
    alternative_answers = _parse_answer(answer)
    for answer_phrases in alternative_answers:
        if all(re.search(rf"\b{re.escape(phrase)}\b", completion) for phrase in answer_phrases):
            return True
    return False


def compute_reward(output_text, ground_truth):
    """Return 1.0 if output_text matches ground_truth under phrase-based rules, else 0.0."""
    return 1.0 if _check_answer(output_text, ground_truth) else 0.0