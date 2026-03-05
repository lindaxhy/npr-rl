from utils import normalize, extract_answer

def compute_reward(output_text, ground_truth):
    pred = normalize(extract_answer(output_text))
    gt = normalize(ground_truth)
    return 1.0 if pred == gt else 0.0