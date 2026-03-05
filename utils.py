import re

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z]', '', text)
    return text

def extract_answer(text):
    match = re.search(r'answer\s*:\s*([A-Za-z]+)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return text.strip().split()[-1]