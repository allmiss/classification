import re

LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = text.replace('``', ' ')
    text = text.replace("''", ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+([,.!?;:)])', r'\1', text)
    text = re.sub(r"\s+'s\b", "'s", text)
    return text

def preprocess_text(text):
    text = clean_text(text)
    text = normalize_text(text)
    return text

if __name__ == '__main__':
    sample = "For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m ."
    print(preprocess_text(sample))
