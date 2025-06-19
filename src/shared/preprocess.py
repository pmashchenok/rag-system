import pymorphy2
import re


morph = pymorphy2.MorphAnalyzer()


def preprocess_text(original_text):
    """Предобработка текста"""
    text = original_text.lower()
    text = re.sub(r"[^а-яa-zё0-9\s]", "", text, flags=re.IGNORECASE | re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    processed_text = ' '.join(lemmas)

    if not processed_text:
        return None

    return {
        'cleaned_text': processed_text,
        'original_text': original_text
    }
