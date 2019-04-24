import re
from pathlib import Path
from textacy.preprocess import preprocess_text

strange_double_quotes = ['«','‹','»','›','„','“','‟', '”', '❝', '❞','❮', '❯', '〝','〞', '〟', '＂']
strange_single_quotes = ['‘', '‛', '’', '❛', '❜', '`', '´']

# only use ' for quotes
def clean_quotes(text):
    text = str(text)
    for q in strange_double_quotes:
        text = text.replace(q,'"')
    for q in strange_single_quotes:
        text = text.replace(q, "'")
    return text


def clean(text, lower=True, **kwargs):
    text = clean_quotes(text)
    text = preprocess_text(text, fix_unicode=True, lowercase=lower, no_urls=True, no_emails=True, transliterate=True, no_numbers=True, no_phone_numbers=True)
    return text


