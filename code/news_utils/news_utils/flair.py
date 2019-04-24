


from flair.models import SequenceTagger
from flair.data import Sentence
import swifter
import unidecode
import spacy
import pandas as pd
from pathlib import Path
import re



strange_quotes = ['«','‹','»','›','„','“','‟','‘','‛', '”', '’', '❛', '❜', '❝', '❞','❮', '❯', '〝','〞', '〟', '＂', '`', '´']

def clean_quotes(df_col):
    for q in strange_quotes + ['"']:
        df_col = df_col.str.replace(q,"'")
    return df_col




def clean_whitespaces(df_col):
    return df_col.str.replace(r'\s+', ' ').str.strip()



def replace_numbers(df_col, token=" xxnumber "):
    return df_col.replace(r"^\d+\s|\s\d+\s|\s\d+$", " xxnumber ")



def replace_ner_in_sent(sent):
    dic = sent.to_dict(tag_type='ner')
    text = dic['text']
    if 'entities' in dic:
        offset = 0
        for ent in dic['entities']:
            len_before = len(text)
            start = ent['start_pos'] + offset
            end = ent['end_pos'] + offset
            text = text[:start] + ' xx' + ent['type'].lower() + ' ' + text[end:]
            len_after = len(text)
            offset += - len_before + len_after
    return text



def replace_ner(t, nlp, tagger):
    t = unidecode.unidecode(t) # TODO: fix for German
    sents = []
    for s in nlp(t).sents:
        sents.append(Sentence(str(s), use_tokenizer=False)) # use_tokenizer important because the text is not whitespace tokenized
    tagger.predict(sents, mini_batch_size=64)
    proc_txt = ' '.join([replace_ner_in_sent(s) for s in sents])
    return proc_txt



def preprocess_df(df, input_col='text', output_col='text_proc'):
    
    nlp = spacy.load('en_core_web_lg', disable=['ner'])

    tagger = SequenceTagger.load('ner-ontonotes')
    
#   clean
    df[output_col] = df[input_col]
    df[output_col] = clean_quotes(df[output_col])
    df[output_col] = clean_whitespaces(df[output_col])
    
#   NER
    df[output_col] = df[output_col].swifter.apply(lambda x: replace_ner(x, nlp, tagger))
    
#   clean
    df[output_col] = replace_numbers(df[output_col])
    df[output_col] = clean_whitespaces(df[output_col]) # the number adds some spaces again

    return df



def peprocess_text(input_path, output_path, **kwargs):
    df = pd.read_csv(input_path, **kwargs)
    df = preprocess_df(df)
    df.to_csv(output_path)


# util to print unicodes
def print_unicode(s):
    for _c in s:
        print(_c)
        print('U+%04x' % ord(_c))






