
import unicodedata
import re
from pathlib import Path

import unidecode


# In[17]:


specials_both = [['ä', 'ae'], ['ü', 'ue'], ['ö', 'oe']]
specials_lower = [['ß', 'ss']]
escape_sequence = 'xxxxx'

strange_quotes = ['«','‹','»','›','„','“','‟','‘','‛', '”', '’', '❛', '❜', '❝', '❞','❮', '❯', '〝','〞', '〟', '＂', '`', '´']


# In[18]:


# only use " for quotes
def clean_quotes(text):
    for q in strange_quotes + ["'"]:
        text = text.replace(q,'"')
    return text

# all whitespaces to single & trim
def clean_whitespaces(text):
    return re.sub(r'\s+', ' ', text).strip()


# In[19]:


def norm(text):
    return unicodedata.normalize('NFC', text)


# In[20]:


def save_replace(text, back=False):
    can = specials_lower + [[norm(x[0]), x[1]] for x in specials_both] +[[norm(x[0].upper()), x[1].upper()] for x in specials_both]
    for c, repl in can:
        if not back:
            text = text.replace(c, escape_sequence + repl + escape_sequence)
        else:
            text = text.replace(escape_sequence + repl + escape_sequence, c)
    return text


# In[21]:


def clean_german(text):
    text = norm(text) # slight preprocssing, do it here to make sure the replacement works
    text = save_replace(text)
    text = clean_quotes(text)
    text = unidecode.unidecode(text) # heavy preprocssing
    text = save_replace(text, back=True)
    text = clean_whitespaces(text)
    return text


