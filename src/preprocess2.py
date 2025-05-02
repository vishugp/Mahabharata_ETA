import pandas as pd 
import numpy as np 
import configparser
import os
import re
import nltk

for package in [
                'tokenizers/punkt', 
                'taggers/averaged_perceptron_tagger', 
                'corpora/stopwords', 
                'help/tagsets'
                ]:
    try:
        nltk.data.find(package)
    except IndexError:
        nltk.download(package)

OHCO = ['book_id','chap_id','sec_id', 'para_num', 'sent_num', 'token_num']

def create_tokendf(filepath):
    ### READING
    LINES = pd.DataFrame(open(filepath, 'r', encoding='utf-8-sig').readlines(), columns=['line_str'])
    LINES = LINES[~LINES['line_str'].apply(lambda x: str(x).strip()=="")].reset_index(drop=True)
    LINES.index.name = 'line_num'
    LINES.line_str = LINES.line_str.str.replace(r'\n+', ' ', regex=True).str.strip()

    book_id_pat = LINES.line_str.str.match(r"BOOK [\d]")
    title = LINES.iloc[LINES.loc[book_id_pat].index[0]+1,0]
    title = re.sub("[-]"," ",title.title())

    #### CLIPPING THE CRUFT
    clip_pats = [
        r"(?i)^om\b",
        r"(?i)(?=.*\bend\b)(?=.*\bparv\w*)"
    ]

    pat_a = LINES.line_str.str.match(clip_pats[0])
    pat_b = LINES.line_str.str.match(clip_pats[1])

    line_a = LINES.loc[pat_a].index[0] - 1
    line_b = LINES.loc[pat_b].index[-1] - 1
    LINES = LINES.loc[line_a : line_b]

    #### SECTION NUMBERS
    sec_pat = r"^\s*(?:SECTION)+"
    num_pat = r"^\s*\d+\s*$"
    sec_lines = LINES.line_str.str.match(sec_pat, case=True) | LINES.line_str.str.match(num_pat, case=True)

    LINES.loc[sec_lines, 'sec_id'] = [int(i + 1) for i in range(LINES.loc[sec_lines].shape[0])]
    LINES.sec_id = LINES.sec_id.ffill()
    LINES.loc[:LINES.loc[sec_lines].index[0], "sec_id"] = 1
    LINES = LINES.loc[~sec_lines]
    LINES.sec_id = LINES.sec_id.astype(int)

    #### PARVA HEADINGS (chap_id)
    parva_pat = r"\s*([^()]*?\s+parva)\)$"
    LINES['chap_name'] = LINES.line_str.str.extract(parva_pat, flags=re.IGNORECASE, expand=False)
    LINES['chap_id'] = ((LINES['chap_name'].notna()).cumsum()) 
    LINES['chap_name'] = LINES['chap_name'].ffill()

    #### CHAP (PARVA) LEVEL
    CHAPS = LINES.groupby(['chap_id'])\
        .line_str.apply(lambda x: '\n'.join(x))\
        .to_frame('chap_str')                      
    CHAPS['chap_str'] = CHAPS.chap_str.str.strip()
    CHAPS['chap_name'] = LINES.groupby('chap_id')['chap_name'].first().values

    #### SECTION LEVEL
    SECTIONS = LINES.groupby(['chap_id', 'sec_id'])\
        .line_str.apply(lambda x: '\n'.join(x))\
        .to_frame('section_str')
    SECTIONS['section_str'] = SECTIONS['section_str'].str.strip()

    #### PARAGRAPHS
    para_pat = r'\n\n+'
    PARAS = SECTIONS['section_str'].str.split(para_pat, expand=True)\
        .stack()\
        .to_frame('para_str')\
        .sort_index()

    PARAS.index.names = OHCO[1:4]

    PARAS['para_str'] = PARAS['para_str'].str.replace(r'\n', ' ', regex=True)\
                                        .str.strip()
    PARAS = PARAS[~PARAS['para_str'].str.match(r'^\s*$')]

    #### SENTENCES
    sent_pat = r'[.?!;:]+'
    SENTS = PARAS['para_str'].str.split(sent_pat, expand=True).stack()\
        .to_frame('sent_str')
    SENTS.index.names = OHCO[1:5]

    SENTS = SENTS[~SENTS['sent_str'].str.match(r'^\s*$')] 
    SENTS.sent_str = SENTS.sent_str.str.strip() 

    #### TOKEN LEVEL
    TOKENS = SENTS.sent_str.apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x.replace("'", "")))))
    TOKENS = TOKENS.stack().to_frame('pos_tuple')
    TOKENS['pos'] = TOKENS.pos_tuple.apply(lambda x: x[1])
    TOKENS['token_str'] = TOKENS.pos_tuple.apply(lambda x: x[0])
    TOKENS['term_str'] = TOKENS.token_str.str.lower().apply(lambda x: re.sub("[^a-zA-Z0-9]", "", x))   
    TOKENS.index.names = OHCO[1:]

    return dict(zip(OHCO, [title, CHAPS, SECTIONS, PARAS, SENTS, TOKENS]))
