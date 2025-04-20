import pandas as pd 
import numpy as np 
import configparser
import os
OHCO = ['book_id','chap_num', 'para_num', 'sent_num', 'token_num']

def create_tokendf(filepath):
    ### READING

    # Reading the TextFile Line by Line and saving as Dataframe
    LINES = pd.DataFrame(open(filepath, 'r', 
                              encoding='utf-8-sig').readlines(), columns=['line_str'])
    LINES.index.name = 'line_num'
    LINES.line_str = LINES.line_str.str.replace(r'\n+', ' ', regex=True).str.strip()




    #### Clipping the Cruft

    # Using RegEx to find the placeholders for Start & End of Text
    clip_pats = [
        r"(?i)^om\b",
        r"(?i)(?=.*\bend\b)(?=.*\bparv\w*)"
    ]

    # Getting the Pattern matches for both
    pat_a = LINES.line_str.str.match(clip_pats[0])
    pat_b = LINES.line_str.str.match(clip_pats[1])

    # Getting the line number
    line_a = LINES.loc[pat_a].index[0] 
    line_b = LINES.loc[pat_b].index[-1] - 2
    LINES = LINES.loc[line_a : line_b]





    #### Getting the Sections/Chapters

    chap_pat = r"^\s*(?:SECTION)+"
    num_pat = r"^\s*\d+\s*$"
    chap_lines = LINES.line_str.str.match(chap_pat, case=True) | LINES.line_str.str.match(num_pat, case=True)

    # Creating a chap_num column as the Chapter Number Index starting from 1
    LINES.loc[chap_lines, 'chap_num'] = [int(i+2) for i in range(LINES.loc[chap_lines].shape[0])]
    # Forward Fill to fill the Chapter Lines with the Chapter Number
    LINES.chap_num = LINES.chap_num.ffill()
    # Relabeling Lines before Chapter 2 that are Chapter 1
    LINES.loc[:LINES.loc[chap_lines].index[0],"chap_num"] = 1
    # Removing Chapter Headers now
    LINES = LINES.loc[~chap_lines]
    # Making the Chapter Numbers Integer Type
    LINES.chap_num = LINES.chap_num.astype('int')


    # Grouping by chap_num and concatenating using \n
    CHAPS = LINES.groupby(OHCO[1:2])\
        .line_str.apply(lambda x: '\n'.join(x))\
        .to_frame('chap_str')                      

    # Cleaning trailing newlines
    CHAPS['chap_str'] = CHAPS.chap_str.str.strip()





    #### PARAGRAPHS
    # RegEx for each paragraph
    para_pat = r'\n\n+'
    PARAS = CHAPS['chap_str'].str.split(para_pat, expand=True)\
        .stack()\
        .to_frame('para_str')\
        .sort_index()

    PARAS.index.names = OHCO[1:3]

    PARAS['para_str'] = PARAS['para_str'].str.replace(r'\n', ' ', regex=True)\
                                        .str.strip()

    PARAS = PARAS[~PARAS['para_str'].str.match(r'^\s*$')]
    PARAS.sample(20)




    #### SENTENCES
    # RegEx for each line ending
    sent_pat = r'[.?!;:]+'
    SENTS = PARAS['para_str'].str.split(sent_pat, expand=True).stack()\
        .to_frame('sent_str')
    SENTS.index.names = OHCO[1:4]

    SENTS = SENTS[~SENTS['sent_str'].str.match(r'^\s*$')] 
    SENTS.sent_str = SENTS.sent_str.str.strip() 

    SENTS.head(25)




    #### TOKEN LEVEL
    # RegEx to Split by space, hyphen or comma
    token_pat = r"[\s',-]+"
    TOKENS = SENTS['sent_str'].str.split(token_pat, expand=True)\
        .stack()\
        .to_frame('token_str')

    TOKENS.index.names = OHCO[1:5]

    TOKENS['term_str'] = TOKENS.token_str.replace(r'[\W_]+', '', regex=True).str.lower()

    TOKENS.head(25)

    return dict(zip(OHCO[1:], [CHAPS, PARAS, SENTS, TOKENS]))
