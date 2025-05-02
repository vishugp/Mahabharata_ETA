import pandas as pd
import numpy as np

OHCO = ['book_id','chap_id','sec_id','para_num', 'sent_num', 'token_num']

def bow_rep(tokens, choice):
    """
    The bow_rep function creates the Bag-Of-Words Represantation from a Token Level Dataframe

    Args:
        tokens (pd.DataFrame): DataFrame with all term_strings with OHCO levels as index
        choice (str): Choice of OHCO Level for Bag-Of-Words

    Returns:
        pd.DataFrame: Bag-Of-Words Representation at the chosen OHCO Level
    """
    bags = dict(
                SENTS = OHCO[:5],
                PARAS = OHCO[:4],
                SECS = OHCO[:3],
                CHAPS = OHCO[:2],
                BOOKS = OHCO[:1]
            )
    
    if choice not in bags.keys():
        raise ValueError("Invalid OHCO Level choice provided. Choose from ['book_id', 'chap_id', 'para_num', 'sent_num', 'token_num']")
    
    return tokens.groupby(bags[choice]+['term_str'])\
        .term_str.count().to_frame('n') 



def compute_TFIDF(bow, tf_type, idf_type = "standard"):
    """
    This function computes the TFIDF DataFrame from the Bag-Of-Words Representation

    Args:
        bow (pd.DataFrame): Bag-Of-Words Count at a specfic OHCO Level
        tf_type (str): Choice of Term Frequency type 

    Raises:
        ValueError: If the Choice of Term Frequency Type does not exist

    Returns:
        Tuple: TF, IDF and TFIDF DataFrames
    """
    DTCM = bow.n.unstack(fill_value=0)
    DF = DTCM.astype('bool').sum() 

    # Calculate TF based on the selected tf_type
    if tf_type == 'sum':
        TF = (DTCM.T / DTCM.T.sum()).T
    elif tf_type == 'max':
        TF = (DTCM.T / DTCM.T.max()).T
    elif tf_type == 'log':
        TF = (np.log2(1 + DTCM.T)).T
    elif tf_type == 'raw':
        TF = DTCM
    elif tf_type == 'double_norm':
        TF = (DTCM.T / DTCM.T.max()).T
    elif tf_type == 'binary':
        TF = DTCM.T.astype('bool').astype('int').T
    else:
        raise ValueError("Invalid tf_type provided. Choose from ['sum', 'max', 'log', 'raw', 'double_norm', 'binary']")
    

    idf = {
    'standard': np.log2(DTCM.shape[0] / DF),
    'max': np.log2(DF.max() / DF),
    'smooth': np.log2((1 + DTCM.shape[0]) / (1 + DF)) + 1
    }
    
    IDF = idf[idf_type]

    TFIDF = TF * IDF

    return {"TF":TF, 
            "IDF":IDF, 
            "TFIDF":TFIDF, 
            "DF":DF}
    