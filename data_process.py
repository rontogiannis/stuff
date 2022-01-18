import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

VARIANT = 'SARS-CoV2_WT'
FEATURE = 'VL'
BINDS = 'Binds to'
NEUT = 'Neutralising Vs'
NODATA = 'ND'

def contains(label, key) :
    if isinstance(label, str) == False or isinstance(key, str) == False :
        return 0
    label_modified = ''.join(re.split(' |_|-', label)).upper()
    key_modified   = ''.join(re.split(' |_|-', key)).upper()
    return 1 if key_modified in label_modified else 0

def add_spaces(seq) :
    if isinstance(seq, str) == False :
        return ''
    return ' '.join(seq)
    
def nonempty(s) :
    if isinstance(s, str) == False :
        return False
    return s.strip() != ''
    
def process_data(train_test_ratio=0.8, train_valid_ratio=0.8) :
    data[BINDS] = data[BINDS].apply(contains, key=VARIANT)
    data[NEUT] = data[NEUT].apply(contains, key=VARIANT)    
    data[BINDS] = data[BINDS] | data[NEUT]
    
    df = data[[FEATURE, BINDS, NEUT]]
    df = df[df[FEATURE].apply(nonempty) == True]
    df = df[df[FEATURE] != NODATA]
    df = df.rename(columns={FEATURE: 'seq', BINDS: 'binds', NEUT: 'neut'})
    df['seq'] = df['seq'].apply(add_spaces)
    
    df = df[df['binds'] == 1]
    df = df[['seq', 'neut']]
    
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('df.csv')
    
    df_full_train, df_test = train_test_split(df, train_size=train_test_ratio, random_state=1)
    df_train, df_valid = train_test_split(df_full_train, train_size=train_valid_ratio, random_state=1)
    
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_valid = df_valid.sample(frac=1).reset_index(drop=True)
    
    df_train = df_train.set_index('seq')
    df_test = df_test.set_index('seq')
    df_valid = df_valid.set_index('seq')
    
    df_train.to_csv('train.csv')
    df_test.to_csv('test.csv')
    df_valid.to_csv('valid.csv')
    
data = pd.read_csv('covabdab_search_results.csv')
process_data()
    


    
