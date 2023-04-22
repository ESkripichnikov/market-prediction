import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    if 'target' in df.columns:
        df['target'] = df['target'].astype(np.int8)
        
    for column in tqdm(df.columns):
        if column not in ['bid_price_perp', 'bid_amount_perp', 'ask_price_perp',
                          'ask_amount_perp', 'bid_price_spot', 'bid_amount_spot',
                          'ask_price_spot', 'ask_amount_spot']:
            df[column] = df[column].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
