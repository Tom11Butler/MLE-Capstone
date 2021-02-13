import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from ast import literal_eval
import math
import numpy as np
import requests
import time
import configparser
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def to_date(x):
    return datetime.strptime(x[:-2]+'01', '%Y-%m-%d')

def load_data():
    df = pd.read_csv('prod-movies/movies.csv'
                    , converters = {'release_date': to_date})
    df['profit'] = df['revenue'] - df['budget']
    df['class'] = df['profit'].apply(lambda x: 1 if x > 0 else 0)
    df.drop('profit', axis=1, inplace=True)
    
    df['year'] = df['release_date'].dt.year
    df['month'] = df['release_date'].dt.strftime('%b')
    
    # create dataframes with these categorical features encoded as binary values
    ONE_HOT_COLS = ['original_language', 'director', 'month']

    # HANDLE THE COLUMNS THAT CAN UNDERGO ONE HOT ENCODING
    cat_dfs = []
    for col in ONE_HOT_COLS:
        onehot_encoder = OneHotEncoder()
        col_encoded = onehot_encoder.fit_transform(df[[col]])
        col_encoded_df = pd.DataFrame(col_encoded.toarray()
                                     , columns=[col+"_"+str(x) for x in
                                                onehot_encoder.categories_[0]])

        if col=='director':
            # find the most proflific directors, take the top 100
            directors = col_encoded_df.transpose().sum(axis=1)
            directors = directors.sort_values(ascending=False)[:100].index
            col_encoded_df = col_encoded_df[directors]

        cat_dfs.append(col_encoded_df)
        
    # append the columns onto the main dataframe
    for i, col in enumerate(ONE_HOT_COLS):
        df = df.merge(cat_dfs[i], left_on=df.index, right_on=cat_dfs[i].index)
        df.drop([col, 'key_0'], axis=1, inplace=True)
        
    # HANDLE THE COMPLICATED CATEGORICAL COLUMNS WITH A WORSE STRUCTURE
    OTHER_CATEGORICAL_COLS = ['genres', 'actors', 'writers', 'prod_comp_names'
                              , 'prod_comp_cntry', 'language']

    df_cat = df[OTHER_CATEGORICAL_COLS].copy()
    # clean the genres data so it is in the same format as all the other columns
    df_cat['genres'] = df_cat['genres'].apply(lambda x: x.replace('|', ';'))
    
    category_counts = {}

    # loop through each columns
    for col in df_cat.columns:
        # generate the set of unique values for the column
        counts_dict = {}
        for i, row in df_cat[[col]].iterrows():
            for key in row[0].split(';'):
                try:
                    count = counts_dict[key]
                    counts_dict[key] += 1
                except:
                    counts_dict[key] = 1

        category_counts[col] = counts_dict
        
    # We have the counts for each of the category values in a dictionary
    # Order the dictionary to take those only in the top 100, say
    category_dict = {}
    for key in category_counts:
        if len(category_counts[key]) > 100:
            df_tmp = pd.DataFrame.from_dict(category_counts[key]
                                           , orient='index'
                                           , columns=['count'])
            df_tmp = df_tmp.sort_values('count', ascending=False).iloc[:100]
            category_dict[key] = list(df_tmp.index)
        else:
            category_dict[key] = [k for k in category_counts[key]]
        
    cat_dfs = []

    for col in df_cat.columns:
        i=0
        df_cat[col] = df_cat[col].apply(lambda x: expand_categories(x, category_dict[col]))
        categories = df_cat[col].str.split(';', expand=True)
        categories.columns = [col+"_"+str(value) for value in categories.columns]

        for cat_col in categories:
            categories[cat_col] = pd.to_numeric(categories[cat_col].apply(lambda x: x[-1]))
            i += 1

        categories.columns = [(col+"_"+item).replace(' ', '_') for item in category_dict[col]]

        cat_dfs.append(categories)
    
    for i, col in enumerate(OTHER_CATEGORICAL_COLS):
        df = df.merge(cat_dfs[i], left_on=df.index, right_on=cat_dfs[i].index)
        df.drop([col, 'key_0'], axis=1, inplace=True)
    
    return df

def expand_categories(row, cat_set):
    """
    Function to expand the categories data into a ;-separated list to include
    a value with each of the categories
    """
    my_str = ""
    
    # iterate through all possible categories
    for category in cat_set:
        # check for matches in this row
        my_str+= category
        if category in row:
            my_str += '-1;'
        else:
            my_str += '-0;'
            
    return my_str[:-1]

if __name__=="__main__":
    print("""
    Hi there,
    
    If you are running this script in a console ... don't!
    Look to the 01-data-engineering-pipeline notebook for the implentation of
    this file's functions.
    
    Thanks!
    """)