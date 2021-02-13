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

def preprocess(df):
    """
    Function to preprocess the data ahead of ML
    """
    
    # drop the following columns - irrelevant now
    DROP_COLUMNS = ['id', 'original_title', 'release_date'
                   , 'tmdbId', 'popularity', 'year']
    df.drop(DROP_COLUMNS, axis=1, inplace=True)
    
    # drop all of the language columns
    DROP_COLUMNS = [col for col in df.columns if col[:3]=="lan"]
    df.drop(DROP_COLUMNS, axis=1, inplace=True)

    # loop through the columns we want to aggregate
    for col_type in [
        "original_language_"
        , "prod_comp_cntry_"
        , "prod_comp_names_"
        , "writers_"
        , "actors_"
        , "genres_"
        , "director_"
    ]:
        # create a dictionary of each unique value and its frequency
        val_freq = {}
        for col in df.columns:
            if col.startswith(col_type):
                val_freq[col] = df[col].sum()

        # create a dataframe from this dictionary; sort by count
        counts = pd.DataFrame.from_dict(
            val_freq
            , orient='index'
            , columns=['count']
        ).sort_values('count', ascending=False)
        counts['frac'] = counts['count'].apply(lambda x: 100*x / df.shape[0])

        # handle special case of production company country
        if col_type == "prod_comp_cntry_":
            DROP_COLUMNS = [col for col in counts.index][3:]

        # handle special case of directors
        elif col_type == "director_":
            DIRECTOR_COLS = [col for col in df.columns
                             if col.startswith("director_")
                             and col!="director_pop"]
            df['established_director'] = df[DIRECTOR_COLS].max(axis=1)
            DROP_COLUMNS = DIRECTOR_COLS

        # handle special case of actors
        elif col_type == "actors_":
            ACTORS_COLS = [col for col in df.columns if "actors" in col]
            df['num_top_100_actors'] = df[ACTORS_COLS].sum(axis=1)
            DROP_COLUMNS = ACTORS_COLS

        # handle all the other cases
        else:
            DROP_COLUMNS = [col for col in counts.query('frac < 2').index]


        df.drop(DROP_COLUMNS, axis=1, inplace=True)
    
    ##########################################################################
    # adjust the data for inflation
    CPI_tf = df['CPIAUCSL'].max()
    df['budget'] = df[['budget', 'CPIAUCSL']].apply(
        cpi_adjust
        , args=(CPI_tf ,)
        , axis=1
    )
    df['revenue'] = df[['revenue', 'CPIAUCSL']].apply(
        cpi_adjust
        , args=(CPI_tf ,)
        , axis=1
    )
    # no longer need CPI data
    df.drop('CPIAUCSL', axis=1, inplace=True)
    
    ########################################################################## 
    # add in useful features about the cast and crew    
    df['cast_crew_sum_pop'] = (
        df['director_pop']
        + df['avg_actor_pop']
        + df['avg_writer_pop']
    )
    df['cast_crew_product_pop'] = (
        df['director_pop']
        * df['avg_actor_pop']
        * df['avg_writer_pop']
    )
    df['runtime'].replace(to_replace=0, value=df['runtime'].median(), inplace=True)
    df = df.query('10000 <= revenue').copy()
    df = df.query('100000 <= budget').copy()
    df.drop('sum_actor_pop', axis=1, inplace=True)
    df.drop('min_writer_pop', axis=1, inplace=True)

    # code to transform columns
    for col in [
        "budget", "director_pop", "avg_writer_pop"
        , "max_writer_pop", "avg_actor_pop", "max_actor_pop"
        , "min_actor_pop", 'cast_crew_sum_pop'
        , 'cast_crew_product_pop'
    ]:
        df['log10_'+col] = df[col].apply(lambda x: math.log10(x))
        df.drop(col, axis=1, inplace=True)
        
    return df

def cpi_adjust(row, CPI_tf):
    """
    Function to adjust the dollar amounts of previous years by the Consumer
    Price Index.
    
    x_tf = x_ts (CPI_tf / CPI_ts)
    
    Args:
    - row: the row of data sent to the function
    - CPI_tf: most recent CPI
    """
    CPI_ts = row['CPIAUCSL']
    x_ts = row.iloc[0]
    
    x_tf = x_ts * (CPI_ts / CPI_tf)
    return x_tf
    
if __name__=="__main__":
    print("""
    Hi there,
    
    If you are running this script in a console ... don't!
    Look to the 01-data-engineering-pipeline notebook for the implentation of
    this file's functions.
    
    Thanks!
    """)