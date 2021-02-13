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

#############################################################################
####    Miscellaneous functions mainly for handling cast/crew dictionaries
#############################################################################


def convert_dict(x):
    try:
        return literal_eval(x)
    except:
        return []
    
def convert_dict(x):
    try:
        return literal_eval(x)
    except:
        return []
def director(crew):
    for item in crew:
        for key, value in item.items():
            if value=='Director': return item['name']
def screenwriter(crew):
    writers = []
    for item in crew:
        for key, value in item.items():
            if value in ['Screenplay', 'Writer', 'Author', 'Story']:
                writers.append(item['name'])
    return ";".join(writers)
def editor(crew):
    editors = []
    for item in crew:
        for key, value in item.items():
            if value=='Editor': editors.append(item['name'])
    return ";".join(editors)

def actors(cast):
    actors = []
    for item in cast:
        actors.append(item['name'])
        if len(actors)>=3: break
        
    return ";".join(actors)

def sum_actor_pop(cast):
    popularity = []
    for item in cast:
        popularity.append(item['popularity'])
        if len(popularity)>=3: break
    return sum(popularity)
def avg_actor_pop(cast):
    popularity = []
    for item in cast:
        popularity.append(item['popularity'])
        if len(popularity)>=3: break
    return sum(popularity)/ 3
def max_actor_pop(cast):
    popularity = []
    for item in cast:
        popularity.append(item['popularity'])
        if len(popularity)>=3: break
    try:
        return max(popularity)
    except:
        return 0
def min_actor_pop(cast):
    popularity = []
    for item in cast:
        popularity.append(item['popularity'])
        if len(popularity)>=3: break
    try:
        return min(popularity)
    except:
        return 0
    
def avg_writer_pop(crew):
    popularity = []
    for item in crew:
        if item['job'] in ['Screenplay', 'Writer', 'Author', 'Story']:
            popularity.append(item['popularity'])
    return sum(popularity)/ 3
def max_writer_pop(crew):
    popularity = []
    for item in crew:
        if item['job'] in ['Screenplay', 'Writer', 'Author', 'Story']:
            popularity.append(item['popularity'])
    
    try:
        return max(popularity)
    except:
        return 0
def min_writer_pop(crew):
    popularity = []
    for item in crew:
        if item['job'] in ['Screenplay', 'Writer', 'Author', 'Story']:
            popularity.append(item['popularity'])
    try:
        return min(popularity)
    except:
        return 0

def director_pop(crew):
    for item in crew:
        for key, value in item.items():
            if value=='Director': return item['popularity']

def to_date(x):
    return datetime.strptime(x[:-2]+'01', '%Y-%m-%d')

def econ_to_date(x):
    return pd.to_datetime(x).to_period('M')
    
#############################################################################
####            Movie details dataset: load and clean
#############################################################################

def load_movie_details(config_filepath, source, destination='data/'
                       , batch_size=100, max_rows=None, clean=True
                       , plot=True):
    """
    Function to build the films dataset
    
    Args:
    - config_filepath: filepath to the config file to load your API key
    - source: directory with movies.csv and links.csv from MovieLens Group
    - destination: filepath to save the dataset. Default is data/
    - batch_size: used for monitoring progress and a creating a time-taken distribution. Default is 100.
    - max_rows: max amount of films to query, for sampling purposes. Default to None, i.e., query to whole dataset
    
    Process:
    
    """
    
    print("START TIME = ", datetime.now().strftime("%H:%M:%S"))
    
    print("Loading the MovieLens data ...")
    
    # read in the API key
    config = configparser.ConfigParser()
    config.read('API.cfg')
    API_KEY = config['API']['KEY']
    
    # load in the movielens dataset to df_movies
    try:
        df_movies = pd.read_csv(os.path.join(source, 'movies.csv'))
        links = pd.read_csv(os.path.join(source, 'links.csv'))
    except:
        print("ERROR: path provided to MovieLens data does not exist")
        return 0
    
    df_movies = df_movies.merge(links, on='movieId')
    # no longer need links so can delete from memory
    del links
    
    # instantiate empty data structures to be filled
    times = []
    master_df = pd.DataFrame()
    
    if max_rows == None:
        max_rows = df_movies.shape[0]
        
        
    print("Beginning API calls for movie details ...")

    for i in range(0, max_rows, batch_size):
        START_TIME=time.time()
        START_INDEX = i
        END_INDEX = START_INDEX + batch_size

        if END_INDEX > df_movies.shape[0]:
            movies = df_movies.iloc[START_INDEX:]
            END_INDEX = df_movies.shape[0]-1
        else:
            movies = df_movies.iloc[START_INDEX:END_INDEX]

        # query the API and create a details dataframe
        df_details = pd.DataFrame()
        for tmdbId in movies['tmdbId']:
            details_query = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US".format(tmdbId, API_KEY)
            details = requests.get(url=details_query).json()

            # if the first row, fill dataframe
            if df_details.shape == (0,0):
                df_details = pd.json_normalize(details)
            else:
                df_temp = pd.json_normalize(details)
                df_details = pd.concat([df_details, df_temp], axis=0)

        if master_df.shape[0] == 0:
            master_df = df_details
        else:
            master_df = pd.concat([master_df, df_details], axis=0)

        times.append(time.time()-START_TIME)
        
        PCT_0 = round(100 * START_INDEX / max_rows)
        PCT_1 = round(100 * END_INDEX / max_rows)
    
        for pct in [10,20,30,40,50,60,70,80,90]:
            if PCT_0 <= pct and PCT_1 > pct:
                print(f"Approximately {PCT_0}% complete ...")
    
    print("Completed all API calls for movie details!")
    
    # saves the raw details dataframe that will be harder to clean
    if not os.path.exists(destination):
        os.mkdir(destination)
        
    master_df.to_csv(
        os.path.join(destination, "raw_movie_details.csv")
        , index=False
    )
        
    df_times = pd.DataFrame({'time':times})
    df_times.to_csv(os.path.join(destination, "batches_times_taken.csv"))
    
    # print out some stats from the dataset
    print("COMPLETION TIME = ", datetime.now().strftime("%H:%M:%S"))
    print('\n')
    print(f"Time to build data: {timedelta(seconds=df_times['time'].sum())}")
    print(f"Raw dataset saved to {destination}/raw_movie_details.csv")
    print(f"Raw dataset row count: {master_df.shape[0]}")
    
    if plot:
        # plot a histogram of the times taken to query 100 movies
        plt.figure(figsize=(10,6))
        plt.title(f"Distribution of query time per {batch_size} API calls")
        plt.xlabel("Time (s)")
        plt.ylabel("Count in bucket")
        plt.hist(x=df_times['time'])
        plt.show()

    if clean:
        master_df = clean_movie_details(destination, df=master_df)
    
    return master_df
    
def clean_movie_details(destination, df=None, source=None
                       , print_info=True):
    """
    Function to clean the raw movie details dataset.
    Input either a dataframe or the location of one

    Args:
    - destination: directory to save the cleaned dataframe to
    - df: dataframe to clean
    - source: directory where the raw movie data is saved as a csv
    """
    
    try:
        if (df==None) & (source==None):
            print("Either the raw dataframe or filepath must be provided")
    except ValueError:
        pass

    if source!=None:
        df = pd.read_csv(os.path.join(source, 'raw_movie_details.csv')
                        , lineterminator='\n')
        
    
    CONVERT_COLUMNS = ['production_companies'
                       , 'production_countries'
                       , 'spoken_languages']
    DROP_COLUMNS = ['backdrop_path', 'genres', 'homepage','imdb_id'
                    , 'poster_path', 'title', 'video', 'vote_count'
                    , 'vote_average', 'overview', 'tagline', 'adult'
                    , 'production_companies', 'production_countries'
                    , 'spoken_languages', 'status']
    
    for col in CONVERT_COLUMNS:
        df[col] = df[col].apply(convert_dict)
    
    # read in the data
    print("Cleaning the raw movie details DataFrame ...")

        # flatten the dictionary for the production company data
    df['num_prods'] = df['production_companies'].apply(
        lambda x: len(x) if type(x)==list else math.nan
    )

    df['prod_comp_names'] = df['production_companies'].apply(
        lambda x: ";".join([company['name'] for company in x])
        if type(x)==list else math.nan
    )

    df['prod_comp_cntry'] = df['production_companies'].apply(
        lambda x: ";".join(list(set([company['origin_country']
                                     for company in x
                                     if company['origin_country']!='']
                                   )))
        if type(x)==list else math.nan
    )

    # flatten the dictionary for the spoken languages data
    df['language'] = df['spoken_languages'].apply(
        lambda x: ";".join([language['english_name'] for language in x])
        if type(x)==list else math.nan
    )
    # count on how many languages in the film
    df['num_languages'] = df['language'].apply(lambda x: len(x.split(';')))

    # drop features we shall not be using
    df.drop(DROP_COLUMNS, axis=1, inplace=True)
    
    df.drop([col for col in df.columns if "collection" in col]
            , axis=1, inplace=True)

    # make sure that this is not throwing a warning
    try:
        df = df[df['success']!=False].copy()
        df.drop(['success','status_code','status_message']
                , axis=1, inplace=True)
    except:
        # sample size may not have been large enough to return a success col
        pass
    
    # convert the release date to a datetime
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    # restrict the release date to 1969 onwards
    df = df.query("release_date>'1969'").copy()
    
    # set the budget and revenue to integers
    df['budget'] = pd.to_numeric(df['budget'])
    df['revenue'] = pd.to_numeric(df['revenue'])
    
    
    # Here, things will change depending on if a dataframe was sent down,
    # or a source directory. If a df, dropping NaN works fine. If a source
    # directory, those NaN show up as '', so we drop them that way
    df.dropna(how='any', subset=['prod_comp_cntry', 'language']
             , inplace=True)
    df = df[df['language']!=''].copy()
    df = df[df['prod_comp_cntry']!=''].copy()
    
    # select only films with both a budget and revenue non-zero
    df = df.query('budget>0 and revenue>0').copy()

    if not os.path.exists(destination):
        os.mkdir(destination)
        
    FILENAME = "movie_details.csv"
    
    df.to_csv(
        os.path.join(destination, FILENAME)
        , index=False
    )
    
    print("Finished cleaning the movie details data!")
    print(f"Cleaned dataset row count: {df.shape[0]}")
    print(f"Cleaned dataset saved to {destination}/{FILENAME}")
    
    if print_info:
        print()
        print(df.info())
    
    return df

#############################################################################
####            Movie cast & crew dataset: load and clean
#############################################################################
def load_movie_cast_crew(config_filepath, source, destination='data/'
                       , batch_size=100, max_rows=None, clean=True
                       , plot=True):
    """
    Function to build the films dataset
    
    Args:
    - config_filepath: filepath to the config file to load your API key
    - source: directory with movie_details.csv
    - destination: filepath to save the dataset. Default is data/
    - batch_size: used for monitoring progress and a creating a time-taken distribution. Default is 100.
    - max_rows: max amount of films to query, for sampling purposes. Default to None, i.e., query to whole dataset
    
    Process:
    
    """
    
    print("START TIME = ", datetime.now().strftime("%H:%M:%S"))
    
    print("Loading the movie details data ...")
    
    # read in the API key
    config = configparser.ConfigParser()
    config.read('API.cfg')
    API_KEY = config['API']['KEY']
    
    # instantiate empty data structures to be filled
    times = []
    master_df = pd.DataFrame()
        
    print("Beginning API calls for movie cast and crew ...")


    # load in the cleaned movie_details dataset
    try:
        df_movies = pd.read_csv(os.path.join(source, 'movie_details.csv'))
    except:
        print("ERROR: path provided to movie details data does not exist")
        return 0
    
    if max_rows == None:
        max_rows = df_movies.shape[0]
    
    # loop through all rows in batches
    for i in range(0, max_rows, batch_size):
        START_TIME=time.time()
        START_INDEX = i
        END_INDEX = START_INDEX + batch_size

        if END_INDEX > df_movies.shape[0]:
            movies = df_movies.iloc[START_INDEX:]
            END_INDEX = df_movies.shape[0]-1
        else:
            movies = df_movies.iloc[START_INDEX:END_INDEX]

        # query the API and create a details dataframe
        df_credits = pd.DataFrame()
        for tmdbId in movies['id']:
            credits_query="https://api.themoviedb.org/3/movie/{}/credits?api_key={}&language=en-US".format(int(tmdbId), API_KEY)
            credits = requests.get(url=credits_query).json()

            # if the first row, fill dataframe
            if df_credits.shape == (0,0):
                df_credits = pd.json_normalize(credits)
            else:
                df_temp = pd.json_normalize(credits)
                df_credits = pd.concat([df_credits, df_temp], axis=0)

        if master_df.shape[0] == 0:
            master_df = df_credits
        else:
            master_df = pd.concat([master_df, df_credits], axis=0)

        times.append(time.time()-START_TIME)
        
        PCT_0 = round(100 * START_INDEX / max_rows)
        PCT_1 = round(100 * END_INDEX / max_rows)
    
        for pct in [10,20,30,40,50,60,70,80,90]:
            if PCT_0 <= pct and PCT_1 > pct:
                print(f"Approximately {PCT_0}% complete ...")
    
    print("Completed all API calls for movie cast and crew!")
    
    # saves the raw details dataframe that will be harder to clean
    if not os.path.exists(destination):
        os.mkdir(destination)
        
    master_df.to_csv(
        os.path.join(destination, "raw_cast_crew.csv")
        , index=False
    )
        
    df_times = pd.DataFrame({'time':times})
    df_times.to_csv(os.path.join(destination, "cc_batches_times_taken.csv"))
    
    # print out some stats from the dataset
    print("COMPLETION TIME = ", datetime.now().strftime("%H:%M:%S"))
    print('\n')
    print(f"Time to build data: {timedelta(seconds=df_times['time'].sum())}")
    print(f"Raw dataset saved to {destination}/raw_cast_crew.csv")
    print(f"Raw dataset row count: {master_df.shape[0]}")
    
    if plot:
        # plot a histogram of the times taken to query 100 movies
        plt.figure(figsize=(10,6))
        plt.title(f"Distribution of query time per {batch_size} API calls")
        plt.xlabel("Time (s)")
        plt.ylabel("Count in bucket")
        plt.hist(x=df_times['time'])
        plt.show()

    if clean:
        master_df = clean_cast_crew(destination, df=master_df)
    
    return master_df

def clean_cast_crew(destination, df=None, source=None
                    , print_info=True):
    """
    Function to clean the cast and crew dataset.
    Input either a dataframe or the location of one

    Args:
    - destination: directory to save the cleaned dataframe to
    - df: dataframe to clean
    - source: directory where the raw movie data is saved as a csv
    """
    
    # read in the data
    print("Cleaning the raw cast and crew DataFrame ...")
    
    try:
        if (df==None) & (source==None):
            print("Either the raw dataframe or filepath must be provided")
    except ValueError:
        pass

    if source!=None:
        df = pd.read_csv(os.path.join(source, 'raw_cast_crew.csv')
                        , lineterminator='\n')
    
    CONVERT_COLUMNS = ['cast','crew']
    DROP_COLUMNS = ['cast','crew']
    for col in CONVERT_COLUMNS:
        df[col] = df[col].apply(convert_dict)
    
    df['director'] = df['crew'].apply(director)
    df['director_pop'] = df['crew'].apply(director_pop)

    df['writers'] = df['crew'].apply(screenwriter)
    df['num_writers'] = df['writers'].apply(lambda x: len(x.split(';')))
    df['avg_writer_pop'] = df['crew'].apply(avg_writer_pop)
    df['max_writer_pop'] = df['crew'].apply(max_writer_pop)
    df['min_writer_pop'] = df['crew'].apply(min_writer_pop)

    df['actors'] = df['cast'].apply(actors)
    df['sum_actor_pop'] = df['cast'].apply(sum_actor_pop)
    df['avg_actor_pop'] = df['cast'].apply(avg_actor_pop)
    df['max_actor_pop'] = df['cast'].apply(max_actor_pop)
    df['min_actor_pop'] = df['cast'].apply(min_actor_pop)
    
    df.drop(DROP_COLUMNS, axis=1, inplace=True)
    
    FILENAME = 'cast_crew.csv'
    
    df.to_csv(
        os.path.join(destination, FILENAME)
        , index=False
    )
    
    print("Finished cleaning the movie details data!")
    print(f"Cleaned dataset row count: {df.shape[0]}")
    print(f"Cleaned dataset saved to {destination}/{FILENAME}")
    
    if print_info:
        print()
        print(df.info())
    
    return df

#############################################################################
####                Merge the datasets together
#############################################################################



def merge_datasets(source, destination, print_info=True):
    """
    Merge the datasets together
    
    Assumes that the macroceonomic data is saved in a directory named
    data/macroeconomics
    """
    print("Merging the movie, cast and crew, and macroeconomic data ...")
    
    # read in the movies, cast&crew, and macroeconomic datasets
    # make use of the converter functionality to get dates into a useful type
    movies = pd.read_csv(os.path.join(source, 'movie_details.csv')
                        , converters = {'release_date': to_date})
    cast_crew = pd.read_csv(os.path.join(source, 'cast_crew.csv'))
    
    # read in the genre data using the original dataset
    genres = pd.read_csv('data/ml-latest/movies.csv')
    links = pd.read_csv('data/ml-latest/links.csv')
    genres = genres.merge(links, on='movieId')[['tmdbId','genres']]
    movies = movies.merge(genres, left_on='id', right_on='tmdbId')
    
    # macroeconomic data
    unemployment = pd.read_csv('data/macroeconomics/UNRATE.csv'
                               , converters = {'DATE': econ_to_date})
    pce = pd.read_csv('data/macroeconomics/PCE.csv'
                     , converters = {'DATE': econ_to_date})
    cpia = pd.read_csv('data/macroeconomics/CPIAUCSL.csv'
                      , converters = {'DATE': econ_to_date})
    
    # merge the cast and crew data to the movies on id
    movies = movies.merge(cast_crew,'inner', on='id')
    
    # handles the release dates and merge to macroeconomic data
    movies['DATE'] = movies['release_date'].apply(lambda x: x.to_period('M'))
    movies = movies.merge(
        unemployment
        , how='left'
        , on='DATE'
    ).merge(
        pce
        , how='left'
        , on='DATE'
    ).merge(
        cpia
        , how='left'
        , on='DATE'
    ).drop(['DATE'], axis=1)
    
    # drop any rows that have any null values
    movies.dropna(inplace=True)
    
    # save the file as movies.csv in the destination directory
    if not os.path.exists(destination):
        os.mkdir(destination)
        
    FILENAME = "movies.csv"
    
    movies.to_csv(
        os.path.join(destination, FILENAME)
        , index=False
    )
    
    if print_info:
        print()
        print(movies.info())
    
    print("Completed! Data ready for analysis")
    
    return movies