# Machine Learning Engineer Capstone Project: Modelling the success of films

This repository contains the files and code for a submission to Udacity for the Machine Learning Engineer Nanodegree. The project is motivated by a personal passion for films, combined with the current state of disarray in Hollywood.

## Contents
1. Motivation
2. What's in the repo
3. Data
4. Licenses

## 1. Motivation

This project aims to take data frmo

## 2. What's in the repo

The repository contains several notebooks that encompass the development process. We have the first for data engineering, the second for EDA, two for the local machine learning development and the last for the SageMaker deployment of the model.

The python files are used for simplifying the notebooks with factored code.

The source data is in the `data` directory. However, building the full dataset with the data engineering notebook would take around 5 hours from experience and require setting up an account with the TMDB API. It is much simpler to use the data in the `prod_movies` directory. This contains the data in that state after running the data engineering notebook.

See the report for more detail on the project.

## 3. Data

The dataset included in this repo was pulled from three distinct sources. These were the:

1. [MovieLens Group](https://grouplens.org/datasets/movielens/latest/) for the base set of films,
2. TMDb API for the details, cast and crew for each of the films, and
3. [FRED](https://fred.stlouisfed.org/series/) - Federal Reserve of Economic Data - for the macroeconomic data in the US.


## 4. Licenses

This product uses the TMDb API but is not endorsed or certified by TMDb.