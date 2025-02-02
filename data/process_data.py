#! /usr/bin/env python3
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads and merges data from files.

    Args:
        messages_filepath (obj): Messages csv data.
        categories_filepath (obj): Categories csv data.

    Returns:
        df: Dataframe with Messages and Categories merged.
    """
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets by id
    df = messages.merge(categories, on='id')
        
    return df
  

def clean_data(df):
    """Cleans data from load_data output.

    Args:
        df (df): df data to be cleaned

    Returns:
        df: Clean df.
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select first row of categories
    row = categories.iloc[0]
   
    # extracts names from row
    category_colnames = row.apply(lambda x: x[:-2])
    
    # assigns extracted column names to categories
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # fix categories grater than one
        categories[column] = categories[column].apply(lambda x: x if x <2 \
            else 1)
    
    # drop the original categories column from df
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], join='inner', axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """Saves cleaned data sqlite db.

    Args:
        df (df): Data to be save in database.
        database_filename (str): Name of the database.
    """
    # create database engine
    engine = create_engine('sqlite:///'+database_filename)
    # export data to database
    df.to_sql('disaster_data', engine, index=False, if_exists='replace')  


def main():
    """Executes steps to process data.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

