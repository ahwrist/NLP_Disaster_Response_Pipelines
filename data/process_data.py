import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories data and merges them into one dataframe.
    
    Args: 
        messages_filepath: File path for emergency messages
        categories_filepath: File path for message categories   
        
    Returns:
        df: Merged dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    messages.head()
    categories = pd.read_csv(categories_filepath)
    categories.head()
    df = pd.merge(messages,categories,on ='id')
    return df


def clean_data(df):
    """Cleans dataframe by extracting category names from category values to be used as colunn names
    in a "Categories" dataframe, then populates with the 0 or 1 values for each category name 
    indicating whether a message falls under the category or not. Then, the new Categories 
    dataframe is concatenated with the original dataframe.

    Args:
        df: Dataframe created from the load_data function

    Returns:
        df: Cleaned dataframe
    """
    categories = df.categories.str.split(';',expand=True)
    row = categories.iloc[0] #
    category_colnames = row.str.split('-').str.get(0) #split at "-" and take first part for each category
    categories.columns = category_colnames #rename the category columns with category_colnames
    
    #convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    categories.replace({2: 1}, inplace=True) #replace all 2s with 1s
    
    df = df.drop(['categories'], axis = 1) # drop the original categories column from `df`
    df = pd.concat([df,categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe
    df = df.drop_duplicates() #remove duplicates
    df=df.dropna(how='all') #drop na
    return df


def save_data(df, database_filename):
    """Save cleaned dataframe to SQL database with file name "messages_categories"

    Args:
        df: Cleaned dataframe
        database_filename
    Output:
        SQL Database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')  


def main():
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
    
##run with python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db