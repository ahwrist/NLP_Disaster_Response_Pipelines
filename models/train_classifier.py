import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import pandas as pd
import numpy as np
import pickle
from sklearn.utils import parallel_backend
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    """Load messages and categories data from sql lite database

    Args:
        database_filepath

    Returns:
        X: messages data
        y: category values
        category_names: names of categories
    """
    # load data from database
    # engine = create_engine('sqlite:///disaster_response_wrist.db')
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories',con=engine)
    # X = df.message.values
    X = df.message
    Y = df.iloc[:,4:].values
    category_names = list(df.iloc[:,4:].columns) #added list 8/20
    return X, Y, category_names


def tokenize(text):
    """Tokenize messages, remove stop words, lowercase words, and lemmatize words

    Args:
        text: Message text from messages in DF

    Returns:
        clean_tokens: List of cleaned, tokenized text
    """
      #Lowercase and strip text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
   
    # tokenize text
    tokens = word_tokenize(text)
    
    #Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #replace urls with "urlplaceholder"
    # url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # detected_urls = re.findall(url_regex, text)
    # for url in detected_urls:
    #     text = text.replace(url, "urlplaceholder")
    
    #remove stop words
    stop_words = stopwords.words("english")
    
    # lemmatize and remove stop words
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return clean_tokens


def build_model():
    """Build ML pipeline and model using gridsearch and range of parameters to optimize model

    Returns:
        cv: Gridsearch model result
    """
    
    #Create pipeline including count vectorizer, tfidf transformer, and a multioutput classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=100)))
    ])
    
    #Designate parameters to test
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'tfidf__norm': ['l1','l2'],
    'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=5, n_jobs = -1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate ML model to understand performance including Preecision, Recall, F1-Score, and Support

    Args:
        model: Machine learning model resulting from gridsearch build_model() function
        X_test (_type_): Messages test set
        Y_test (_type_): Categories values test set
        category_names (_type_): Names of messages categories
    """
    #use model to predict values with X_test data
    predicted = model.predict(X_test)
    #convert y_test data to dataframe
    y_test_df = pd.DataFrame(Y_test, columns=category_names)
    #convert predicted data to dataframe
    predicted_df = pd.DataFrame(predicted, columns = category_names)
    
    #print classification report
    print(classification_report(y_test_df, predicted_df, target_names= category_names, zero_division=1))


def save_model(model, model_filepath):
    """Save model to pickle file

    Args:
        model: ML model
        model_filepath: designated file path for ML model
    """
    # create pickle file based on steps from medium article here: https://medium.com/@maziarizadi/pickle-your-model-in-python-2bbe7dba2bbb
    pickle.dump(model, open(model_filepath,'wb')) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        with parallel_backend('multiprocessing'):
        
            print('Building model...')
            model = build_model()
            
            print('Training model...')
            model.fit(X_train, Y_train)
            
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    
#Run with python train_classifier.py ../data/DisasterResponse.db classifier.pkl