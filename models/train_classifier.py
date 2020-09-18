#! /usr/bin/env python3
import sys
import pandas as pd
import sqlalchemy
import re
import nltk
nltk.download([ 'stopwords', 'punkt', 'wordnet','averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score
import pickle


def load_data(database_filepath):
    """Loads data from database.

    Args:
        database_filepath (str): Path to database

    Returns:
        obj: Returns X, Y, and category_names
    """
    
    table ='disaster_data'
    
    engine = sqlalchemy.create_engine("sqlite:///"+database_filepath) 
    
    df = pd.read_sql_table(table, engine)

    X = df['message']
    
    Y = df.drop(columns=['original', 'genre','id','message'])
    
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """normalizes,lemmatize, and tokenizes text.

    Args:
        text (str): Text to be normalized.

    Returns:
        obj: Returns list of tokens.
    """    
    
    stop_words = stopwords.words("english")
    
    lemmatizer = WordNetLemmatizer()
        
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
     
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize root and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Lemmatize verbs by specifying pos
    tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in tokens]
        
    return tokens

def build_model(clf=RandomForestClassifier(random_state=143)):
    """Builds model using sklearn pipeline and GridSearchCV.

    Args:
        clf (obj, optional): Classifier used. Defaults to \
            RandomForestClassifier(random_state=143).

    Returns:
        obj: Model.
    """
  
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))])

    parameters = {
    # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    # 'features__text_pipeline__vect__max_features': (500, 2500, 5000, 10000),
    # 'features__text_pipeline__tfidf__use_idf': (True, False),
    # 'clf__estimator__bootstrap': [True, False],
    # 'clf__estimator__max_depth': [25, 50, 100, 250, 500], 
    # 'clf__estimator__max_features': ['auto', 'sqrt'],
    # 'clf__estimator__min_samples_leaf': [1, 2],
    # 'clf__estimator__min_samples_split': [2],
    'clf__estimator__n_estimators': [25]
    # 'clf__estimator__criterion': ['entropy', 'gini'],
    # 'features__transformer_weights': (
    #         {'text_pipeline': 1, 'starting_verb': 0.5},
    #         {'text_pipeline': 0.5, 'starting_verb': 1},
    #         {'text_pipeline': 0.8, 'starting_verb': 1},)
                
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2,verbose=1)
 
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates fitted model using various metrics.

    Args:
    
        `model` (obj): Fitted model.
        
        `X_test` (obj): X testing data.
        
        `Y_test` (obj): Y testing data.
        
        `category_names` (obj): Names of classification columns.
    """
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test,Y_pred, target_names=category_names))
    
    print(f' Accuracy {round(accuracy_score(Y_test, Y_pred)*100,2)}','%')

def save_model(model):
    """Saves trained model to a pickle file.

    Args:
        model (obj): Trained model.
    """
    pickle.dump(model, open('classifier.pkl', 'wb'))

def main():
    """Excecutes steps to train a classifier.
    """
    print(sys.argv)
    if len(sys.argv) == 2:
        database_filepath = sys.argv[1]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\
             test_size=0.2, random_state=143 )
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: classifier.pkl')
        save_model(model)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the argument'\
              ' \n\nExample: python'\
              'train_classifier.py ../data/DisasterResponse.db')

if __name__ == '__main__':
 
    main()
    
   
    
   