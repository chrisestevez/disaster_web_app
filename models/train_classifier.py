import sys
import pandas as pd
import sqlalchemy
import re
import nltk
nltk.download([ 'stopwords', 'punkt', 'wordnet','averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP']:
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    
    table ='disaster_data'
    
    engine = sqlalchemy.create_engine("sqlite:///%s" % database_filepath)
    
    df = pd.read_sql_table(table, engine)
    X = df['message']
    Y = df.drop(columns=['original', 'genre','id','message'])
    category_names = Y.columns
    
    return X, Y, category_names # df, 


def tokenize(text):
    
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
    
    # Reduce words to their stems
    tokens = [PorterStemmer().stem(w) for w in tokens]
    
    return tokens


def build_model(clf=RandomForestClassifier(random_state=143, n_jobs=-2)):
    

    
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))#,

            #('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(clf))])

    parameters = {
    # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    # 'features__text_pipeline__vect__max_features': (None,500, 1000, 2500, 5000,
    #                                                  10000),
    # 'features__text_pipeline__tfidf__use_idf': (True, False)#,
    'clf__estimator__bootstrap': [True, False]#,
    # 'clf__estimator__max_depth': [10, 25, 50, None],
    # 'clf__estimator__max_features': ['auto', 'sqrt'],
    # 'clf__estimator__min_samples_leaf': [1, 2, 4],
    # 'clf__estimator__min_samples_split': [2, 5, 10],
    # 'clf__estimator__n_estimators': [50, 100, 250, 500, 1000],
    # 'clf__estimator__criterion': ['entropy', 'gini'],
    # 'features__transformer_weights': (
    #         {'text_pipeline': 1, 'starting_verb': 0.5},
    #         {'text_pipeline': 0.5, 'starting_verb': 1},
    #         {'text_pipeline': 0.8, 'starting_verb': 1},)
                
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 3) #  n_jobs=-2,

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    
    print('-'*50)
    
    database_filepath ='./data/DisasterResponse.db'
    
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\
             test_size=0.2, random_state=143 )
    
    print('-'*50)
    print('Building model...')
    model = build_model()
    
    print('-'*50)    
    print('Training model...')
    model.fit(X_train, Y_train)
  
 
    
   
    
   