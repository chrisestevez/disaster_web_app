import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data', engine)

# load model
model = load("./models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Website index.

    Returns:
        html: Webpage with classifier and plots.
    """
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    
    genre_names = list(genre_counts.index)
    
    category_counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    
    category_names = list(category_counts.index)
    
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                   'tickangle': 40,'size': 8
                }
                
            }
        }
    ]
    

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Classification results

    Returns:
        html: Displays classification results.
    """
    # save user input in query
    query = request.args.get('query', '') 
    # print(query)
    #Clean query
    # query = tokenize(query)
    # print('cleaned---------------',query)

    # use model to predict classification for query
    classification_labels = model.predict(tokenize(query))[0] # model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Executes Flask app.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()