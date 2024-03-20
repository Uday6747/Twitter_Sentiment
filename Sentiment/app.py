from flask import Flask, render_template
from flask import Flask, render_template, request

# utilities :
import re # regular expression library
import numpy as np
import pandas as pd

# plotting :
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk :
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# sklearn :
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# time library :
import time


app= Flask(__name__)

global file_name
# Importing the dataset :
DATASET_COLUMNS=['target','text']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('Data1.csv',
                 encoding=DATASET_ENCODING)
df.columns = DATASET_COLUMNS

# Display of the first 5 lines :
df.sample(5)

df.info()



# Replacing the values to ease understanding :
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df['target'] = le.fit_transform(df['target'])
print(df['target'])

# Selecting the text and Target column for our further analysis :
data = df[['text','target']]
data



# Making statement text in lower case :
data['text'] = data['text'].str.lower()
data['text'].tail()



'''def preprocessing(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'(www.[^s]+)|(https?://[^s]+)', ' ', text)

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    # Remove repeating characters
    text = re.sub(r'(.)\1+', r'\1', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Apply stemming
    st = nltk.PorterStemmer()
    tokens = [st.stem(word) for word in tokens]

    # Apply lemmatization
    lm = nltk.WordNetLemmatizer()
    tokens = [lm.lemmatize(word) for word in tokens]

    return ' '.join(tokens)'''

def preprocessing(text):
    if isinstance(text, str):  # Check if text is a string
        # Convert text to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'(www.[^s]+)|(https?://[^s]+)', ' ', text)

        # Remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join(word for word in text.split() if word not in stop_words)

        # Remove repeating characters
        text = re.sub(r'(.)\1+', r'\1', text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Apply stemming
        st = nltk.PorterStemmer()
        tokens = [st.stem(word) for word in tokens]

        # Apply lemmatization
        lm = nltk.WordNetLemmatizer()
        tokens = [lm.lemmatize(word) for word in tokens]

        return ' '.join(tokens)
    else:
        return ''  # Return empty string for non-string values

# Example usage:
# Assuming dataset is your DataFrame containing 'text' column
data['processed_text'] = data['text'].apply(preprocessing)
print(data['processed_text'].head())
# Assuming dataset_test is your DataFrame containing 'text' column


 
# Separating input feature and label :
print(data.columns)

X = data.processed_text
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorizing text data
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single', methods=['GET', 'POST'])
def single():
    if request.method == 'POST':
        clf = LogisticRegression(C=2.1, solver='liblinear', multi_class='auto')
        clf.fit(X_train_transformed, y_train)
        y_pred = clf.predict(X_test)
        # Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics
        # Module Accuracy: How often is the classifier correct?
        cl_acc = metrics.accuracy_score(y_test, y_pred)
        st1 = request.form['single_comm']
        st = preprocessing(st1)
        st = vectorizer.transform([st])
        pred = clf.predict(st)
        print(pred)

        if pred==0:
            predi = 'Negative'
        elif pred==1:
            predi = 'Neutral'
        elif pred==2:
            predi = "Positive"
        else:
            predi = None

        from textblob import TextBlob

        # Perform sentiment analysis
        analysis = TextBlob(st1)

        # Get polarity (sentiment) score
        polarity = analysis.sentiment.polarity

        # Get subjectivity score
        subjectivity = analysis.sentiment.subjectivity

        # Take the absolute value of the polarity score
        polarity = abs(polarity)
        conf = polarity * 100
            
        print("Polarity:", polarity * 100)
        print("Subjectivity:", subjectivity * 100)
        return render_template('single.html', pred=predi, conf=conf)
    else:
        # Handle other HTTP methods
        return render_template('single.html', predi=None, conf=0)

@app.route('/file')
def file():
    return render_template('file.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':

        return 'No selected file'
    print(file.filename)
    global file_name
    file_name = file.filename
    # Save the uploaded file to a specific location
    file.save(file.filename)
    # Process the file (e.g., read it, analyze it, etc.)
    # Here you can add your file processing logic
    
    return 'File uploaded successfully'

@app.route('/file_result')
def file_result():
    DATASET_COLUMNS_test=['text']
    DATASET_ENCODING = "ISO-8859-1"
    df_test = pd.read_csv(file_name,
                    encoding=DATASET_ENCODING)
    df_test.columns = DATASET_COLUMNS_test
    # Display of the first 5 lines :
    print(df_test.sample(5))
    print(df_test.info())
    dataset_test = df_test[['text']]
    dataset_test['text'] = dataset_test['text'].str.lower()
    dataset_test['text'].tail()
    dataset_test['processed_text'] = dataset_test['text'].apply(preprocessing)
    print(dataset_test['processed_text'].head())
    print(dataset_test.columns)
    dataset = dataset_test.processed_text
    dataset = vectorizer.transform(dataset)
    return render_template('result.html')

'''
@app.route('')
def accuracy():
    return render_template('tag.html')
'''
'''
@app.route('')
def accuracy():
    return render_template('accuracy.html')

@app.route('')
def accuracy():
    return render_template('data_info.html')

@app.route('')
def accuracy():
    return render_template('report.html')

@app.route('')
def accuracy():
    return render_template('about.html')
'''
if __name__ == '__main__':
    app.run(debug=True)