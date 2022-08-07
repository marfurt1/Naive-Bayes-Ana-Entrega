import pandas as pd
import numpy as np
import nltk #text processing
import re
import unicodedata
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from transformers import pipeline

#load the words to use as filter
nltk.download('stopwords')

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

#Preprocess the data by eliminating the package name column and putting all reviews in lower case.
df.drop(['package_name'],axis=1, inplace=True)
df['review'] = df['review'].str.lower()

#remove the stopword
stop = stopwords.words('english')
def remove_stopwords(text):
  if text is not None:
    #list of the word in the text
    words = text.strip().split()
    words_filtered = []
    for word in words:
      if word not in stop:
        words_filtered.append(word)
    result = " ".join(words_filtered) #join the word in a text with space separation
  else:
      result = None
  return result
    
# elimina espacio libre al principio y al final
df['review'] = df['review'].str.strip()

def normalize_str(text_string):
    if text_string is not None:
        result=unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
    else:
        result=None 
    return result
    df['review']=df['review'].apply(normalize_str)
df['review']=df['review'].str.replace('!','')
df['review']=df['review'].str.replace(',','')
df['review']=df['review'].str.replace('&','')
df['review']=df['review'].str.normalize('NFKC')
df['review']=df['review'].str.replace(r'([a-zA-Z])\1{2,}',r'\1',regex=True)


 
#check if a work only contain letter
def word_only_letters(word):
    for c in word:
        cat = unicodedata.category(c)
        if cat not in ('Ll','Lu'):  #only letter upper y lower
            return False
    return True
# clean only letter
def text_only_letters(text):
    if text is not None:
        #list of the word in the text
        words = text.strip().split()
        words_filtered = []
        for word in words:
            if word_only_letters(word):
                words_filtered.append(word)
            result = " ".join(words_filtered) #join the word in a text with space separation
    else:
        result = None
    return result

#remove multi letter looove iiiitttt, or repeat secuence
def replace_multiple_letters(message):
  if message is not None:
    result = re.sub(r'(.+?)\1+', r'\1', message)
    #result = re.sub(r"([a-zA-Z])＼1{2,}", r"＼1", message)
  else:
    result = None
  return result

df_interim = df.copy()
df_interim['review'] = df_interim['review'].apply(remove_stopwords)
df_interim['review'] = df_interim['review'].apply(text_only_letters)
df_interim['review'] = df_interim['review'].apply(replace_multiple_letters)

#copy to df
df = df_interim.copy()

# Separate target and predictor
X = df['review']
y = df['polarity']

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
vectorizer.get_feature_names_out()


# load the model from data
filename = '../models/multinomial.pkl'
load_model = pickle.load(open(filename, 'rb'))

Z2 = vectorizer.transform(['I do not like this app'])
# Modelo Multinomial loaded
print(f"Multi: {load_model.predict(Z2.toarray())}")

sentiment_pipeline = pipeline("sentiment-analysis")

data = ["I love you", "I hate you"]
sentiment_pipeline(data)



