import numpy as np
import pandas as pd
import pickle
import nltk #text processing
import re
import unicodedata
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#load the words to use as filter
nltk.download('stopwords')

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

#Preprocess the data by eliminating the package name column and putting all reviews in lower case.
df.drop(['package_name'],axis=1, inplace=True)
#chante all to lower case
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


# load the model from data

#filename = 'modelo_final.sav' #use absolute path
loaded_model = pickle.load(open('../models/modelo_final.sav', 'rb'))

#Predict using the model 
#predigo el target y para los valores seteados, selecciono cualquiera para ver
print('Predicted ] : \n', loaded_model.predict(X_test[10:17]))
print('Class ] : \n', y_test[10:17])
