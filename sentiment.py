import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Tools for preprocessing input data
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

data=pd.read_csv("sentiment_data.tsv",delimiter="\t")[:2000]
#Processing Message
def processing(review):
    # Remove email addresses with 'emailaddr'
    raw_review = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', " ", review)

    # Remove URLs with 'httpaddr'
    raw_review = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', " ", raw_review)

    # Remove non-letters
    raw_review = re.sub("[^a-zA-Z]", " ", raw_review)

    # Remove numbers
    raw_review = re.sub('\d+(\.\d+)?', " ", raw_review)
    #converting to lower case  and spliting it
    raw_review=raw_review.lower().split()
    # Gather the list of stopwords in English Language
    stop =  set(stopwords.words("english"))
    # Remove stop words and stemming the remaining words
    meaningfull_word=[ps.stem(w) for w in raw_review if w not in stop]
    # Join the tokens back into one string separated by space,
    # and return the result.
    return " ".join(meaningfull_word)

ps = PorterStemmer()
clean_rev_corpus=[]
review_size=data.review.size

for i in range(review_size):
    clean_rev_corpus.append(processing(data["review"][i]))

print("orginal text : \n\n",data["review"][0])
print("refined text : \n\n",clean_rev_corpus[0])
#Preparing Vectors for each message
cv = CountVectorizer()
data_inp = cv.fit_transform(clean_rev_corpus)
data_inp = data_inp.toarray()
from wordcloud import WordCloud, STOPWORDS
stopword=set(STOPWORDS)
#Creating WordCloud ¶
from wordcloud import WordCloud,STOPWORDS
def show_word(data,title=None):
    word_cloud = WordCloud(background_color="black",stopwords=stopword
                          ,max_font_size=40,max_words=200,scale=3,random_state=1
                          ).generate(str(data))
    fig =  plt.figure(1,figsize=(15,15))
    plt.axis("off")
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(word_cloud)
    plt.show()
show_word(clean_rev_corpus)
#time to train
data_out=data["sentiment"]
#train test model
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(data_inp,data_out,test_size=0.20,random_state=0)
plt.figure(figsize=(8,8))
data["sentiment"].value_counts().plot.bar()

#Preparing ML Models
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
model_gn=GaussianNB()
model_gn.fit(xtrain,ytrain)
model_dtc = DecisionTreeClassifier()
model_dtc.fit(xtrain,ytrain)
model_rfc = RandomForestClassifier(n_estimators=100,random_state=0)
model_rfc.fit(xtrain,ytrain)

#predicting
prediction_nvb = model_gn.predict(xtest)
prediction_rf = model_rfc.predict(xtest)
prediction_dt = model_dtc.predict(xtest)
#Results Naive Bayes
print ("Accuracy for Naive Bayes : %0.5f \n\n" % accuracy_score(ytest, prediction_nvb))
print ("Classification Report Naive bayes: \n", classification_report(ytest, prediction_nvb))
#Results Decision Tree
print ("Accuracy for Decision Tree: %0.5f \n\n" % accuracy_score(ytest, prediction_dt))
print ("Classification Report Decision Tree: \n", classification_report(ytest, prediction_dt))
#Results Random Forest
print ("Accuracy for Random Forest: %0.5f \n\n" % accuracy_score(ytest, prediction_rf))
print ("Classification Report Random Forest: \n", classification_report(ytest, prediction_rf))