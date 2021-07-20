#!/usr/bin/env python
# coding: utf-8


import nltk, re, savefig, sklearn, string, sys
from IPython.core.display import display
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")


if sys.version_info[0]<3: input=raw_input




##### Uncomment is ready to activate the following command, if you get suspended, but the following command is not secure.
##### 2021-07-15
##### https://qiita.com/DisneyAladdin/items/9733a7e640a175c23f39

import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

####

# Singular Value Decomposition
#
# M[11314, 792] = U[11314, 20] * Sigma[20, 20] * V_T[20, 792]

###



# Download the dataset

from sklearn.datasets import fetch_20newsgroups
X_train, y_train = fetch_20newsgroups(subset='train', return_X_y=True)
X_test, y_test = fetch_20newsgroups(subset='test', return_X_y=True)

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
stop_words = list(set(stopwords.words("english")))
stop_words += list(string.punctuation)
stop_words += ['__', '___']


# Uncomment and run the 3 lines below if you haven't got these packages already
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')


# The following command is getting rid of unwanted words to count including E-mail address, top level of domain in URL, numeric characters.
def rmv_emails_websites(string):
    """Function removes emails, websites and numbers"""
    new_str = re.sub(r"\S+@\S+", '', string)
    new_str = re.sub(r"\S+.co\S+", '', new_str)
    new_str = re.sub(r"\S+.ed\S+", '', new_str)
    new_str = re.sub(r"[0-9]+", '', new_str)
    return new_str

X_train = list(map(rmv_emails_websites, X_train)) 
X_test  = list(map(rmv_emails_websites, X_test))


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(X_train)

print( bag_of_words.todense() )



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(lowercase=True, 
                        stop_words=stop_words, 
                        tokenizer=tokenizer.tokenize, 
                        max_df=0.2,
                        min_df=0.02
                       )
tfidf_train_sparse = tfidf.fit_transform(X_train)
tfidf_train_df = pd.DataFrame(tfidf_train_sparse.toarray(), 
                        columns=tfidf.get_feature_names())


tfidf_test_sparse = tfidf.transform(X_test)
tfidf_test_df = pd.DataFrame(tfidf_train_sparse.toarray(), 
                        columns=tfidf.get_feature_names())


from sklearn.decomposition import TruncatedSVD

lsa_obj = TruncatedSVD(n_components=20, n_iter=100, random_state=42)
plain_lsa_data = lsa_obj.fit_transform(bag_of_words)
tfidf_lsa_data = lsa_obj.fit_transform(tfidf_train_df)
Sigma = lsa_obj.singular_values_
V_T = lsa_obj.components_.T

print( plain_lsa_data )
sns_plot = sns.barplot(x=list(range(len(Sigma))), y = Sigma)
fig = sns_plot.get_figure()
fig.savefig("output3.png")


###

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.style.use("ggplot")
import seaborn as sns
sns.set()


sns_plot =    sns.barplot(x=list(range(len(Sigma))), y = Sigma)
fig = sns_plot.get_figure()
fig.savefig("output.png")



##### the following assignment of values are written by myself, so it may not be correct.

lsa_term_topic = lsa_obj.components_.T
eda_train =  tfidf_train_df


term_topic_matrix = pd.DataFrame(data=lsa_term_topic,
                                 #index = eda_train.columns,
                                 index = tfidf.get_feature_names(),
                                 columns = [f'Latent_concept_{r}' for r in range(0,V_T.shape[1])])


topic_encoded_df = pd.DataFrame( plain_lsa_data,
                                 columns = [f'Latent_concept_{r}' for r in range(0,V_T.shape[1])])

topic_encoded_df["X_train"] = X_train
array=["X_train"] + [f'Latent_concept_{r}' for r in range(0,V_T.shape[1])]
display(topic_encoded_df[array])

plt.figure()
data = term_topic_matrix[f'Latent_concept_1']
data = data.sort_values(ascending=False)
top_10 = data[:10]
plt.title('Top terms along the axis of Latent concept 1')
fig = sns.barplot(x= top_10.values, y=top_10.index)
fig.figure.savefig("output_top10.png") 

plt.figure()
last_5 = data[-5:]
plt.title('last terms along the axis of Latent concept 1')
fig = sns.barplot(x= last_5.values, y=last_5.index)
fig.figure.savefig("output_last5.png")   
 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


logreg_lsa = LogisticRegression()
logreg     = LogisticRegression()
logreg_param_grid = [{'penalty':['l1', 'l2']},
                 {'tol':[0.0001, 0.0005, 0.001]}]
grid_lsa_log = GridSearchCV(estimator=logreg_lsa,
                        param_grid=logreg_param_grid, 
                        scoring='accuracy', cv=5,
                        n_jobs=-1)
grid_log = GridSearchCV(estimator=logreg,
                        param_grid=logreg_param_grid, 
                        scoring='accuracy', cv=5,
                        n_jobs=-1)
best_lsa_logreg = grid_lsa_log.fit(tfidf_lsa_data, y_train).best_estimator_
best_reg_logreg = grid_log.fit(tfidf_train_df, y_train).best_estimator_
print("Accuracy of Logistic Regression on LSA train data is :", best_lsa_logreg.score(tfidf_lsa_data, y_train))
print("Accuracy of Logistic Regression with standard train data is :", best_reg_logreg.score(tfidf_train_df, y_train))
