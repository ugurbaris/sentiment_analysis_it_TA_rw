#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import re
import spacy
import string
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import seaborn as sns
from sklearn.metrics import roc_curve, auc , roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[ ]:


dataloaded = pd.read_csv("C:\\Users\\ugur_\\Downloads\\dataset_winter_2020\\development.csv")
data = dataloaded.copy()


# In[ ]:


data.shape


# In[ ]:


f = lambda x: 1 if x=="pos" else 0
data['class'] = data['class'].map(f)
data


# In[ ]:


def infdata(data):
    data['word_count'] = [len(text.split()) for text in data["text"]]                          

    data['special_chars'] = [sum(char in string.punctuation for char in text)                                 for text in data["text"]]   


# In[ ]:


infdata(data)


# In[ ]:


data.isnull().sum()


# In[ ]:


import seaborn as sns
plt.figure(figsize=(10, 8))
sns.countplot(data['class'])

plt.plot(0, label ="Negative")
plt.plot(1, label ="Positive")
plt.ylabel("Number of Documents")
plt.title("Data Distrubution")

for v in [0, 1]:
    plt.text(v, (data['class'] == v).sum(), str((data['class'] == v).sum()));

# plt.savefig('SimpleBar.png')
plt.legend()
plt.show()


# # Data Preprocessing & Cleaning

# In[ ]:


# !python -m spacy download it_core_news_sm
nlp = spacy.load("it_core_news_sm",disable=["tagger", "ner","parser"])
stop_words = [stopword for stopword in open('stopwords_it2.txt','r').read().split('\n')]
#https://www.ranks.nl/stopwords/italian
punctuations = string.punctuation


# In[ ]:


def cleaner(texts):
    parentheses_separation = re.compile("(<br\s*/><br\s*/>)|(\()|(\))|(\/)")
    
    texts = [parentheses_separation.sub(" ", text) for text in texts]
    texts = ''.join(texts)
    
    doc = unidecode.unidecode(texts)
    doc = nlp(doc)
    
    tokens = [token.lemma_.lower().strip() for token in doc]
    
    tokens = [token for token in tokens if 
                                          (token not in punctuations)
                                          and 
                                          (not token[0].isdigit())
                                          and
                                          (len(token) > 2)
                                          and
                                          (token not in stop_words)]
              
    tokens = [re.sub(r"[^a-zA-Z0-9]", "", token) for token in tokens]
                                                                     
    tokens = ' '.join(tokens)
    
    tokens= re.sub(r'\s+', ' ', tokens, flags=re.I)
    return tokens


# In[ ]:





# # Data Cleaning(Development) 

# In[ ]:


clean_data = []
import time
start_time = time.time()
for numberprocesed ,i in enumerate(data['text']):
    clean_data.append(cleaner(i))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"numberprocessed : {numberprocesed}")


# # Model Development

# Data split to training and test paritions.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(clean_data, data["class"],test_size=0.3,stratify=data["class"])
len(X_train), len(X_test), len(y_train), len(y_test),len(y_train)+len(y_test)


# In[ ]:


def G_S(X_train, y_train, X_test, y_test, classifier, score, param_grid, n_folds ):
    print("Parameter Tuning %s" % score)
    print()
    clf = GridSearchCV(classifier, param_grid, cv=n_folds,scoring=score, verbose=10, n_jobs=4)
    clf.fit(X_train, y_train)
    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores accross folds:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
    print()
    print("classification report:")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred,digits=4))
    print(confusion_matrix(y_true, y_pred,labels=[1,0]))
    return clf


# In[ ]:


score = 'f1_weighted'

pipeline_linsvc = Pipeline([
     ('tfidf', TfidfVectorizer()),
     ('classifier', LinearSVC())])

parameters_linsvc = {
     'tfidf__max_df': [(0.3),(0.4),(0.5),(0.6),(0.7),(0.8)],
     'tfidf__min_df': [5,20],
     'tfidf__ngram_range': [(1,1),(1,2)],
     'classifier__C': [0.1,0.3,0.5,0.7,0.8,0.9,1,10,100],
     'classifier__class_weight' : ['balanced']
             }


best_classifier_linsvc = G_S(X_train, y_train, X_test, y_test, pipeline_linsvc, score, parameters_linsvc,5)


# In[ ]:




pipeline_logit = Pipeline([
     ('tfidf', TfidfVectorizer()),
     ('classifier', LogisticRegression())])

parameters_logit = {
     'tfidf__max_df': [(0.3),(0.4),(0.5),(0.6),(0.7),(0.8)],
     'tfidf__min_df': [5,20],
     'tfidf__ngram_range': [(1,1),(1,2)],
     'classifier__C': [0.1,0.3,0.6,0.9,10,100,1000],
     'classifier__class_weight':["balanced"],
     'classifier__max_iter': [100,200,300]
             }


best_classifier_logit = G_S(X_train, y_train, X_test, y_test, pipeline_logit, score, parameters_logit,5)


# In[ ]:


plt.figure(figsize=(8,8))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

roc_auc = roc_auc_score(y_test, best_classifier_logit.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, best_classifier_logit.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='LR (auc = %0.3f)' % roc_auc, color='navy')
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


classifier = best_classifier_logit.best_estimator_.named_steps['classifier']
best_tfidf = best_classifier_logit.best_estimator_.named_steps['tfidf']

positive_words = {}
negative_words = {}
feature_to_coef = {
    word: coef for word, coef in zip(best_tfidf.get_feature_names(), classifier.coef_[0])
}
for negative in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=False)[:200]:
    negative_words[negative[0]] = abs(negative[1])

for positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:200]:
    positive_words[positive[0]] = positive[1]


# # WordClouds

# In[ ]:


import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS

from PIL import Image
thumbsup_mask_path = "C:\\Users\\ugur_\\thumbs_up_pos.jpeg" 
thumbsdown_mask_path = "C:\\Users\\ugur_\\thumbs_down_neg.jpg" 
def create_word_cloud(string,mask_path):
    maskArray = np.array(Image.open(mask_path))
    cloud = WordCloud(background_color = "white", max_words = 50, mask = maskArray)
    cloud.generate_from_frequencies(string)
    plt.figure(figsize=(15,10))
    plt.imshow(cloud,interpolation='bilinear')
    plt.axis('off')
    plt.savefig("wordcloud.png", format="png")
#https://www.clipart.email/clipart/thumbs-up-clipart-emoji-9264.html
#https://www.hiclipart.com/free-transparent-background-png-clipart-isrib


# In[ ]:


create_word_cloud(positive_words,thumbsup_mask_path)


# In[ ]:


create_word_cloud(negative_words,thumbsdown_mask_path)


# # Evaluation set 

# In[ ]:


dataevaluation = pd.read_csv("C:\\Users\\ugur_\\Downloads\\dataset_winter_2020\\evaluation.csv")
evaluationset = dataevaluation.copy()


# Cleaning the evaluation set

# In[ ]:


eval_dataclean = []
import time
start_time = time.time()
for numberprocesed ,i in enumerate(evaluationset["text"]):
    eval_dataclean.append(cleaner(i))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"numberprocesed : {numberprocesed}")


# # Logistic Regression

# Training Logistic Regression model with best parameters found in GridSearch and all available data.

# In[ ]:


print(best_classifier_logit.best_estimator_.named_steps["classifier"])
print(best_classifier_logit.best_estimator_.named_steps["tfidf"])


# In[ ]:


tfidf_log = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
                encoding='utf-8',
                input='content', lowercase=True, max_df=0.8, max_features=None,
                min_df=5, ngram_range=(1, 2), norm='l2', preprocessor=None,
                smooth_idf=True, stop_words=None, strip_accents=None,
                sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None, use_idf=True, vocabulary=None)

clean_data_tfidf_logistic_reg = tfidf_log.fit_transform(clean_data)

LR = LogisticRegression(C=100, class_weight='balanced', dual=False,
                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
LR.fit(clean_data_tfidf_logistic_reg, data["class"])


# In[ ]:


evaluation_features = tfidf_log.transform(eval_dataclean)
df = pd.DataFrame(data={"Id": np.arange(0,evaluationset["text"].size),
                        "Predicted":LR.predict(evaluation_features)})
f = lambda x: "pos" if x==1 else "neg"
df['Predicted'] = df['Predicted'].map(f)

df.to_csv("./finallogistic.csv", sep=',',index=False)  


# In[ ]:


df


# # Linear SVC

# Training LinearSVC model with best parameters found in GridSearch and all available data.

# In[ ]:


print(best_classifier_linsvc.best_estimator_.named_steps["classifier"])
print(best_classifier_linsvc.best_estimator_.named_steps["tfidf"])


# In[ ]:


tfidf_linsvc = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
                input='content', lowercase=True, max_df=0.5, max_features=None,
                min_df=5, ngram_range=(1, 2), norm='l2', preprocessor=None,
                smooth_idf=True, stop_words=None, strip_accents=None,
                sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b',
                tokenizer=None, use_idf=True, vocabulary=None)

clean_data_tfidf_linsvc = tfidf_linsvc.fit_transform(clean_data)

SVC = LinearSVC(C=0.9, class_weight='balanced', dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
SVC.fit(clean_data_tfidf_linsvc, data["class"])


# In[ ]:


evaluation_features = tfidf_linsvc.transform(eval_dataclean)
df2 = pd.DataFrame(data={"Id": np.arange(0,evaluationset["text"].size),
                        "Predicted":SVC.predict(evaluation_features)})
f = lambda x: "pos" if x==1 else "neg"
df2['Predicted'] = df2['Predicted'].map(f)

df2.to_csv("./finallinsvc.csv", sep=',',index=False)   


# In[ ]:


df2


# In[ ]:




