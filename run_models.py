import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import ftr_data_clean as dc
import train as trn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

import pickle
import split
import sys

if (len(sys.argv) > 1):
    if((sys.argv[1])=='demo' ):
       trn.show_graphics()


#Fetch Train data:
X_train_tfidf = trn.tf_idf_vector
Y_train = trn.Y_train


'''
#Read and clean test data file:
test_filename = "ftr_final_dataset_test.csv"
test_news = pd.read_csv(test_filename)
Y_test = test_news['class']

test_news['text'] = dc.merge_and_clean(test_news)
'''

X_test = split.X_test
X_test_tfidf = trn.tfidf_transformer.transform(trn.countV.transform(X_test))
Y_test = split.y_test


# Preparing models:

names = ["Multinmomial Naive Bayes", "Logistic Regression", "Linear SVM", "Linear Stochastic Gradient Descent Classifier",
         "Random Forest Classifier"]

classifiers = [
	MultinomialNB(),
	LogisticRegression(solver='lbfgs'),
	LinearSVC(dual=False),
	SGDClassifier(loss='hinge', penalty='l2', alpha=1e-6, max_iter=1000000, tol=1e-3),
	RandomForestClassifier(n_estimators=200,n_jobs=3)]


#clf = MultinomialNB()
from sklearn import metrics
final_clf = classifiers[0]
best_f1_score = 0.0
epsilon = 0.005
for name, clf in zip(names, classifiers):
	clf.fit(X_train_tfidf, Y_train)
	Y_pred_class = clf.predict(X_test_tfidf)
	#print(clf.predict_proba(X_test_tfidf))
	##################################################################

	print("-----------" + name + "-----------")
	print("ACCURACY  =   " + str(metrics.accuracy_score(Y_test, Y_pred_class)))
	print("F1 Score  =   " + str(metrics.f1_score(Y_test, Y_pred_class)))
	score = metrics.f1_score(Y_test, Y_pred_class)
	if (score > (best_f1_score+epsilon)):
		print("Found better performing model "+name)
		best_f1_score = score
		final_clf = clf
		final_name = name
#
# Save the model with best score

#pipeline_final = Pipeline([
#        (final_name+'_tfidf',TfidfVectorizer(strip_accents='unicode', stop_words=trn.stopwords, max_features=10000, #ngram_range=(1,3),use_idf=True,smooth_idf=False)),
#        (final_name+'_clf', final_clf)
#        ])

#pipeline_final.fit(split.X_train, split.y_train)
#predicted_final = pipeline_final.predict(split.X_test)
#

print("Saving "+final_name+" Model for future use for predicting")
pickle.dump(final_clf,open('ftr_final_model.sav','wb'))


