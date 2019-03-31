# Football-Transfer_Predictor
An NLP-based fake news detection algorithm applied for the particular field of Football Transfer News

The objective of this project is to learn basic NLP techniques and ML Algorithms and their implementation in Python.
Currently, the scope is limited  to 
	a) using simple BoW and n_gram representations with CountVector and TF-IDF encodings.
	b) applying Simple Classfiers such as Multinomial Naive Bayes, LogisticRegression and SVM.
	c) The rumour data set being used is small and under construction. It is modeled after the LIAR Dataset but is simplified for binary labels.
  
  The Python versions and packages being used are:
	Python 3.7.1
	conda-4.6.7
	numpy-1.16.2
	pandas-0.23.4
	scikit-learn-0.20.2
	seaborn-0.9.0

Stopwords File
	./Data/MyStopwords.txt
  
 Corpus File
  ftr_articles_raw.csv
  This file is created by scrapping the web for pre-specified list of popular 40 popular European footbal related websites such as Marca.com, Goal.com, etc.
  Scrapping is done using scrapy web crawler engine and spiders for each of the website.
  Results are filtered for English language aticles only. Results with empty headline or body text are filtered out. Body Text may have Non-English langauge text mixed with English.
  Some HTML Markups and Links may also be present.
