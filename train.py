import ftr_data_clean as dc
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import split


X_train = split.X_train 
Y_train = split.y_train

demo = True


###########################################################

def visualize_data():
    print("***** Data Exploration in progress ****")
    print()
    print(" Showing graph of distribution of TRUE and FALSE labels in training set:  ")
    print()
    print("****************************************************************************")
    #distribution of classes for prediction
    sns.countplot(x=Y_train, palette='hls')
    plt.show()

    
    print("*** Top 25 Common Words ***")
    #Identify common words
    freq = pd.Series(' '.join(X_train).split()).value_counts()[:25]
    print(freq)

    print("*** 25 Rare Words  ***")
    freq1 =  pd.Series(' '.join(X_train).split()).value_counts()[-25:]
    print(freq1)

    

###########################################################

#Count-Vectorization

def get_countVectorizer_stats(train_count):
    
    print("Number of articles fed for training model: " + str(train_count.shape[0]))
    print("Number of words considered as features for model: " + str(train_count.shape[1]))


###########################################################

def get_top_k_words(corpus, k=None, wcnt=3):
    vec = CountVectorizer(max_df=0.75, strip_accents='unicode', stop_words=stopwords, max_features=10000, ngram_range=(wcnt,wcnt)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    top_words = words_freq[:k]
    top_df = pd.DataFrame(top_words)
    x_label = str(wcnt)+'_gram'
    
    '''
    print("Top K "+x_label+" frequencies")
    print(top_df)
    bot_words = words_freq[-k:]
    bot_df = pd.DataFrame(bot_words)
    print("Bottom K "+x_label+" frequencies")
    print(bot_df)
    '''

    #Barplot of most freq words
    top_df.columns=[x_label, "Freq"]
    title = x_label+"_histogram_"+split.dataset_name
    sns.set(rc={'figure.figsize':(13,8)})
    g = sns.barplot(x=x_label, y="Freq", data=top_df)
    g.set_xticklabels(g.get_xticklabels(), rotation=30)
    plt.title(title)
    #plt.savefig('./graphs/'+title+'.png')
    plt.savefig('{}.png'.format(title))
    plt.show()



def Visualise_ngrams():    
    print("**** Plotting Top 20 Uni-gram Histograms ****")
    get_top_k_words(X_train, k=20, wcnt=1)
    #ignore = input("Press ENTER to continue\n")
    print("**** Plotting Top 20 Bi-gram Histograms ****")
    get_top_k_words(X_train, k=20, wcnt=2)
    #ignore = input("Press ENTER to continue\n")
    #print("**** Plotting Top 20 Tri-gram Histograms ****")
    #get_top_k_words(train_news['body_text'], k=20, wcnt=3)
    #ignore = input("Press ENTER to continue\n")


###########################################################

# TF-IDF Characteristics of the corpus


def sort_coo(coo_matrix, rev):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=rev)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results





def print_tfidf_stats(keywords):
    print("\n===Top 10 Keywords===")
    for k in keywords:
        print(k,keywords[k])
    input("*** Press ENTER ***")

    print("\n===Bottom 10 Keywords===")
    #sort the tf-idf vectors by Ascending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo(), rev=False)
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)

    for k in keywords:
        print(k,keywords[k])
    input("*** Press ENTER ***")


##########################################################

def show_graphics():
    visualize_data()
    ignore = input("Press ENTER to continue\n")

    print("Count Vectorization donw using BoWs, bi-grams and tri-grams: ")
    print(" ")
    get_countVectorizer_stats(train_count)
    ignore = input("Press ENTER to continue\n")

    Visualise_ngrams()

    print_tfidf_stats(keywords)


##########################################################



stopwords=dc.get_stop_words("./Mystopwords.txt")

    



countV = CountVectorizer(max_df=0.75, strip_accents='unicode', stop_words=stopwords, max_features=10000, ngram_range=(1,3))
train_count = countV.fit_transform(X_train.values)
    

# Perform TF-IDF
feature_names=countV.get_feature_names()

# Compute IDF for training dataset
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(train_count)
tf_idf_vector=tfidf_transformer.transform(countV.transform(X_train))

# Sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo(), rev=True)

#Extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    

##########################################################
    
