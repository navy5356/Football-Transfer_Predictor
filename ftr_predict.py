# -*- coding: utf-8 -*-

###########################################################################
#
#	Project	: Predicting validity of Football Transfer Rumours  (for BITS-Pilani APOGEE 2019)
#	File	: ftr_predict.py
#	Created : Sun 3 Mar 2019
#	Modified: 30 Mar 2019
#	Author	: Navaneeth Raghunath
#	
###########################################################################

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import sys

# We use Pickle Persistence Store module for saving the final model
import pickle

import train as trn

if (len(sys.argv) > 1):
    if((sys.argv[1])=='demo' ):
        trn.show_graphics()

'''demo = True

if(demo):
    trn.show_graphics()
'''

model_file_name = 'ftr_final_model.sav'
print("Using Model file: "+model_file_name)


#function to run for prediction
def detecting_rumour_validity(var):    
#    tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)
    statement_tfidf = trn.tfidf_transformer.transform(trn.countV.transform([var]))


#retrieving the best model for prediction call
#    loaded_model = pickle.load(open(model_file_name, 'rb'))
#    print(loaded_model)
    prediction = loaded_model.predict(statement_tfidf)
#    prob = loaded_model.predict_proba([var])


    return(print("The rumour is ",prediction[0]))
#    return (print("The rumour is ",prediction),
#        print("The truth probability is ",prob[0][1]))


if __name__ == '__main__':
    loaded_model = pickle.load(open(model_file_name, 'rb'))
    print(loaded_model)
    loop = True

    while loop == True:
        rumor = input("State your transfer rumour or type **** to stop: ")

        if rumor == "****":
                print("Thanks for trying out FTR detector")
                loop = 	False
        else:
            print("Rumour: " + str(rumor))
            detecting_rumour_validity(rumor)



