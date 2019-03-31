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


# We use Pickle Persistence Store module for saving the final model
import pickle

import train as trn

def  load_ftr_model(model_file_name='ftr_final_model.sav'):
	print("Using Model file: "+model_file_name)
	loaded_model = pickle.load(open(model_file_name, 'rb'))
	print(loaded_model)
	return(loaded_model)


#function to run for prediction
def detecting_rumour_validity(model, var):    
#    tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True)
    statement_tfidf = trn.tfidf_transformer.transform(trn.countV.transform([var]))
    prediction = model.predict(statement_tfidf)
    return(prediction[0])


if __name__ == '__main__':
    model = load_ftr_model()
    loop = True
    while loop == True:
        rumor = input("State your transfer rumour or type **** to stop: ")

        if rumor == "****":
                print("Thanks for trying out FTR detector")
                loop = 	False
        else:
            print("Rumour: " + str(rumor))
            pred = detecting_rumour_validity(model, rumor)
            print("The rumour is likely to be ", pred)



