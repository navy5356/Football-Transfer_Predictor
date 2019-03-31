# -*- coding: utf-8 -*-

###########################################################################
#
#	Project	: Predicting validity of Football Transfer Rumours (for BITS-Pilani APOGEE 2019)	
#	File	: ftr_data_preprocess.py
#	Created : Sun 3 Mar 2019
#	Moiefied: 8/Mar/2019	
#	Author	: Navaneeth Raghunath
#	File	: Module to load dataset and check data quality
###########################################################################

import pandas as pd
import csv
import re
import numpy as np

import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt


##################################################################

def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
###################################################################

def merge_and_clean(news):
    print("***** Data Cleaning in progress ****")
    news['text'] = news['headline']+' '+news['body_text']
    news['text'] = news['text'].apply(lambda x:pre_process(x))
    return news['text']

###################################################################


def get_stop_words(stop_file_path):

    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


###################################################################


