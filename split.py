from sklearn.model_selection import train_test_split
import csv
import ftr_data_clean as dc
import pandas as pd

dataset_name = 'ftr_final_dataset_unsplit'
filename = dataset_name+'.csv'

articles = pd.read_csv(filename)


articles['text'] = dc.merge_and_clean(articles) 
print("Splitting Dataset into training (75% and test datasets (25%)")
X_train, X_test, y_train, y_test = train_test_split(articles['text'], articles['class'], test_size=0.20, random_state=55)

