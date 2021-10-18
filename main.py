import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #ca me permet de savoir combien de bonnes prédictions mon modéle fait

#Prise des données de sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset) # ca va display tous les valeurs de ma dataset avec le nm des colones et "0" et "1" pour benin ou malin

#Utilisation de panda pour facilité d'analyse des données
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
data_frame.head() #display les 5 premieres valeurs de mon datatset mais bcp mieux organisé comparé a précédement
data_frame.tail()

data_frame['Rubrique'] = breast_cancer_dataset.target
#nmbre de colomnes et rangées
data_frame.shape
#prise d'info dur la données traitée (type, colomne etc...)
data_frame.info()
# je verifie s'il manque des valeur dans les colomne, normalement il devrait y en avoir 0
data_frame.isnull().sum()


data_frame['Rubrique'].value_counts() #
data_frame.groupby('Rubrique').mean() #regroupe les cas malin et bénin en deux rangées 0 et 1


