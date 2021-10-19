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

# J'isole ma colone rubrique 
X = data_frame.drop(columns='Rubrique', axis=1)
Y = data_frame['Rubrique']
print(X)
print(Y)

#Je test mes donnée en données "testeuse" et donnée d'entrainement
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2) #20% de mes données seront des données "testetuse" et le reste des données d'netrainement 
print(X.shape, X_train.shape, X_test.shape)

#J'ajuste l'ensemble de mes données "X_train" et "Y_train" à mon modele de régression logistique
model = LogisticRegression()

model.fit(X_train, Y_train) #ici le modélé va chercher la realtoin entre cex deux données donc le svaleurs de notre tableau

# Je predicte si le résultat des mes données "X_train" sera 0 ou 1 en utilisant le modéle et en les comparant aux vraies valeurs Y_train
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Exactitude sur les training data = ', training_data_accuracy)

#IDEM pour les données X_test et Y_test
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Exactitude sur les test data = ', test_data_accuracy)

# Données de kaggle
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
#Conversion des input en numpy array
input_data_as_numpy_array = np.asarray(input_data)
#Là je montre au modéle que je veux uniquement predire le reuslatt pour une seule valeur
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction) 
if (prediction[0] == 0):
    print('Le cancer du sein est malin')

else:
  print('Le cancer du sein est bénin')