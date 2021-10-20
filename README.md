
# Classification du cancer du sein ML
Nous sommes en plein mois de prévention du cancer du sein, voilà une bonne occasion d'implémenter un modéle prédictant le caractére bénin ou malin dans un jeu de données 


## Bréve explication du fonctionnement

La première importante étape est d'importer les librairies utile a ce projet.
On importe par exemple la fonction "train_test_split" pour séparer nos donnée d'entrainement à nos données testeuses. On a utilisé le modéle de régresion logistique et la fonction "accuracy_score" afin d'évaluer le pourcentage de prédictions correctes.
```bash
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
```
Ici on a chargé le jeu de données de la librairie sklearn.
```bash
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
```
On charge notre donnée dans un dataframe.
```bash
data_frame['Rubrique'] = breast_cancer_dataset.target
```
On ajoute la colomne Rubrique et pointe 0 ou 1 s'il est bénin ou malin
```bash
data_frame.groupby('Rubrique').mean()
```
La partie la plus importante est ici, on regroupe les données et remarquons que les valeurs sont légérement plus élevées pour les cas malin que bénin
```bash
prediction = model.predict(input_data_reshaped)
print(prediction) 
if (prediction[0] == 0):
    print('Le cancer du sein est malin')

else:
  print('Le cancer du sein est bénin')
```
La derniere étape est de construire un système qui predicte qui nous informe si le cancer est bénin ou malin