# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Charger le dataset depuis un fichier CSV
df = pd.read_csv('/Users/dongok/Downloads/auto-mpg.csv')

# Sélectionner 'mpg' comme caractéristique (X) et 'displacement' comme cible (y)
X = df[['mpg']]  # X est une DataFrame
y = df['displacement']  # y est une Series

# Diviser le dataset en 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Instancier le modèle de régression linéaire
lr = LinearRegression()

# Entraîner le modèle sur l'ensemble d'entraînement
lr.fit(X_train, y_train)

# Prédire les valeurs pour l'ensemble de test
y_pred = lr.predict(X_test)

# Évaluer l'erreur quadratique moyenne de l'ensemble de test
mse = mean_squared_error(y_test, y_pred)
print(f"Erreur quadratique moyenne (MSE) de l'ensemble de test: {mse}")

# Utiliser matplotlib pour créer le graphique
plt.figure(figsize=(10, 6))  # Définir la taille du graphique
plt.scatter(X_test, y_test, color='blue', label='Valeurs réelles', alpha=0.6)  # Tracer les valeurs réelles
plt.plot(X_test, y_pred, color='red', label='Ligne de régression')  # Tracer la ligne de régression
plt.title('Régression linéaire: Displacement en fonction de MPG')  # Ajouter un titre
plt.xlabel('MPG')  # Nommer l'axe des x
plt.ylabel('Displacement')  # Nommer l'axe des y
plt.legend()  # Ajouter une légende
plt.show()  # Afficher le graphique