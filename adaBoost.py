from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# Charger les données
features_file = 'acsincome_ca_features.csv'
labels_file = 'acsincome_ca_labels.csv'

features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)

# Afficher les premières lignes pour inspection
print("Features Preview:")
print(features.head())
print("\nLabels Preview:")
print(labels.head())

print("-----------------------------------------------------------")


# Mélanger les données
X_all, y_all = shuffle(features, labels, random_state=1)

# Sélectionner une fraction du dataset (par exemple, 10%)
num_samples = int(len(X_all) * 0.1)
X, y = X_all[:num_samples], y_all[:num_samples]

# Afficher les tailles des datasets pour vérification
print(f"Taille du dataset complet: {len(X_all)}")
print(f"Taille du sous-ensemble sélectionné: {len(X)}")

print("-----------------------------------------------------------")

# Standardiser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train set et test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Afficher les tailles des ensembles pour vérification
print(f"Taille de l'ensemble d'entraînement: {X_train.shape}")
print(f"Taille de l'ensemble de test: {X_test.shape}")

print("-----------------------------------------------------------")


# Initialisation du modèle AdaBoost
adaboost_model = AdaBoostClassifier(random_state=1)

# Validation croisée
adaboost_cv_scores = cross_val_score(adaboost_model, X_train, y_train.values.ravel(), cv=5)

print(f"Scores de validation croisée pour AdaBoost: {adaboost_cv_scores}")
print(f"Moyenne des scores de validation croisée: {adaboost_cv_scores.mean()}")

print("-----------------------------------------------------------")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Entraînement du modèle AdaBoost
adaboost_model.fit(X_train, y_train.values.ravel())

# Prédiction sur l'ensemble de test
y_pred_adaboost = adaboost_model.predict(X_test)

# Évaluation du modèle
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
report_adaboost = classification_report(y_test, y_pred_adaboost)
conf_matrix_adaboost = confusion_matrix(y_test, y_pred_adaboost)

print(f"Accuracy pour AdaBoost: {accuracy_adaboost}")
print(f"Rapport de classification pour AdaBoost:\n{report_adaboost}")
print(f"Matrice de confusion pour AdaBoost:\n{conf_matrix_adaboost}")


print("-----------------------------------------------------------")


# Initialisation du modèle AdaBoost
adaboost_model = AdaBoostClassifier(random_state=1)

# Définition de la grille de paramètres
param_grid_adaboost = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1]
}

# Mise en place de GridSearchCV pour le modèle AdaBoost
grid_search_adaboost = GridSearchCV(estimator=adaboost_model, param_grid=param_grid_adaboost, cv=5, n_jobs=-1, verbose=2)

# Recherche des meilleurs paramètres
grid_search_adaboost.fit(X_train, y_train.values.ravel())

# Affichage des meilleurs paramètres
print("Meilleurs paramètres trouvés pour AdaBoost : ", grid_search_adaboost.best_params_)

# Réinitialisation du modèle AdaBoost avec les meilleurs paramètres
optimized_adaboost = AdaBoostClassifier(
    n_estimators=grid_search_adaboost.best_params_['n_estimators'],
    learning_rate=grid_search_adaboost.best_params_['learning_rate'],
    random_state=1
)

# Entraînement du modèle optimisé
optimized_adaboost.fit(X_train, y_train.values.ravel())

# Prédiction sur l'ensemble de test avec le modèle optimisé
y_pred_optimized_adaboost = optimized_adaboost.predict(X_test)

# Évaluation du modèle optimisé
accuracy_optimized_adaboost = accuracy_score(y_test, y_pred_optimized_adaboost)
report_optimized_adaboost = classification_report(y_test, y_pred_optimized_adaboost)
conf_matrix_optimized_adaboost = confusion_matrix(y_test, y_pred_optimized_adaboost)

print(f"Accuracy optimisé pour AdaBoost: {accuracy_optimized_adaboost}")
print(f"Rapport de classification optimisé pour AdaBoost:\n{report_optimized_adaboost}")
print(f"Matrice de confusion optimisée pour AdaBoost:\n{conf_matrix_optimized_adaboost}")


