from sklearn.ensemble import GradientBoostingClassifier
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
num_samples = int(len(X_all) * 0.2)
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

# Initialisation du modèle Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=1)
# Validation croisée
gb_cv_scores = cross_val_score(gb_model, X_train, y_train.values.ravel(), cv=5)


print(f"Scores de validation croisée pour Gradient Boosting: {gb_cv_scores}")
print(f"Moyenne des scores de validation croisée: {gb_cv_scores.mean()}")

print("-----------------------------------------------------------")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# Entraînement du modèle Gradient Boosting sur l'ensemble d'entraînement
gb_model.fit(X_train, y_train.values.ravel())

# Prédiction sur l'ensemble de test
y_pred_gb = gb_model.predict(X_test)

# Évaluation du modèle
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)

print(f"Accuracy pour Gradient Boosting: {accuracy_gb}")
print(f"Rapport de classification pour Gradient Boosting:\n{report_gb}")
print(f"Matrice de confusion pour Gradient Boosting:\n{conf_matrix_gb}")


print("------------------------------------------------------------------")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Initialisation du modèle Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=1)

# Définition de la grille des hyperparamètres à tester
param_grid = {
    'n_estimators': [100, 200],  # Nombre d'arbres dans le modèle
    'learning_rate': [0.01, 0.1, 0.5],  # Taux d'apprentissage
    'max_depth': [3, 5],  # Profondeur maximale de chaque arbre
    'min_samples_split': [2, 4]  # Nombre minimal d'échantillons requis pour diviser un nœud
}

# Création de l'objet GridSearchCV
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Lancement de la recherche sur la grille
grid_search.fit(X_train, y_train.values.ravel())

# Affichage des meilleurs paramètres
print("Meilleurs paramètres trouvés : ", grid_search.best_params_)

# Entraînement du modèle avec les meilleurs paramètres sur l'ensemble d'entraînement complet
best_gb_model = grid_search.best_estimator_

# Prédiction sur l'ensemble de test
y_pred_best_gb = best_gb_model.predict(X_test)

# Évaluation du modèle optimisé
accuracy_best_gb = accuracy_score(y_test, y_pred_best_gb)
report_best_gb = classification_report(y_test, y_pred_best_gb)
conf_matrix_best_gb = confusion_matrix(y_test, y_pred_best_gb)

print(f"Accuracy optimisé pour Gradient Boosting: {accuracy_best_gb}")
print(f"Rapport de classification optimisé pour Gradient Boosting:\n{report_best_gb}")
print(f"Matrice de confusion optimisée pour Gradient Boosting:\n{conf_matrix_best_gb}")





