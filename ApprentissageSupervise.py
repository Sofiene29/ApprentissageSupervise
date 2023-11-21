import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

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



# Initialisation du modèle RandomForest
rf_model = RandomForestClassifier(random_state=1)

# 1. Validation Croisée

cv_scores = cross_val_score(rf_model, X_train, y_train.values.ravel(), cv=5)
print(f"Scores de validation croisée pour RandomForest: {cv_scores}")
print(f"Moyenne des scores de validation croisée: {cv_scores.mean()}")

print("-----------------------------------------------------------")

# 2. Entraînement et Évaluation du Modèle
rf_model.fit(X_train, y_train.values.ravel())
y_pred_rf = rf_model.predict(X_test)

# Prédiction sur l'ensemble de test
y_pred_rf = rf_model.predict(X_test)

# Évaluation du modèle
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Accuracy pour RandomForest: {accuracy_rf}")
print(f"Rapport de classification pour RandomForest:\n{report_rf}")
print(f"Matrice de confusion pour RandomForest:\n{conf_matrix_rf}")

print("-----------------------------------------------------------")

# 3. Recherche des Hyperparamètres


# Définition de la grille de paramètres
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Mise en place de GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Recherche des meilleurs paramètres
grid_search.fit(X_train, y_train.values.ravel())

# Affichage des meilleurs paramètres
best_params = grid_search.best_params_
print("Meilleurs paramètres trouvés : ", best_params)


# Réinitialisation du modèle RandomForest avec les meilleurs paramètres
rf_optimized = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=1
)

# Entraînement du modèle
rf_optimized.fit(X_train, y_train.values.ravel())

# Prédiction sur l'ensemble de test
y_pred_optimized = rf_optimized.predict(X_test)

# Réévaluation du modèle
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
report_optimized = classification_report(y_test, y_pred_optimized)
conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"Accuracy optimisé pour RandomForest: {accuracy_optimized}")
print(f"Rapport de classification optimisé pour RandomForest:\n{report_optimized}")
print(f"Matrice de confusion optimisée pour RandomForest:\n{conf_matrix_optimized}")












