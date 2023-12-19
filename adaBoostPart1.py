import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Charger les données
features_file = 'acsincome_ca_features.csv'
labels_file = 'acsincome_ca_labels.csv'

features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)

# Chargement des données pour le Nevada et le Colorado
features_nv = pd.read_csv('acsincome_ne_allfeaturesTP2.csv')
labels_nv = pd.read_csv('acsincome_ne_labelTP2.csv')

features_co = pd.read_csv('acsincome_co_allfeaturesTP2.csv')
labels_co = pd.read_csv('acsincome_co_labelTP2.csv')

# Afficher les premières lignes pour inspection
print("Features Preview:")
print(features.head())
print("\nLabels Preview:")
print(labels.head())

# Mélanger les données
X_all, y_all = shuffle(features, labels, random_state=1)

# Sélectionner une fraction du dataset (par exemple, 20%)
num_samples = int(len(X_all) * 0.2)
X, y = X_all[:num_samples], y_all[:num_samples]

# Standardiser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train set et test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Initialisation du modèle AdaBoost
adaboost_model = AdaBoostClassifier(random_state=1)

# Validation croisée
adaboost_cv_scores = cross_val_score(adaboost_model, X_train, y_train.values.ravel(), cv=5)
print(f"Scores de validation croisée pour AdaBoost: {adaboost_cv_scores}")
print(f"Moyenne des scores de validation croisée: {adaboost_cv_scores.mean()}")

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
best_params_adaboost = grid_search_adaboost.best_params_
print("Meilleurs paramètres trouvés pour AdaBoost : ", best_params_adaboost)

# Réinitialisation du modèle AdaBoost avec les meilleurs paramètres
optimized_adaboost = AdaBoostClassifier(
    n_estimators=best_params_adaboost['n_estimators'],
    learning_rate=best_params_adaboost['learning_rate'],
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

# Préparation des données du Nevada et du Colorado
X_nv_scaled = scaler.transform(features_nv)
X_co_scaled = scaler.transform(features_co)

# Prédiction et évaluation sur le Nevada
y_pred_nv = optimized_adaboost.predict(X_nv_scaled)
accuracy_nv = accuracy_score(labels_nv, y_pred_nv)
report_nv = classification_report(labels_nv, y_pred_nv)
conf_matrix_nv = confusion_matrix(labels_nv, y_pred_nv)

print("Evaluation sur le Nevada")
print(f"Accuracy optimisé pour AdaBoost: {accuracy_nv}")
print(f"Rapport de classification optimisé pour AdaBoost:\n{report_nv}")
print(f"Matrice de confusion optimisée pour AdaBoost:\n{conf_matrix_nv}")

# Prédiction et évaluation sur le Colorado
y_pred_co = optimized_adaboost.predict(X_co_scaled)
accuracy_co = accuracy_score(labels_co, y_pred_co)
report_co = classification_report(labels_co, y_pred_co)
conf_matrix_co = confusion_matrix(labels_co, y_pred_co)

print("Evaluation sur le Colorado")
print(f"Accuracy optimisé pour AdaBoost: {accuracy_co}")
print(f"Rapport de classification optimisé pour AdaBoost:\n{report_co}")
print(f"Matrice de confusion optimisée pour AdaBoost:\n{conf_matrix_co}")
