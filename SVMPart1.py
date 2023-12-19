import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
print("-----------------------------------------------------------")

# Mélanger les données
X_all, y_all = shuffle(features, labels, random_state=1)

# Sélectionner une fraction du dataset (par exemple, 10%)
num_samples = int(len(X_all) * 0.2)
X, y = X_all[:num_samples], y_all[:num_samples]

# Standardiser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train set et test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Initialisation du modèle SVM
svm_model = SVC(random_state=1)

# Validation croisée pour SVM
svm_cv_scores = cross_val_score(svm_model, X_train, y_train.values.ravel(), cv=5)
print(f"Scores de validation croisée pour SVM: {svm_cv_scores}")
print(f"Moyenne des scores de validation croisée: {svm_cv_scores.mean()}")

# Entraînement du modèle SVM
svm_model.fit(X_train, y_train.values.ravel())

# Prédiction sur l'ensemble de test
y_pred_svm = svm_model.predict(X_test)

# Évaluation du modèle
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print(f"Accuracy pour SVM: {accuracy_svm}")
print(f"Rapport de classification pour SVM:\n{report_svm}")
print(f"Matrice de confusion pour SVM:\n{conf_matrix_svm}")
print("------------------------------------------------------")

# Définition de la grille de paramètres pour SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly']
}

# Mise en place de GridSearchCV pour le modèle SVM
grid_search_svm = GridSearchCV(estimator=SVC(random_state=1), param_grid=param_grid_svm, cv=5, n_jobs=-1, verbose=2)

# Recherche des meilleurs paramètres
grid_search_svm.fit(X_train, y_train.values.ravel())

# Affichage des meilleurs paramètres
best_params_svm = grid_search_svm.best_params_
print("Meilleurs paramètres trouvés pour SVM : ", best_params_svm)
print("------------------------------------------------------")

# Réinitialisation du modèle SVM avec les meilleurs paramètres trouvés
optimized_svm = SVC(C=best_params_svm['C'], gamma=best_params_svm['gamma'], kernel=best_params_svm['kernel'], random_state=1)

# Entraînement du modèle optimisé
optimized_svm.fit(X_train, y_train.values.ravel())

# Prédiction sur l'ensemble de test avec le modèle optimisé
y_pred_optimized_svm = optimized_svm.predict(X_test)

# Évaluation du modèle optimisé
accuracy_optimized_svm = accuracy_score(y_test, y_pred_optimized_svm)
report_optimized_svm = classification_report(y_test, y_pred_optimized_svm)
conf_matrix_optimized_svm = confusion_matrix(y_test, y_pred_optimized_svm)
print(f"Accuracy optimisé pour SVM: {accuracy_optimized_svm}")
print(f"Rapport de classification optimisé pour SVM:\n{report_optimized_svm}")
print(f"Matrice de confusion optimisée pour SVM:\n{conf_matrix_optimized_svm}")

# Préparation des données du Nevada et du Colorado
X_nv_scaled = scaler.transform(features_nv)
X_co_scaled = scaler.transform(features_co)

# Prédiction et évaluation sur le Nevada
y_pred_nv_svm = optimized_svm.predict(X_nv_scaled)
accuracy_nv_svm = accuracy_score(labels_nv, y_pred_nv_svm)
report_nv_svm = classification_report(labels_nv, y_pred_nv_svm)
conf_matrix_nv_svm = confusion_matrix(labels_nv, y_pred_nv_svm)

print("Evaluation sur le Nevada")
print(f"Accuracy optimisé pour SVM: {accuracy_nv_svm}")
print(f"Rapport de classification optimisé pour SVM:\n{report_nv_svm}")
print(f"Matrice de confusion optimisée pour SVM:\n{conf_matrix_nv_svm}")

# Prédiction et évaluation sur le Colorado
y_pred_co_svm = optimized_svm.predict(X_co_scaled)
accuracy_co_svm = accuracy_score(labels_co, y_pred_co_svm)
report_co_svm = classification_report(labels_co, y_pred_co_svm)
conf_matrix_co_svm = confusion_matrix(labels_co, y_pred_co_svm)

print("Evaluation sur le Colorado")
print(f"Accuracy optimisé pour SVM: {accuracy_co_svm}")
print(f"Rapport de classification optimisé pour SVM:\n{report_co_svm}")
print(f"Matrice de confusion optimisée pour SVM:\n{conf_matrix_co_svm}")
