import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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

# Initialisation du modèle SVM
svm_model = SVC(random_state=1)

# 1.Validation croisée
svm_cv_scores = cross_val_score(svm_model, X_train, y_train.values.ravel(), cv=5)

print(f"Scores de validation croisée pour SVM: {svm_cv_scores}")
print(f"Moyenne des scores de validation croisée: {svm_cv_scores.mean()}")

print("-----------------------------------------------------------")


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



from sklearn.model_selection import GridSearchCV

# Définition de la grille de paramètres
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
print("Meilleurs paramètres trouvés pour SVM : ", grid_search_svm.best_params_)

print("------------------------------------------------------")
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# Réinitialisation du modèle SVM avec les meilleurs paramètres
optimized_svm = SVC(C=1, gamma='scale', kernel='rbf', random_state=1)

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




