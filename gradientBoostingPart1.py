import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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

# Mélanger et sélectionner une fraction du dataset
X_all, y_all = shuffle(features, labels, random_state=1)
num_samples = int(len(X_all) * 0.2)
X, y = X_all[:num_samples], y_all[:num_samples]

# Standardiser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Initialisation du modèle Gradient Boosting et validation croisée
gb_model = GradientBoostingClassifier(random_state=1)
gb_cv_scores = cross_val_score(gb_model, X_train, y_train.values.ravel(), cv=5)
print(f"Scores de validation croisée pour Gradient Boosting: {gb_cv_scores}")
print(f"Moyenne des scores de validation croisée: {gb_cv_scores.mean()}")

# Entraînement et évaluation du modèle
gb_model.fit(X_train, y_train.values.ravel())
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
print(f"Accuracy pour Gradient Boosting: {accuracy_gb}")
print(f"Rapport de classification pour Gradient Boosting:\n{report_gb}")
print(f"Matrice de confusion pour Gradient Boosting:\n{conf_matrix_gb}")

# Optimisation des hyperparamètres avec GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5],
    'min_samples_split': [2, 4]
}
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train.values.ravel())
print("Meilleurs paramètres trouvés : ", grid_search.best_params_)

# Entraînement du modèle optimisé
best_gb_model = grid_search.best_estimator_
y_pred_best_gb = best_gb_model.predict(X_test)
accuracy_best_gb = accuracy_score(y_test, y_pred_best_gb)
report_best_gb = classification_report(y_test, y_pred_best_gb)
conf_matrix_best_gb = confusion_matrix(y_test, y_pred_best_gb)
print(f"Accuracy optimisé pour Gradient Boosting: {accuracy_best_gb}")
print(f"Rapport de classification optimisé pour Gradient Boosting:\n{report_best_gb}")
print(f"Matrice de confusion optimisée pour Gradient Boosting:\n{conf_matrix_best_gb}")

# Préparation des données du Nevada et du Colorado
X_nv_scaled = scaler.transform(features_nv)
X_co_scaled = scaler.transform(features_co)

# Prédiction et évaluation sur le Nevada
y_pred_nv_gb = best_gb_model.predict(X_nv_scaled)
accuracy_nv_gb = accuracy_score(labels_nv, y_pred_nv_gb)
report_nv_gb = classification_report(labels_nv, y_pred_nv_gb)
conf_matrix_nv_gb = confusion_matrix(labels_nv, y_pred_nv_gb)
print("Evaluation sur le Nevada")
print(f"Accuracy pour Gradient Boosting (Nevada): {accuracy_nv_gb}")
print(f"Rapport de classification (Nevada):\n{report_nv_gb}")
print(f"Matrice de confusion (Nevada):\n{conf_matrix_nv_gb}")

# Prédiction et évaluation sur le Colorado
y_pred_co_gb = best_gb_model.predict(X_co_scaled)
accuracy_co_gb = accuracy_score(labels_co, y_pred_co_gb)
report_co_gb = classification_report(labels_co, y_pred_co_gb)
conf_matrix_co_gb = confusion_matrix(labels_co, y_pred_co_gb)
print("Evaluation sur le Colorado")
print(f"Accuracy pour Gradient Boosting (Colorado): {accuracy_co_gb}")
print(f"Rapport de classification (Colorado):\n{report_co_gb}")
print(f"Matrice de confusion (Colorado):\n{conf_matrix_co_gb}")
