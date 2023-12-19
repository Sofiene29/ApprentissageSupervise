import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

# Charger les données
features = pd.read_csv('acsincome_ca_features.csv')
labels = pd.read_csv('acsincome_ca_labels.csv')

# Mélanger les données et sélectionner 20% du dataset
features, labels = shuffle(features, labels, random_state=1)
num_samples = int(len(features) * 0.2)
features, labels = features.iloc[:num_samples, :], labels.iloc[:num_samples, :]

# Standardiser les features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Séparation en train set et test set
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=1)

# Initialisation des meilleurs modèles avec les paramètres obtenus de GridSearch
svm_model = SVC(C=1, gamma='scale', kernel='rbf', random_state=1)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=1)
adaboost_model = AdaBoostClassifier(n_estimators=150, learning_rate=1, random_state=1)
gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_split=2, random_state=1)

# Entraînement des modèles
svm_model.fit(X_train, y_train.values.ravel())
rf_model.fit(X_train, y_train.values.ravel())
adaboost_model.fit(X_train, y_train.values.ravel())
gb_model.fit(X_train, y_train.values.ravel())

# a. Calcul des corrélations initiales entre features et label
corr_initial = pd.DataFrame(features).corrwith(pd.DataFrame(labels).iloc[:, 0])
print("Corrélations initiales entre features et label :")
print(corr_initial)

# b. Calcul des corrélations entre features et labels prédits
def calculate_correlations(X_test, predictions):
    if len(set(predictions)) > 1:
        return pd.DataFrame(X_test).corrwith(pd.Series(predictions))
    else:
        return "Toutes les prédictions sont identiques, impossible de calculer la corrélation."

svm_predictions = svm_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
adaboost_predictions = adaboost_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

corr_svm = calculate_correlations(X_test, svm_predictions)
corr_rf = calculate_correlations(X_test, rf_predictions)
corr_adaboost = calculate_correlations(X_test, adaboost_predictions)
corr_gb = calculate_correlations(X_test, gb_predictions)

print("Corrélations avec labels prédits par SVM :")
print(corr_svm)
print("Corrélations avec labels prédits par Random Forest :")
print(corr_rf)
print("Corrélations avec labels prédits par AdaBoost :")
print(corr_adaboost)
print("Corrélations avec labels prédits par Gradient Boosting :")
print(corr_gb)

# c. Évaluation de l'importance des features avec permutation_importance
importance_rf = permutation_importance(rf_model, X_train, y_train.values.ravel(), n_repeats=10, random_state=1)
importance_adaboost = permutation_importance(adaboost_model, X_train, y_train.values.ravel(), n_repeats=10, random_state=1)
importance_gb = permutation_importance(gb_model, X_train, y_train.values.ravel(), n_repeats=10, random_state=1)
importance_svm = permutation_importance(svm_model, X_train, y_train.values.ravel(), n_repeats=10, random_state=1)

# Afficher l'importance de chaque feature pour chaque modèle
for model_name, importances in zip(['Random Forest', 'AdaBoost', 'Gradient Boosting', 'SVM'],
                                   [importance_rf, importance_adaboost, importance_gb, importance_svm]):
    print(f"\nImportance des features pour {model_name}:")
    for i, imp in enumerate(importances.importances):
        print(f"Feature {i}: Importance moyenne = {imp.mean()}, Écart-type = {imp.std()}")
