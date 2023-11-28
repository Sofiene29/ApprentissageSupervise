import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle



# Charger les données
features_file = 'acsincome_ca_features.csv'
labels_file = 'acsincome_ca_labels.csv'

features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)

# Mélanger les données
X_all, y_all = shuffle(features, labels, random_state=1)

# Sélectionner une fraction du dataset (par exemple, 10%)
num_samples = int(len(X_all) * 0.1)
#um_samples = int(len(X_all) * 0.5)
X, y = X_all[:num_samples], y_all[:num_samples]

# Standardiser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparer en train set et test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

# Initialisation du modèle Random Forest
rf_model = RandomForestClassifier(random_state=1)

# Entraînement du modèle Random Forest
rf_model.fit(X_train, y_train.values.ravel())

# Prédiction sur l'ensemble de test
y_pred_rf = rf_model.predict(X_test)

# Évaluation du modèle
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Affichage des performances du modèle
print(f"Accuracy pour Random Forest: {accuracy_rf}")
print(f"Rapport de classification pour Random Forest:\n{report_rf}")
print(f"Matrice de confusion pour Random Forest:\n{conf_matrix_rf}")
