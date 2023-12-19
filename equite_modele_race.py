import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils import shuffle

# Charger les données
features = pd.read_csv('acsincome_ca_features.csv')
labels = pd.read_csv('acsincome_ca_labels.csv')
race_data = pd.read_csv('acsincome_ca_group_Race.csv')

# Mélanger les données et réduire la taille du dataset à 20%
features, labels, race_data = shuffle(features, labels, race_data, random_state=1)
subset_size = int(0.2 * len(features))
features, labels, race_data = features.iloc[:subset_size], labels.iloc[:subset_size], race_data.iloc[:subset_size]

# Séparer en train set et test set
X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(features, labels, race_data, test_size=0.2, random_state=1)

# Aligner les index pour race_data avec X_test
race_test.index = X_test.index
y_test.index = X_test.index

# Standardiser les features (avec et sans RAC1P)
scaler = StandardScaler()
X_train_scaled_with_race = scaler.fit_transform(X_train)
X_test_scaled_with_race = scaler.transform(X_test)

X_train_scaled_no_race = scaler.fit_transform(X_train.drop('RAC1P', axis=1))
X_test_scaled_no_race = scaler.transform(X_test.drop('RAC1P', axis=1))

# Définir les modèles
models = {
    'SVM': SVC(C=1, gamma='scale', kernel='rbf', random_state=1),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=1),
    'AdaBoost': AdaBoostClassifier(n_estimators=150, learning_rate=1, random_state=1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_split=2, random_state=1)
}

# a. Calculer la matrice de confusion pour chaque race et chaque modèle
for model_name, model in models.items():
    model.fit(X_train_scaled_with_race, y_train.values.ravel())
    y_pred = model.predict(X_test_scaled_with_race)

    for race in race_data['RAC1P'].unique():
        race_mask = race_test['RAC1P'] == race
        cm = confusion_matrix(y_test[race_mask], y_pred[race_mask])
        print(f"{model_name} - Race {race}:\n{cm}\n")
# b. et c. Calculer les métriques d'équité avec et sans la feature 'RAC1P'
def calculate_equity_metrics(y_test, y_pred, race_data):
    metrics = {}
    for race in race_data['RAC1P'].unique():
        race_mask = race_data['RAC1P'] == race
        cm = confusion_matrix(y_test[race_mask], y_pred[race_mask])
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else 0
        tpr = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm.shape == (2, 2) else 0
        metrics[race] = {'FPR': fpr, 'TPR': tpr}
    return metrics

# Analyse avec la feature 'RAC1P'
for model_name, model in models.items():
    model.fit(X_train_scaled_with_race, y_train.values.ravel())
    y_pred_with_race = model.predict(X_test_scaled_with_race)
    metrics_with_race = calculate_equity_metrics(y_test, y_pred_with_race, race_test)
    print(f"{model_name} - Métriques d'équité avec RAC1P:\n{metrics_with_race}\n")

# Analyse sans la feature 'RAC1P'
for model_name, model in models.items():
    model.fit(X_train_scaled_no_race, y_train.values.ravel())
    y_pred_no_race = model.predict(X_test_scaled_no_race)
    metrics_no_race = calculate_equity_metrics(y_test, y_pred_no_race, race_test)
    print(f"{model_name} - Métriques d'équité sans RAC1P:\n{metrics_no_race}\n")
