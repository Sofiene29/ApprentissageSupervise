import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils import shuffle

# Charger les données
features_file = 'acsincome_ca_features.csv'
labels_file = 'acsincome_ca_labels.csv'
sex_data_file = 'acsincome_ca_group_Sex.csv'

features = pd.read_csv(features_file)
labels = pd.read_csv(labels_file)
sex_data = pd.read_csv(sex_data_file)

# Mélanger les données et sélectionner une fraction du dataset
X_all, y_all, sex_all = shuffle(features, labels, sex_data, random_state=1)
num_samples = int(len(X_all) * 0.2)
X, y, sex_data_filtered = X_all[:num_samples], y_all[:num_samples], sex_all[:num_samples]

# Séparer en train set et test set
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(X, y, sex_data_filtered, test_size=0.2, random_state=1)

# Aligner les index
sex_test.index = X_test.index
y_test.index = X_test.index

# Standardiser les features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définir les modèles
models = {
    'SVM': SVC(C=1, gamma='scale', kernel='rbf', random_state=1),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, min_samples_leaf=4, random_state=1),
    'AdaBoost': AdaBoostClassifier(n_estimators=150, learning_rate=1, random_state=1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, min_samples_split=2, random_state=1)
}

# a. Calculer la matrice de confusion pour chaque sexe et chaque modèle
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train.values.ravel())
    y_pred = model.predict(X_test_scaled)

    for sex in [1, 2]:  # 1 pour Male, 2 pour Female
        sex_mask = sex_test['SEX'] == sex
        cm = confusion_matrix(y_test[sex_mask], y_pred[sex_mask])
        print(f"{model_name} - Sexe {sex}:\n{cm}\n")


# b. et c. Calculer les métriques d'équité avec et sans la feature 'SEX'
def calculate_equity_metrics(y_test, y_pred, sex_data):
    cm_male = confusion_matrix(y_test[sex_data['SEX'] == 1], y_pred[sex_data['SEX'] == 1])
    cm_female = confusion_matrix(y_test[sex_data['SEX'] == 2], y_pred[sex_data['SEX'] == 2])
    fpr_male = cm_male[0, 1] / (cm_male[0, 0] + cm_male[0, 1]) if cm_male[0, 0] + cm_male[0, 1] > 0 else 0
    fpr_female = cm_female[0, 1] / (cm_female[0, 0] + cm_female[0, 1]) if cm_female[0, 0] + cm_female[0, 1] > 0 else 0
    tpr_male = cm_male[1, 1] / (cm_male[1, 0] + cm_male[1, 1]) if cm_male[1, 0] + cm_male[1, 1] > 0 else 0
    tpr_female = cm_female[1, 1] / (cm_female[1, 0] + cm_female[1, 1]) if cm_female[1, 0] + cm_female[1, 1] > 0 else 0
    return {'FPR Male': fpr_male, 'FPR Female': fpr_female, 'TPR Male': tpr_male, 'TPR Female': tpr_female}

# Analyse avec la feature 'SEX'
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train.values.ravel())
    y_pred = model.predict(X_test_scaled)
    metrics = calculate_equity_metrics(y_test, y_pred, sex_test)
    print(f"{model_name} - Métriques d'équité avec SEX:\n{metrics}\n")

# Enlever la feature 'SEX' et réanalyser
X_train_no_sex_scaled = scaler.fit_transform(X_train.drop('SEX', axis=1))
X_test_no_sex_scaled = scaler.transform(X_test.drop('SEX', axis=1))

for model_name, model in models.items():
    model.fit(X_train_no_sex_scaled, y_train.values.ravel())
    y_pred_no_sex = model.predict(X_test_no_sex_scaled)
    metrics_no_sex = calculate_equity_metrics(y_test, y_pred_no_sex, sex_test)
    print(f"{model_name} - Métriques d'équité sans SEX:\n{metrics_no_sex}\n")