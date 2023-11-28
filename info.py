import matplotlib.pyplot as plt
import seaborn as sns

# Matrices de confusion pour Random Forest et SVM
conf_matrix_rf = [[1950, 336], [400, 1228]]
conf_matrix_svm = [[1940, 346], [424, 1204]]

# Étiquettes pour les axes x et y des matrices de confusion
labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

# Fonction pour créer un heatmap à partir d'une matrice de confusion
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Création des graphiques de chaleur pour les matrices de confusion
plot_confusion_matrix(conf_matrix_rf, 'Random Forest Confusion Matrix')
plot_confusion_matrix(conf_matrix_svm, 'SVM Confusion Matrix')
