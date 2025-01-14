import kaggle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Étape 1 : Télécharge le dataset depuis Kaggle
dataset_name = 'promptcloud/walmart-product-review-dataset'
kaggle.api.dataset_download_files(dataset_name, path='./', unzip=True)

# Étape 2 : Charge les données dans un DataFrame
data = pd.read_csv('marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv', sep='\t')
print("Aperçu des données :\n", data.head())  # Affiche les premières lignes pour vérifier

# Étape 3 : Sélectionne un sous-ensemble de 500 lignes
data_subset = data.head(500)

# Étape 4 : Sélectionne les colonnes pertinentes pour le clustering
data = data[['Product Price', 'Product Reviews Count', 'Product Available Inventory']]

# Étape 5 : Supprime les lignes avec des valeurs manquantes
data = data.dropna()

# Étape 6 : Normalise les données
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print("Données normalisées (5 premières lignes) :\n", data_scaled[:5])

# Étape 7 : Applique k-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Ajoute les clusters au DataFrame original
data['Cluster'] = clusters

# Étape 8 : Visualisation des Clusters
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', s=50)
plt.xlabel('Product Price (normalisé)')
plt.ylabel('Product Reviews Count (normalisé)')
plt.title('Visualisation des Clusters avec k-Means')
plt.colorbar(label='Cluster')

# Affiche la fenêtre sans bloquer l'exécution du code
plt.show()

# Étape 9 : Résumé des Clusters
cluster_summary = data.groupby('Cluster').mean()
cluster_summary['Nombre de produits'] = data['Cluster'].value_counts()
print("\nRésumé des Clusters :\n", cluster_summary)

# Exporte le résumé des clusters en CSV
cluster_summary.to_csv('cluster_summary.csv', index=False)
print("\nRésumé des clusters exporté dans le fichier 'cluster_summary.csv'")