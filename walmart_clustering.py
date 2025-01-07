import kaggle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Télécharge le dataset depuis Kaggle
dataset_name = 'promptcloud/walmart-product-review-dataset'
kaggle.api.dataset_download_files(dataset_name, path='./', unzip=True)

# Charge les données dans un DataFrame
data = pd.read_csv('marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv', sep='\t')
print(data.head())  # Affiche les premières lignes pour vérifier

# Sélectionne un sous-ensemble 500 lignes
data_subset = data.head(500) 

# Sélectionne les colonnes pertinentes pour le clustering
data = data[['Product Price', 'Product Reviews Count', 'Product Available Inventory']]

# Supprime les lignes avec des valeurs manquantes
data = data.dropna()

# Normalise les données
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print(data_scaled[:5])  # Afficher les 5 premières lignes après normalisation
