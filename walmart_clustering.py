import kaggle
import pandas as pd

# Télécharger le dataset depuis Kaggle
dataset_name = 'promptcloud/walmart-product-review-dataset'
kaggle.api.dataset_download_files(dataset_name, path='./', unzip=True)

# Charger les données dans un DataFrame
data = pd.read_csv('marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv', sep='\t')
print(data.head())  # Affiche les premières lignes pour vérifier
