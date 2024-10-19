import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


file_path = 'data_ecu.txt'
data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Longitude", "Latitude"])

# Data cleaning
data = data.replace({'error': 0, 999.0: 0})

data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
data['Latitude'] = pd.to_numeric(data['Latitude'].str.replace(',', '.'), errors='coerce')

data['Longitude'] = data['Longitude'].fillna(0)
data['Latitude'] = data['Latitude'].fillna(0)

coordinates = data[['Longitude', 'Latitude']].values

# Aplicarcion de  K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster_label'] = kmeans.fit_predict(coordinates)

# Generar archivo con ID de la ubicacion y label del cluster
output_file = 'clusters.txt'
with open(output_file, 'w') as f:
    for loc_id, c_label in enumerate(data['cluster_label']):
        f.write(f"{loc_id} {c_label}\n")
