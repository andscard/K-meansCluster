import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


file_path = 'data_ecu.txt'
data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Longitude", "Latitude"])

# Data cleaning
data = data.replace('error', np.nan)
data = data.dropna()
data = data.replace('999.0', np.nan)

data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
data['Latitude'] = pd.to_numeric(data['Latitude'].str.replace(',', '.'), errors='coerce')

data = data.fillna(0)
coordinates = data[['Longitude', 'Latitude']].values

# Aplicarcion de  K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster_label'] = kmeans.fit_predict(coordinates)

# Generar archivo con ID de la ubicacion y label del cluster
output_file = 'clusters.txt'
with open(output_file, 'w') as f:
    for loc_id, c_label in enumerate(data['cluster_label']):
        f.write(f"{loc_id} {c_label}\n")
