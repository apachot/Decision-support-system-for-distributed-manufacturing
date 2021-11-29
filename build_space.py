import numpy as np
import pandas as pd
import csv
from sklearn.manifold import MDS

nomenclature_HS2017 = pd.read_csv('./data/HS2017_4digits.csv', delimiter=',', header=1).to_numpy()
correspondance_HS2017_HS1992 = pd.read_csv('./data/correspondance_HS2017_HS1992_4digits.csv', delimiter=',', header=1).to_numpy()
correspondance_HS2017_NACE2 = pd.read_csv('./data/correspondance_NACE_2_HS_2017_4digits_weighted.csv', delimiter=',', header=1).to_numpy()
nomenclature_NACE2 = pd.read_csv('./data/NACE_2.csv', delimiter=',', header=None).to_numpy()
productive_jumps = pd.read_csv('./data/hs92_proximities.csv', delimiter=',', header=1, dtype={'proximity': float}).to_numpy()
proximity_matrix = pd.read_csv('./data/proximity_matrix.csv', delimiter=',', header=1, dtype={'proximity': float}).to_numpy()
dissimilarity_matrix = pd.read_csv('./data/dissimilarity_matrix.csv', delimiter=',', header=1, dtype={'proximity': float}).to_numpy()


print (dissimilarity_matrix.shape)
print(dissimilarity_matrix[:1])

var_dimension = 100
embedding = MDS(n_components=var_dimension, verbose=1)
X_transformed = embedding.fit_transform(dissimilarity_matrix)
print(X_transformed.shape)
print(X_transformed[:1])

pd.DataFrame(X_transformed).to_csv("./data/product_vector_space_dim_"+str(var_dimension)+".csv")
