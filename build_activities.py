import numpy as np
import pandas as pd
import csv

var_dimension = 100

nomenclature_HS2017 = pd.read_csv('./data/HS2017_4digits.csv', delimiter=',', header=1).to_numpy()
correspondance_HS2017_HS1992 = pd.read_csv('./data/correspondance_HS2017_HS1992_4digits.csv', delimiter=',', header=1).to_numpy()
correspondance_HS2017_NACE2 = pd.read_csv('./data/correspondance_NACE_2_HS_2017_4digits_weighted.csv', delimiter=',', header=1).to_numpy()
nomenclature_NACE2 = pd.read_csv('./data/NACE_2.csv', delimiter=',', header=None).to_numpy()
productive_jumps = pd.read_csv('./data/hs92_proximities.csv', delimiter=',', header=1, dtype={'proximity': float}).to_numpy()
proximity_matrix = pd.read_csv('./data/proximity_matrix.csv', delimiter=',', header=1, dtype={'proximity': float}).to_numpy()
dissimilarity_matrix = pd.read_csv('./data/dissimilarity_matrix.csv', delimiter=',', header=0).to_numpy()
product_vector_space_dim_100 = pd.read_csv('./data/product_vector_space_dim_'+str(var_dimension)+'.csv', delimiter=',', header=0).to_numpy()

print (product_vector_space_dim_100.shape)
print(product_vector_space_dim_100[:3])

print (dissimilarity_matrix.shape)
print(dissimilarity_matrix[:3])
