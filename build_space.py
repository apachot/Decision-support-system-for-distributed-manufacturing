import numpy as np
import pandas as pd
import csv
from sklearn.manifold import MDS

dissimilarity_matrix = pd.read_csv('./data/dissimilarity_matrix.csv', delimiter=',', header=0, index_col=0).to_numpy()


print (dissimilarity_matrix.shape)
print(dissimilarity_matrix[:1])

var_dimension = 100
embedding = MDS(n_components=var_dimension, verbose=1, dissimilarity='precomputed')
dissimilarity_matrix_transformed = embedding.fit_transform(dissimilarity_matrix)
print(dissimilarity_matrix_transformed.shape)
print(dissimilarity_matrix_transformed[:1])

pd.DataFrame(dissimilarity_matrix_transformed).to_csv("./data/product_vector_space_dim_"+str(var_dimension)+".csv")
