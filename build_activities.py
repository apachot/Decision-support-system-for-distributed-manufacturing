import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity


var_dimension = 100

nomenclature_HS2017 = pd.read_csv('./data/HS2017_4digits.csv', delimiter=',', header=0).to_numpy()
correspondance_HS2017_NACE2 = pd.read_csv('./data/correspondance_NACE_2_HS_2017_4digits_weighted.csv', delimiter=',', header=0).to_numpy()
#nomenclature_NACE2 = pd.read_csv('./data/NACE_2.csv', delimiter=',', header=None).to_numpy()
nomenclature_NACE2 = pd.read_csv('./data/NACE_2_with_HS.csv', delimiter=',', header=0).to_numpy()
product_vector_space = pd.read_csv('./data/product_vector_space_dim_'+str(var_dimension)+'.csv', delimiter=',', header=0, index_col=0).to_numpy()

print (product_vector_space.shape)
print(product_vector_space[:3])


print (nomenclature_HS2017.shape)
print(nomenclature_HS2017[:3])

tab_activities_vectors = np.zeros((len(nomenclature_NACE2), var_dimension))
#tab_real_NACE = []

for i in range(0, len(nomenclature_NACE2)):
	NACE_code = nomenclature_NACE2[i,0]
	print('code = ', NACE_code)
	idx_NACE_code =  np.where(correspondance_HS2017_NACE2[:,0]==float(NACE_code))
	nb_NACE_code = len(idx_NACE_code[0])
	if (nb_NACE_code > 0):
		tab_HS_vectors = np.zeros((nb_NACE_code, var_dimension))
		tab_HS_weights = np.zeros(nb_NACE_code)
		print("trouvé, len=",nb_NACE_code)
		for j in range(0, nb_NACE_code):
			HS_code2017 = correspondance_HS2017_NACE2[idx_NACE_code[0],1][j]
			HS_weight = correspondance_HS2017_NACE2[idx_NACE_code[0],2][j]
			tab_HS_weights[j] = HS_weight
			print("HS_code2017=",HS_code2017,"HS_weight=",HS_weight)
			#looking for the index of the HS code in the nomenclature
			idx_HS_code2017 =  np.where(nomenclature_HS2017[:,0]==float(HS_code2017))
			if (len(idx_HS_code2017[0]) > 0):
				#we load the product vector
				HS_vector = product_vector_space[idx_HS_code2017[0],:][0]
				print("HS-vector=",HS_vector)
				print(HS_vector.shape)
				print(tab_HS_vectors.shape)
				tab_HS_vectors[j] = HS_vector
			else:
				print('error, HS2017 code not found in the nomenclature')

		print("tab_HS_vectors=", tab_HS_vectors)
		#we do a weighted average of vectors
		average_HS_vector = np.average(tab_HS_vectors, axis=0, weights=tab_HS_weights)
		print("average_HS_vector=", average_HS_vector)
		tab_activities_vectors[i] = average_HS_vector


	else:
		print("not found")

print ("tab_activities_vectors=", tab_activities_vectors)
print(tab_activities_vectors.shape)

pd.DataFrame(tab_activities_vectors).to_csv("./data/tab_activities_vectors.csv")

#looking for similarities between activities
activities_proximities = cosine_similarity(tab_activities_vectors)
print (activities_proximities.shape)
print(activities_proximities[:3])
pd.DataFrame((activities_proximities+1)/2).to_csv("./data/activities_proximities.csv")
