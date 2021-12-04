import numpy as np
import pandas as pd
import csv

officies_2018 = pd.read_csv('./data/french_officies_2018.csv', delimiter=',', header=0).to_numpy()
IO_matrix = pd.read_csv('./data/IO-matrix', delimiter=',', header=0).to_numpy()
nomenclature_NACE2 = pd.read_csv('./data/NACE_2_with_HS.csv', delimiter=',', header=0).to_numpy()

print (officies_2018.shape)
print(officies_2018[:3])

print (IO_matrix.shape)
print(IO_matrix[:3])


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
