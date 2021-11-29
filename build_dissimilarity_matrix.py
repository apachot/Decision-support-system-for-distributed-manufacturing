import numpy as np
import pandas as pd
import csv

nomenclature_HS2017 = pd.read_csv('./data/HS2017_4digits.csv', delimiter=',', header=0).to_numpy()
correspondance_HS2017_HS1992 = pd.read_csv('./data/correspondance_HS2017_HS1992_4digits.csv', delimiter=',', header=0).to_numpy()
productive_jumps = pd.read_csv('./data/hs92_proximities.csv', delimiter=',', header=0, dtype={'proximity': float}).to_numpy()

print (productive_jumps.shape)
print(productive_jumps[:1])

#building a dissimilarity matrix from harvard proximities
dissimilarity_matrix = np.ones((1225,1225))
for i in range(0, len(productive_jumps)):
	HS_code1_1992 = productive_jumps[i,0]
	HS_code2_1992 = productive_jumps[i,1]
	HS_proximity = productive_jumps[i,2]
	
	idx_HS_code1 =  np.where(correspondance_HS2017_HS1992[:,1]==float(HS_code1_1992))
	idx_HS_code2 =  np.where(correspondance_HS2017_HS1992[:,1]==float(HS_code2_1992))
	if (len(idx_HS_code1[0]) > 0) and (len(idx_HS_code2[0]) > 0):
		#print("trouvé idx_HS_code1 et idx_HS_code2")
		HS_code1_2017 = correspondance_HS2017_HS1992[idx_HS_code1[0],0][0]
		HS_code2_2017 = correspondance_HS2017_HS1992[idx_HS_code2[0],0][0]
		
		#looking for the index of each hs 2017 code in the nomenclature
		idx_HS_code1_nomenclature =  np.where(nomenclature_HS2017[:,0]==float(HS_code1_2017))
		idx_HS_code2_nomenclature =  np.where(nomenclature_HS2017[:,0]==float(HS_code2_2017))

		dissimilarity_matrix[idx_HS_code1_nomenclature[0], idx_HS_code1_nomenclature[0]] = 1
		dissimilarity_matrix[idx_HS_code2_nomenclature[0], idx_HS_code2_nomenclature[0]] = 1
		dissimilarity_matrix[idx_HS_code1_nomenclature[0], idx_HS_code2_nomenclature[0]] = HS_proximity
		dissimilarity_matrix[idx_HS_code2_nomenclature[0], idx_HS_code1_nomenclature[0]] = HS_proximity


		print (i/len(productive_jumps), " done (",i , "over" , len(productive_jumps),")")
	else:
		print("not found : HS_code1_1992=", HS_code1_1992, "HS_code2_1992=", HS_code2_1992, "HS_proximity=", HS_proximity)

pd.DataFrame(dissimilarity_matrix).to_csv("./data/dissimilarity_matrix.csv")
