import numpy as np
import pandas as pd
import csv

NACE_2_with_HS = pd.read_csv('./data/NACE_2_with_HS.csv', delimiter=',', header=0).to_numpy()
activities_proximities = pd.read_csv('./data/activities_proximities.csv', delimiter=',', header=0, index_col=0).to_numpy()

NACE_proximities = [['source','target','weight']]
print (activities_proximities.shape)
print(activities_proximities[:3])

print (NACE_2_with_HS.shape)
print(NACE_2_with_HS[:3])
for i in range(0, len(NACE_2_with_HS)):
	NACE_code = NACE_2_with_HS[i,0]
	print ("NACE_code=", NACE_code)
	for j in range(i, len(activities_proximities[i])):
		if (NACE_2_with_HS[i,0] != NACE_2_with_HS[j]):
			print(NACE_code, "-", NACE_2_with_HS[j], "->", activities_proximities[i,j])
			if (activities_proximities[i,j] >.8):
				NACE_proximities.append([NACE_code,NACE_2_with_HS[j,0],activities_proximities[i,j]])


pd.DataFrame(NACE_proximities).to_csv("./data/NACE_proximities.csv", header=False, index=False)