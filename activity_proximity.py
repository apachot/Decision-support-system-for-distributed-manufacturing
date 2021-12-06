#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:12:49 2021

@author: nsauzeat
"""
import numpy as np
import pandas as pd
import csv
from sklearn.metrics.pairwise import cosine_similarity
import pdb
import os 
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance

def read_table(var_dimension):
    # Import HS2017 nomenclature as HS_2017
    nomenclature_HS2017 = pd.read_csv('./data/HS2017_4digits.csv', delimiter=',', header=0)
    # Impoort HS2017 to NACE2 crosswalk as correspondance_HS2017_NACE2
    correspondance_HS2017_NACE2 = pd.read_csv('./data/correspondance_NACE_2_HS_2017_4digits_weighted.csv', delimiter=',', header=0)
    # Import NACE 2 nomenclautre as  nomenclature_NACE2
    nomenclature_NACE2 = pd.read_csv('./data/NACE_2_with_HS.csv', delimiter=',', header=0)
    # Import product_vector_space
    product_vector_space = pd.read_csv('./data/product_vector_space_dim_'+str(var_dimension)+'.csv', delimiter=',', header=0, index_col=0)
    # Import HS1992 to HS2017 crosswalk as correspondance_HS2017_HS1992
    correspondance_HS2017_HS1992 = pd.read_csv('./data/correspondance_HS2017_HS1992_4digits.csv', delimiter=',', header=0)
    # Import productive jumps from harvard studies as productive_jumps
    productive_jumps = pd.read_csv('./data/hs92_proximities.csv', delimiter=',', header=0, dtype={'proximity': float})
    return  nomenclature_HS2017,correspondance_HS2017_NACE2,nomenclature_NACE2, product_vector_space,correspondance_HS2017_HS1992,productive_jumps

def construct_activities_proximities(NACE2,correspondance_HS2017_NACE2,HS2017,product_vector_space):
    # Merge HS2017 with HS2017 to NACE2 crosswalk
    concatenate_data = HS2017.merge(correspondance_HS2017_NACE2, left_on=HS2017["Id"], right_on= correspondance_HS2017_NACE2["HS4"])
    # Drop unused column
    concatenate_data.drop(columns=["key_0","Label","Id"],inplace=True)
    # Insert HS2017 nomebnclature in product_vector_space
    product_vector_space.insert(column="HS", value=HS2017["Id"],loc=0)
    # Merfe product_vector_space with HS2017 to NACE2 crosswalk
    product_vector_final = concatenate_data.merge(product_vector_space, left_on=concatenate_data["HS4"], right_on=product_vector_space["HS"])
    # Drop redundant columns
    product_vector_final.drop(columns=["key_0","HS"], inplace=True)
    # Compute average proximity between NACE using HS proximities weighted by their contributions in each NACE sectors
    group_weighted_mean = group_weighted_mean_factory(product_vector_final, "weight")
    activities_prox =  product_vector_final.groupby(["NACE2"]).agg(group_weighted_mean)  # Define
    # Compute average activities proximities by cosine similarity method
    activities_proximities = pd.DataFrame(data = (cosine_similarity(activities_prox.iloc[:,2:])+1)/2, index=activities_prox.index )
    return activities_proximities

def group_weighted_mean_factory(df: pd.DataFrame, weight_col_name: str):
    # Ref: https://stackoverflow.com/a/69787938/
    def group_weighted_mean(x):
        try:
            return np.average(x, weights=df.loc[x.index, weight_col_name])
        except ZeroDivisionError:
            return np.average(x)
    return group_weighted_mean

def build_dissimilarity(prod_jump):
    # Create a dataframe to register commodities proximities
    matrix= pd.DataFrame(index= prod_jump.commoditycode_1.unique(), columns =prod_jump.commoditycode_1.unique())
    # fill dataframe with 0.0
    matrix.fillna(0.0,inplace=True)
    # Build a group by object to reorganize dataframe of prod_jumps
    grouped_hs = prod_jump.groupby("commoditycode_1")
    # Fill dataframe with productive_jumps
    for i in matrix.index:
        matrix.loc[i]= grouped_hs.get_group(i)["proximity"].values
    return matrix


def build_space(matrix,var):
    #embedding = MDS(n_components=var, verbose=1, dissimilarity='precomputed')
    #dissimilarity_matrix_transformed = embedding.fit_transform(1-matrix)
    #products_proximities = cosine_similarity(dissimilarity_matrix_transformed)
    stress=[]
    max_range = 150
    for dim in range(118, max_range):
        # Set up the MDS object
        mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=0)
        # Apply MDS
        pts = mds.fit_transform(1-matrix)
        # Retrieve the stress value
        stress.append(mds.stress_)
        # Plot stress vs. n_components
    plt.plot(range(118, max_range), stress)
    plt.xticks(range(118, max_range, 2))
    plt.xlabel('n_components')
    plt.ylabel('stress')
    plt.show()
    pdb.set_trace()

    #return products_proximities, dissimilarity_matrix_transformed

def build_space_production(var):
    intermediate_consumption = pd.read_excel("./data/intermediate_consumption.xlsx",index_col=0)
    #embedding = MDS(n_components=var, verbose=1, dissimilarity='precomputed')
    #intermediate_consumption_transformed = embedding.fit_transform(intermediate_consumption.to_numpy())
    #intermediate_consumption_proximities =cosine_similarity(intermediate_consumption)
    wass_distance = intermediate_consumption.to_dict("index")
    for index in intermediate_consumption.index:
        for col in intermediate_consumption.columns:
            wass_distance[index][col]= wasserstein_distance(intermediate_consumption.loc[index],intermediate_consumption.loc[col])

    return intermediate_consumption_proximities

def main():
    var_dimension = 100
    # call load dataframe function
    nomenclature_HS2017,correspondance_HS2017_NACE2,nomenclature_NACE2, product_vector_space,correspondance_HS2017_HS1992,productive_jumps = read_table(var_dimension)
    # Call the construction of activities proximities function
    activities_proximities= construct_activities_proximities(nomenclature_NACE2,correspondance_HS2017_NACE2,nomenclature_HS2017, product_vector_space)
    # Call the construction of dissimilarity matrix
    dissimilarity_matrix   =  build_dissimilarity(productive_jumps)
    # Call function that build the space production of product similarities
    products_proximities,dissimilarity_matrix_transformed =build_space(dissimilarity_matrix, var_dimension)
    # Extract the dataframe of products_proximities as csv file
    pd.DataFrame(products_proximities+1).to_csv("./data/products_proximities.csv")
    # Extract the dataframe of dissimilarity matrix as csv file
    pd.DataFrame(dissimilarity_matrix_transformed).to_csv("./data/product_vector_space_dim_"+str(var_dimension)+".csv")
    # Build the production similarities of intermediate_consumption
    intermediate_consumption_proximities =build_space_production(var_dimension)


main()


