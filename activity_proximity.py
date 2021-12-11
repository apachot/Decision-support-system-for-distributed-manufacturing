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


""" Function that import all files needed to compute activity_proximity

All files are imported from the data folder inside  the gitlab repo

Import HS2017 nomenclature as nomenclature_HS2017
Import HS 2017 to NACE rev.2 crosswalk as correspondance_HS2017_NACE2
Import NACE rev.2 nomenclature as nomenclature_NACE2
Import HS 1992 to HS 2017 crosswalk as correspondance_HS2017_HS1992
Import productive jumps computed by Harvad Space Productivity lab as productive_jumps

Return all tables in main
"""
def read_table(var_dimension):
    # Import HS2017 nomenclature as HS_2017
    nomenclature_HS2017 = pd.read_csv('./data/in/HS2017_4digits.csv', delimiter=',', header=0)
    # Impoort HS2017 to NACE2 crosswalk as correspondance_HS2017_NACE2
    correspondance_HS2017_NACE2 = pd.read_csv('./data/in/correspondance_NACE_2_HS_2017_4digits_weighted.csv', delimiter=',', header=0)
    # Import NACE 2 nomenclautre as  nomenclature_NACE2
    nomenclature_NACE2 = pd.read_csv('./data/in/NACE_2_with_HS.csv', delimiter=',', header=0)
    # Import HS1992 to HS2017 crosswalk as correspondance_HS2017_HS1992
    correspondance_HS2017_HS1992 = pd.read_csv('./data/in/correspondance_HS2017_HS1992_4digits.csv', delimiter=',', header=0)
    # Import productive jumps from harvard studies as productive_jumps
    productive_jumps = pd.read_csv('./data/in/hs92_proximities.csv', delimiter=',', header=0, dtype={'proximity': float})
    return  nomenclature_HS2017,correspondance_HS2017_NACE2,nomenclature_NACE2,correspondance_HS2017_HS1992,productive_jumps


"""
Function that construct proximities between NACE rev.2

Need :
    - NACE rev.2 nomenclature, called NACE2 inside function
    - HS 2017 to NACE rev. 2 crosswalk called correspondance_HS2017_NACE2 inside function
    - HS2017 nomenclature called HS2017 inside the function
    -  The dissimilarity matrix for HS product computed inside build_space function, as product_vector_space

Function make the crosswalk between the  dissimilarity of product matrix in HS 2017 and the dissimilarity of activity in NACE rev. 2 
And compute activities_proximities with cosine similarity method and add 1 then divide by 2

Return a dataframe of activities_proximities with NACE rev. 2 nomenclature
Return a vector of activity_proximities [ TO IMPLEMENT]
"""
def construct_activities_proximities(NACE2,correspondance_HS2017_NACE2,HS2017,product_vector_space):
    # Merge HS2017 with HS2017 to NACE2 crosswalk
    concatenate_data = HS2017.merge(correspondance_HS2017_NACE2, left_on=HS2017["Id"], right_on= correspondance_HS2017_NACE2["HS4"])
    # Drop unused column
    concatenate_data.drop(columns=["key_0","Label","Id"],inplace=True)
    # Insert HS2017 nomebnclature in product_vector_space
    prod_vec_space =pd.DataFrame(product_vector_space, index=correspondance_HS2017_NACE2["HS4"].sort_values().unique())    # Merge product_vector_space with HS2017 to NACE2 crosswalk
    product_vector_final = concatenate_data.merge(prod_vec_space, left_on=concatenate_data["HS4"], right_on=prod_vec_space.index)
    # Drop redundant columns
    product_vector_final.drop(columns=["key_0"], inplace=True)
    # Compute average proximity between NACE using HS proximities weighted by their contributions in each NACE sectors
    group_weighted_mean = group_weighted_mean_factory(product_vector_final, "weight")
    activities_prox =  product_vector_final.groupby(["NACE2"]).agg(group_weighted_mean)  # Define
    # Compute average activities proximities by cosine similarity method
    activities_proximities = pd.DataFrame(data = (cosine_similarity(activities_prox.iloc[:,2:])+1)/2, index=activities_prox.index.values, columns=activities_prox.index.values )
    return activities_proximities

"""
Function used to do the group_weighted_mean when we group by sector of activity`
the dissimilarity matrix of product

Return a weighted_mean of proximity for each sector

"""
def group_weighted_mean_factory(df: pd.DataFrame, weight_col_name: str):
    # Ref: https://stackoverflow.com/a/69787938/
    def group_weighted_mean(x):
        try:
            return np.average(x, weights=df.loc[x.index, weight_col_name])
        except ZeroDivisionError:
            return np.average(x)
    return group_weighted_mean

"""
Function that construct the dissimilarity matrix

Need : 
    - File of productive jumps computed by Harvard, called prod_jump inside function

Create a dataframe that group by HS 1992 commodity code all the proximity HS 1992 code linked with as columns

Return this dataframe 
"""
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

"""
Function that construct dissimilarity matrix in HS 2017 nomenclature's code

Need :
    - The dissimilarity matrix in HS 1992 code, called dissimilarity inside the function
    - The HS 1992 to HS 2017 crosswalk, called correspondance inside the function

Return the dissimilarity matrix in HS 2017 code as Dataframe
"""
def correspondance_HS17_HS92(dissimilarity,correspondance):
    dict_correspondance =dict(zip(correspondance["HS1992"].values,correspondance["HS2017"].values))
    dissimilarity.rename(index=dict_correspondance,columns=dict_correspondance, inplace=True)
    transition_1 = pd.DataFrame(index= dissimilarity.index, columns=dissimilarity.columns.unique(), data= dissimilarity.groupby(dissimilarity.index, axis=1).mean())
    transition_2 = pd.DataFrame(index= transition_1.index.unique(), columns= transition_1.columns, data =  transition_1.groupby(transition_1.index).mean())
    for index in correspondance["HS2017"]:
        if index not in transition_2.index:
            transition_2.loc[index]=1
            transition_2[index]=1
        transition_2.loc[index,index]=1
    transition_2.sort_index( inplace=True)
    transition_2.sort_index(axis=1, inplace=True)
    return transition_2

"""
Function that construct the cosine similarity between HS 2017 codes

Need :
    - The MDS method from sklearn.manifold to build a multidimensional scale for dissimilarity matrix
    - The cosine_similarity method from sklearn.metrics.pairwise to build cosine similarity between HS codes
Return :
    - the cosine similarity vectors  as product_proximities in form of numpy array
    - the dissimilarity matrix embedded as dissimilarity_matrix_transformed in form of numpy array
"""
def build_space(matrix,var):
    embedding = MDS(n_components=var, verbose=1, dissimilarity='precomputed')
    dissimilarity_matrix_transformed = embedding.fit_transform(1-matrix)
    products_proximities = cosine_similarity(dissimilarity_matrix_transformed)
    return products_proximities, dissimilarity_matrix_transformed


"""
Function that construct the cosine similarity between NACE rev.2 intermediate consumption

Need nothing except the intermediate_consumption file

Return cosine similarities between NACE rev.2 sectors of activity  as intermediate_consumption_proximites in form of pandas DataFrame
"""

def build_space_production():
    intermediate_consumption = pd.read_excel("./data/in/intermediate_consumption.xlsx",index_col=0)

    intermediate_consumption_proximities =cosine_similarity(intermediate_consumption)
    intermediate_consumption_proximities = pd.DataFrame(intermediate_consumption_proximities,index=intermediate_consumption.index, columns=intermediate_consumption.columns )
    """
    wass_distance = intermediate_consumption.to_dict("index")
    for index in intermediate_consumption.index:
        for col in intermediate_consumption.columns:
            wass_distance[index][col]= wasserstein_distance(intermediate_consumption.loc[index],intermediate_consumption.loc[col])
    """
    return intermediate_consumption_proximities


"""
Function that construct  gephi file for pandas Dataframe which have as index and columns names the NACE rev.2 nomenclature

Need the data file to transform, called data inside 

Return the data file in gephi format as gephi_data_final

"""
def build_gephi_file(data):
    gephi_data= data.stack()
    gephi_data= gephi_data.reset_index(level=[0,1])
    gephi_data_final = pd.DataFrame(columns=["source","target","weight"],data= gephi_data.values)
    gephi_data_final = gephi_data_final.loc[gephi_data_final["source"]!=gephi_data_final["target"]]
    gephi_data_final =      gephi_data_final.loc[gephi_data_final["weight"]>0.75]
    return gephi_data_final

"""
Function that build sql format or gephi format when 0 in NACE rev.2 index and columns are missing

Need the data file to transform, called data inside

Call :
    - traitement_source function to rebuild the source column
    - traitement_dest function to rebuild the target column
Return the data file in gephi and sql format as data_2
"""
def build_sql_format(data):
    data.source = data.source.astype(str)
    data.target = data.target.astype(str)
    data_1= traitement_source(data)
    data_2 = traitement_target(data_1)
    data_2.rename(columns={"source":"src","target":"dest"},inplace=True)

    return data_2


"""

Need the data from build_sql_format

Function that add 0 in Nace rev.2 sectors for source column

Return the dataframe with NACE rev.2 sectors completed  as data
"""
def traitement_source(data):
    for value in data.source.unique():
        after = value.split(".")[0]
        if len(after)<2 :
                data.loc[data.source==value,"source"]= "0" +value
                after = value.split(".")[1]
                if len(after)<2:
                    data.loc[data.source=="0" +value,"source"]= "0"+ value + "0"
        else:
                after = value.split(".")[1]
                if len(after)<2:
                    data.loc[data.source==value,"source"]= value + "0"

    return data

"""

Need the data from build_sql_format, called data inside function

Function that add 0 in Nace rev.2 sectors for target column

Return the dataframe with NACE rev.2 sectors completed for target column as data
"""
def traitement_target(data):
    for value in data.target.unique():
        after = value.split(".")[0]
        if len(after)<2 :
            data.loc[data.target==value,"target"]= "0" +value
            after = value.split(".")[1]
            if len(after)<2:
                data.loc[data.target=="0" + value,"target"]="0"+ value + "0"
        else:
            after = value.split(".")[1]
            if len(after)<2:
                data.loc[data.target==value,"target"]= value + "0"

    return data


"""
Main function that call all other.

Could be replace by a class that englobe all function inside this file latter
"""
def main():
    var_dimension = 100
    # call load dataframe function
    nomenclature_HS2017,correspondance_HS2017_NACE2,nomenclature_NACE2, correspondance_HS2017_HS1992,productive_jumps = read_table(var_dimension)
    # Call the construction of dissimilarity matrix in HS 1992 product nomenclature
    dissimilarity_matrix_HS92   =  build_dissimilarity(productive_jumps)
    # Call function that build the dissimilarity matrix in HS2017 nomenclature of product
    dissimilarity_matrix_HS2017 = correspondance_HS17_HS92(dissimilarity_matrix_HS92, correspondance_HS2017_HS1992)
    # Build prooduct proximities in HS 2017 nomenclature code with the cosine similarity method
    products_proximities,dissimilarity_matrix_transformed =build_space(dissimilarity_matrix_HS2017, var_dimension)
    #Construct activities proximities in NACE rev.2 nomenclature with the product proximities in HS 2017 code
    activities_proximities= construct_activities_proximities(nomenclature_NACE2,correspondance_HS2017_NACE2,nomenclature_HS2017,dissimilarity_matrix_transformed)
    # Build gephi file for activity_proximity
    activity_proximities_gephi= build_gephi_file(activities_proximities)
    # Build SQL format for activity_proximities (add 0 to NACE rev.2 sectors that missing)
    activity_proximities_final = build_sql_format(activity_proximities_gephi)
    # Build intermediate_consumption proximities with cosine similarities
    intermediate_consumption_proximities =build_space_production()
    # Build intermediate consumption proximities in gephi format
    interm_cons_prox_gephi = build_gephi_file(intermediate_consumption_proximities)
    # Export dissimilarity matrix with HS 2017 nomenclature  in the path
    pd.DataFrame(dissimilarity_matrix_HS2017).to_csv("./data/out/dissimilarity_matrix_HS2017.csv")
    # Export product proximities in HS 2017 nomenclature  in the path
    pd.DataFrame(products_proximities+1).to_csv("./data/out/products_proximities.csv")
    # Export dissimilarity matrix embedded in the path
    pd.DataFrame(dissimilarity_matrix_transformed).to_csv("./data/out/product_vector_space_dim_"+str(var_dimension)+".csv")
    # Export activities proximities in gephi format in the path
    activity_proximities_gephi.to_csv("./data/out/NACE_proximities_gephi.csv")
    # Export activities proximities computed with productive jump in the path
    activity_proximities_final.to_csv("./data/out/NACE_proximities_sql.csv",index=False)
    # Export intermediate consumption proximities in the path
    interm_cons_prox_gephi.to_csv("./data/out/intermediate_cons_proximities_sql.csv",index=False)




# Launch the function main
main()
