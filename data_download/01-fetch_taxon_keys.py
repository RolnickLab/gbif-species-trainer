#!/usr/bin/env python
# coding: utf-8

"""
Author       : Aditya Jain
Date Started : May 9, 2022
About        : This script fetches unique taxon keys for species list from GBIF database
"""

from pygbif import occurrences as occ
from pygbif import species as species_api
import pandas as pd
import os
import tqdm
import urllib
import json
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', help = 'absolute path of the species list')
parser.add_argument('--column_name', help = 'column name of the species entries')
parser.add_argument('--output_name', help = 'output name of the file')
args   = parser.parse_args()

file_path   = args.file_path
column_name = args.column_name

def get_gbif_key_backbone(name, place):
    """ given a species name, this function returns the unique gbif key and other 
        attributes using backbone API
    """
    
    # default values
    taxon_key      = [-1]
    order          = ['NA']
    family         = ['NA']
    genus          = ['NA']    
    search_species = [name]
    gbif_species   = ['NA']     # the name returned on search, can be different from the search
    confidence     = ['']
    status         = ['NA']
    match_type     = ['NONE']
    place          = [place]

    data = species_api.name_backbone(name=name, strict=True, rank='species')

    if data['matchType'] == 'NONE':
        confidence    = [data['confidence']]
    else:
        taxon_key     = [data['usageKey']]
        order         = [data['order']]
        family        = [data['family']]
        genus         = [data['genus']]
        confidence    = [data['confidence']]
        gbif_species  = [data['species']]
        status        = [data['status']]
        match_type    = [data['matchType']]
  
    df = pd.DataFrame(list(zip(taxon_key, order, family, genus,
                               search_species, gbif_species, confidence,
                               status, match_type, place)),
                    columns =['taxon_key_guid', 'order_name', 'family_name',
                              'genus_name', 'search_species_name', 'gbif_species_name',
                              'confidence', 'status', 'match_type', 'source'])
    return df


# #### Get the list of macro-moth species from the list
data          = pd.read_csv(file_path, index_col=False)
species_list  = []

for indx in data.index:
    species_list.append(data[column_name][indx])



data_final = pd.DataFrame(columns =['taxon_key_guid', 'order_name', 'family_name',
                              'genus_name', 'search_species_name', 'gbif_species_name',
                              'confidence', 'status', 'match_type', 'source'], dtype=object)
for name in species_list:
    data       = get_gbif_key_backbone(name, 'uksi_09May2022')
    data_final = data_final.append(data, ignore_index = True)
    
data_final.to_csv(data_dir + 'UK-MacroMoth-List_09May2022.csv', index=False)  







