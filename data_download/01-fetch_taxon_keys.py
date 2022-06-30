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
parser.add_argument('--file_path', help = 'path of the species list', required=True)
parser.add_argument('--column_name', help = 'column name of the species entries', required=True)
parser.add_argument('--output_filename', help = 'output name of the file', required=True)
args   = parser.parse_args()

file_path    = args.file_path
column_name  = args.column_name
out_filename = args.output_filename

def get_gbif_key_backbone(name):
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
								status, match_type)),
					columns =['taxon_key_gbif_id', 'order_name', 'family_name',
                              'genus_name', 'search_species_name', 'gbif_species_name',
                              'confidence', 'status', 'match_type'])
	return df

# fetch species names from the list
data          = pd.read_csv(file_path, index_col=False)
species_list  = []
for indx in data.index:
	species_list.append(data[column_name][indx])

data_final = pd.DataFrame(columns =['taxon_key_gbif_id', 'order_name', 'family_name',
                              'genus_name', 'search_species_name', 'gbif_species_name',
                              'confidence', 'status', 'match_type'], dtype=object)

# fetch taxonomy data from GBIF
for name in species_list:
	data       = get_gbif_key_backbone(name)
	data_final = data_final.append(data, ignore_index = True)
    

# save the file
data_final.to_csv(out_filename + '.csv', index=False)  







