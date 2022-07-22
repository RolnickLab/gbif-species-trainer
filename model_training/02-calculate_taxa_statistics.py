#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date Started  : July 20, 2022
About         : Calculates information and statistics regarding the taxonomy and data
"""
import pandas as pd
import json
import argparse

def convert_to_numeric_labels(args):
	"""converts string labels (species, genus, family) to numeric labels"""
	
	data_list   = args.species_list
	write_dir   = args.write_dir	
	data        = pd.read_csv(data_list, keep_default_na=False)

	species_list = list(set(data['gbif_species_name']))
	genus_list   = list(set(data['genus_name']))
	family_list  = list(set(data['family_name']))

	try:
		species_list.remove('NA')
		genus_list.remove('NA')
		family_list.remove('NA')
	except:
		pass

	print(f'Total families: {len(family_list)}, genuses: {len(genus_list)}, species: {len(species_list)}')
	
	list_data = {}
	list_data['family_list']  = family_list
	list_data['genus_list']   = genus_list
	list_data['species_list'] = species_list
	list_data['Note']         = 'The integer index in their respective list will be the numeric class label'

	with open(write_dir + args.numeric_labels_filename + '.json', 'w') as outfile:
		json.dump(list_data, outfile)


def taxon_hierarchy(args):
	"""saves the taxon hierarchy for each species"""
	
	data_list   = args.species_list
	write_dir   = args.write_dir
	data        = pd.read_csv(data_list, keep_default_na=False)
	
	taxon_hierar         = {}
	taxon_hierar['Note'] = 'The 0th index is genus and 1st index is family'

	for indx in data.index:
		if data['gbif_species_name'][indx] not in taxon_hierar.keys() and data['gbif_species_name'][indx]!='NA':
			taxon_hierar[data['gbif_species_name'][indx]] = [data['genus_name'][indx], data['family_name'][indx]]
        
	with open(write_dir + args.taxon_hierarchy_filename + '.json', 'w') as outfile:
		json.dump(taxon_hierar, outfile)


def count_training_points(args):
	"""counts the number of training points for each taxa"""
	
	train_data             = pd.read_csv(args.train_split_file)
	final_count            = {}
	final_count['family']  = {}
	final_count['genus']   = {}
	final_count['species'] = {}
	
	for indx in train_data.index:
		if train_data['family'][indx] not in final_count['family'].keys():
			final_count['family'][train_data['family'][indx]] = 1
		else:
			final_count['family'][train_data['family'][indx]] += 1
        
		if train_data['genus'][indx] not in final_count['genus'].keys():
			final_count['genus'][train_data['genus'][indx]] = 1
		else:
			final_count['genus'][train_data['genus'][indx]] += 1
        
		if train_data['species'][indx] not in final_count['species'].keys():
			final_count['species'][train_data['species'][indx]] = 1
		else:
			final_count['species'][train_data['species'][indx]] += 1 
			
	with open(args.write_dir + args.training_points_filename + '.json', 'w') as outfile:
		json.dump(final_count, outfile)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--species_list', help = 'path to the species list', required=True)
	parser.add_argument('--write_dir', help = 'path to the directory for saving the information', required=True)
	parser.add_argument('--numeric_labels_filename', help = 'filename for numeric labels file', required=True)
	parser.add_argument('--taxon_hierarchy_filename', help = 'filename for taxon hierarchy file', required=True)
	parser.add_argument('--training_points_filename', help = 'filename for storing the count of training points', required=True)
	parser.add_argument('--train_split_file', help = 'path to the training split file', required=True)
	args   = parser.parse_args()

	convert_to_numeric_labels(args)
	taxon_hierarchy(args)
	count_training_points(args)


