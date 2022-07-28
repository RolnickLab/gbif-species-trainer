#!/usr/bin/env python
# coding: utf-8

"""
Author        : Aditya Jain
Date Started  : July 20, 2022
About         : Division of dataset into train, validation and test sets
"""
import os
import glob
import random
import pandas as pd
import argparse

def prepare_split_list(global_pd, new_list, fields):
    """
    prepares a global csv list for every type of data split
    
    Args:
        global_pd: a global list into which new entries will be appended
        new_list : list of new entries to be appended to global list   
        fields   : contains the column names
    """
    new_data = []
    
    for path in new_list:
        path_split = path.split('/')
        
        filename   = path_split[-1]
        species    = path_split[-2]
        genus      = path_split[-3]
        family     = path_split[-4]
        
        new_data.append([filename, family, genus, species])
        
    new_data  = pd.DataFrame(new_data, columns=fields, dtype=object)    
    global_pd = global_pd.append(new_data, ignore_index=True)
    
    return global_pd        


def create_data_split(args):
	"""main function for creating the dataset split"""

	data_dir    = args.data_dir       # root directory of data
	write_dir   = args.write_dir      # split files to be written
	train_spt   = args.train_ratio    # train set ratio
	val_spt     = args.val_ratio      # validation set ration
	test_spt    = args.test_ratio     # test set ratio
	assert train_spt+val_spt+test_spt==1, 'Train, val and test ratios should exactly sum to 1'
	
	fields     = ['filename', 'family', 'genus', 'species']
	train_data = pd.DataFrame(columns = fields, dtype=object)
	val_data   = pd.DataFrame(columns = fields, dtype=object)
	test_data  = pd.DataFrame(columns = fields, dtype=object)

	for family in os.listdir(data_dir):
		if os.path.isdir(data_dir + '/' + family):
        
			for genus in os.listdir(data_dir + family):
				if os.path.isdir(data_dir + '/' + family + '/' + genus):
                
					for species in os.listdir(data_dir + family + '/' + genus):
						if os.path.isdir(data_dir + '/' + family + '/' + genus + '/' + species):
            
							file_data  = glob.glob(data_dir + family + '/' + genus + '/' + species + '/*.jpg')
							random.shuffle(file_data)
            
							total      = len(file_data)
							train_amt  = round(total*train_spt)
							val_amt    = round(total*val_spt)            
             
							train_list = file_data[:train_amt]
							val_list   = file_data[train_amt:train_amt+val_amt]
							test_list  = file_data[train_amt+val_amt:]
            
							train_data = prepare_split_list(train_data, train_list, fields)
							val_data   = prepare_split_list(val_data, val_list, fields)
							test_data  = prepare_split_list(test_data, test_list, fields)            


	# saving the lists to disk
	train_data.to_csv(write_dir + args.filename + '-train-split.csv', index=False)
	val_data.to_csv(write_dir + args.filename + '-val-split.csv', index=False)
	test_data.to_csv(write_dir + args.filename + '-test-split.csv', index=False)
	
	# printing stats
	print('Training data size: ', len(train_data))
	print('Validation data size: ', len(val_data))
	print('Testing data size: ', len(test_data))
	print('Total images: ', len(train_data)+len(val_data)+len(test_data))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', help = 'path to the root directory containing the data', required=True)
	parser.add_argument('--write_dir', help = 'path to the directory for saving the split files', required=True)
	parser.add_argument('--train_ratio', help = 'proportion of data for training', required=True, type=float)
	parser.add_argument('--val_ratio', help = 'proportion of data for validation', required=True, type=float)
	parser.add_argument('--test_ratio', help = 'proportion of data for testing', required=True, type=float)
	parser.add_argument('--filename', help = 'initial name for the split files', required=True)
	args   = parser.parse_args()

	create_data_split(args)




