#!/usr/bin/env python
# coding: utf-8

# In[9]:


'''
Author: Aditya Jain
Date  : 11th June, 2021
About : This script counts the no. of training points for each taxa
'''
import pandas as pd
import json

TRAIN_SET   = '/home/mila/a/aditya.jain/mothAI/deeplearning/data/01-train_split.csv'
WRITE_DIR   = "/home/mila/a/aditya.jain/mothAI/deeplearning/data/"


# In[8]:


train_data             = pd.read_csv(TRAIN_SET)
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


# In[10]:


with open(WRITE_DIR + 'count_train-points.json', 'w') as outfile:
    json.dump(final_count, outfile)

