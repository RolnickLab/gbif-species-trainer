#!/usr/bin/env python
# coding: utf-8

"""
Author       : Aditya Jain
Date Started : May 10, 2022
About        : This script scraps data from GBIF for a list of moth species
"""

import pygbif
from pygbif import occurrences as occ
from pygbif import species as species_api
import pandas as pd
import os
import tqdm
import urllib
import json
import time
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--write_directory', help = 'path of the folder to save the data', required=True)
parser.add_argument('--species_key_filepath', help = 'path of csv file containing list of species names along with unique GBIF taxon keys', required=True)
parser.add_argument('--max_images_per_species', help = 'maximum number of images to download for any speices', default=500, type=int)
parser.add_argument('--resume_session', help = 'False/True; whether resuming a previously stopped downloading session', required=True, type=bool)
args   = parser.parse_args()

write_dir     = args.write_directory
species_list  = args.species_key_filepath
MAX_DATA_SP   = args.max_images_per_species    

LIMIT_DOWN     = 300                                      # GBIF API parameter for max results per page 
MAX_SEARCHES   = 11000                                    # maximum no. of points to iterate
moth_data      = pd.read_csv(species_list)


# Downloading gbif data

def inat_metadata_gbif(data):
    """ returns the relevant gbif metadata for a GBIF observation"""

    fields    = ['decimalLatitude', 'decimalLongitude',
            'order', 'family', 'genus', 'species', 'acceptedScientificName',
            'year', 'month', 'day',
            'datasetName', 'taxonID', 'acceptedTaxonKey', 'lifeStage', 'datasetName']

    meta_data = {}

    for field in fields:
        if field in data.keys():
            meta_data[field] = data[field]
        else:
            meta_data[field] = ''

    return meta_data


dataset_list       = []
taxon_key          = list(moth_data['taxon_key_gbif_id'])         # list of taxon keys
species_name       = list(moth_data['search_species_name'])       # list of species name that is searched
gbif_species_name  = list(moth_data['gbif_species_name'])         # list of species name returned by gbif [can be different from above or NA]
columns            = ['taxon_key_gbif_id', 'search_species_name', 'gbif_species_name', 'count']
count_list         = pd.DataFrame(columns = columns, dtype=object)            

# if resuming the download session
if args.resume_session is True:
    count_list         = pd.read_csv(write_dir + 'datacount.csv')         

for i in range(len(taxon_key)):
    if int(taxon_key[i]) in count_list['taxon_key_gbif_id'].tolist():
        print(f'Already downloaded for {species_name[i]}')
        continue
    print('Downloading for: ', species_name[i])
    begin   = time.time()
    if taxon_key[i] == -1:          # taxa not there on GBIF
        count_list = count_list.append(pd.DataFrame([['NA', species_name[i], 'NA', 'NA']],
                                                columns=columns), ignore_index=True)
    else: 
        data        = occ.search(taxonKey = int(taxon_key[i]), mediatype='StillImage', limit=1)
        total_count = data['count']   

        if total_count==0:            # no image data found for the species 
            count_list = count_list.append(pd.DataFrame([[int(taxon_key[i]), species_name[i], gbif_species_name[i], 0]],
                                                columns=columns), ignore_index=True)    
        else:
            image_count = 0                                   # images downloaded for every species
            max_count   = min(total_count, MAX_DATA_SP)
            total_pag   = math.ceil(MAX_SEARCHES/LIMIT_DOWN)  # total pages to be fetched with max 300 entries each
            offset      = 0

            family  = data['results'][0]['family']
            genus   = data['results'][0]['genus']  
            species = data['results'][0]['species']

            m_data  = {}                                 # dictionary variable to store metadata
            write_loc = write_dir + family + "/" + genus + "/" + species 

            try:    
                os.makedirs(write_loc)                     # creating hierarchical structure for image storage 
            except:
                pass

            for j in range(total_pag):
                try:
                    data       = occ.search(taxonKey = int(taxon_key[i]), mediatype='StillImage', 
                               limit=LIMIT_DOWN, offset=offset)        
                except:
                    tot_points = 0
                else:
                    tot_points = len(data['results'])

                for k in range(tot_points):                     
                    if data['results'][k]['media'] and 'lifeStage' in data['results'][k].keys():
                        if data['results'][k]['lifeStage']=='Adult' or data['results'][k]['lifeStage']=='Imago':          
                            gbifid   = data['results'][k]['gbifID']
                            
                            if 'identifier' in data['results'][k]['media'][0].keys():
                                image_url   = data['results'][k]['media'][0]['identifier']            
                                try:
                                    urllib.request.urlretrieve(image_url, write_loc + '/' + gbifid + '.jpg')
                                    image_count += 1              
                                    meta_data      = inat_metadata_gbif(data['results'][k])   # fetching metadata
                                    m_data[gbifid] = meta_data                                
                                    if meta_data['datasetName'] not in dataset_list:
                                        dataset_list.append(meta_data['datasetName'])
                                except:
                                    continue  
                    if image_count >= MAX_DATA_SP:
                        break
                
                offset += LIMIT_DOWN
                if image_count >= MAX_DATA_SP:
                    break

            with open(write_loc + '/' + 'metadata.txt', 'w') as outfile:
                json.dump(m_data, outfile)       

            count_list = count_list.append(pd.DataFrame([[int(taxon_key[i]), species_name[i], 
                         gbif_species_name[i], int(image_count)]],
                                                columns=columns), ignore_index=True)
      
            end = time.time()
            print('Time taken to download data for ', gbif_species_name[i], ' is - ', round(end-begin), 'sec for ', image_count, ' images')

    count_list.to_csv(write_dir + 'datacount.csv', index=False)




