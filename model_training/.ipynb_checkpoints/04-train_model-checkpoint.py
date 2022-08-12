#!/usr/bin/env python
# coding: utf-8

"""
Author       : Aditya Jain
Date Started : June 14, 2022
About        : This is the main training file for training the uk moth classifier
"""

import wandb
import torchvision.models as torchmodels
import torch
from torch import nn
from torchsummary import summary
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import torch.optim as optim
import datetime
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
import argparse
from data import dataloader
import random

from models.build_model import build_model
from data.mothdataset import MOTHDataset
from training_params.loss import Loss
from training_params.optimizer import Optimizer
from evaluation.micro_accuracy_batch import MicroAccuracyBatch
from evaluation.micro_accuracy_batch import add_batch_microacc, final_microacc
from evaluation.macro_accuracy_batch import MacroAccuracyBatch
from evaluation.macro_accuracy_batch import add_batch_macroacc, final_macroacc, taxon_accuracy
from evaluation.confusion_matrix_data import confusion_matrix_data
from evaluation.confusion_data_conversion import ConfusionDataConvert

def train_model(args):
	"""main function for training"""

	config_file   = args.config_file
	f             = open(config_file)
	config_data   = json.load(f)
	print(json.dumps(config_data, indent=3))
	
	wandb.init(project=config_data['training']['wandb']['project'], entity=config_data['training']['wandb']['entity'])
	wandb.init(settings=wandb.Settings(start_method="fork"))

	image_resize  = config_data['training']['image_resize']
	batch_size    = config_data['training']['batch_size']
	label_list    = config_data['dataset']['label_info']
	epochs        = config_data['training']['epochs']
	loss_name     = config_data['training']['loss']['name']
	early_stop    = config_data['training']['early_stopping']
	start_val_los = config_data['training']['start_val_loss']

	label_read    = json.load(open(label_list))
	species_list  = label_read['species_list']
	genus_list    = label_read['genus_list']
	family_list   = label_read['family_list']

	no_species_cl = config_data['model']['species_num_classes']
	no_genus_cl   = config_data['model']['genus_num_classes']
	no_family_cl  = config_data['model']['family_num_classes']
	model_type    = config_data['model']['type']
	preprocess_mode = config_data['model']['preprocess_mode']

	opt_name      = config_data['training']['optimizer']['name']
	learning_rate = config_data['training']['optimizer']['learning_rate']
	momentum      = config_data['training']['optimizer']['momentum']

	mod_save_pth  = config_data['training']['model_save_path']
	mod_name      = config_data['training']['model_name']
	mod_ver       = config_data['training']['version']
	DTSTR         = datetime.datetime.now()
	DTSTR         = DTSTR.strftime("%Y-%m-%d-%H-%M")
	save_path     = mod_save_pth + mod_name + '_' + mod_ver + '_' + model_type + '_' + DTSTR + '.pt'

	taxon_hierar  = config_data['dataset']['taxon_hierarchy']
	label_info    = config_data['dataset']['label_info']


	# Loading Model
	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(device)
	model = build_model(config_data).to(device)

	# Loading Data
	# Training data loader
	train_dataloader = dataloader.build_webdataset_pipeline(
		sharedurl=args.train_webdataset_url,
		input_size=image_resize,
		batch_size=batch_size,
		is_training=True,
		num_workers=args.dataloader_num_workers,
		preprocess_mode=preprocess_mode)


	# Validation data loader
	val_dataloader = dataloader.build_webdataset_pipeline(
		sharedurl=args.val_webdataset_url,
		input_size=image_resize,
		batch_size=batch_size,
		is_training=False,
		num_workers=args.dataloader_num_workers,
		preprocess_mode=preprocess_mode)

	# Testing data loader
	test_dataloader = dataloader.build_webdataset_pipeline(
		sharedurl=args.test_webdataset_url,
		input_size=image_resize,
		batch_size=batch_size,
		is_training=False,
		num_workers=args.dataloader_num_workers,
		preprocess_mode=preprocess_mode)


	# Loading Loss function and Optimizer
	loss_func = Loss(loss_name).func()
	optimizer = Optimizer(opt_name, model, learning_rate, momentum).func()


	# Model Training
	lowest_val_loss = start_val_los
	early_stp_count = 0

	for epoch in tqdm(range(epochs)):
		train_loss      = 0
		train_batch_cnt = 0
		val_loss        = 0
		val_batch_cnt   = 0
		s_time          = time.time()
    
		global_microacc_data_train = None
		global_microacc_data_val   = None
    
		# model training on training dataset
		model.train()                      
		for image_batch, label_batch in train_dataloader:    
			image_batch, label_batch = image_batch.to(device), label_batch.to(device)          
        
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs   = model(image_batch)      
			t_loss    = loss_func(outputs, label_batch)
			t_loss.backward()
			optimizer.step()        
			train_loss += t_loss.item()
        
			# micro-accuracy calculation
			micro_accuracy_train          = MicroAccuracyBatch(outputs, label_batch, label_info, taxon_hierar).batch_accuracy()   
			global_microacc_data_train    = add_batch_microacc(global_microacc_data_train, micro_accuracy_train)
			train_batch_cnt += 1
		train_loss = train_loss/train_batch_cnt

        
		# model evaluation on validation dataset
		model.eval()                       
		for image_batch, label_batch in val_dataloader:
			image_batch, label_batch = image_batch.to(device), label_batch.to(device)        
        
			outputs   = model(image_batch)        
			v_loss    = loss_func(outputs, label_batch)
			val_loss += v_loss.item()    
        
			# micro-accuracy calculation
			micro_accuracy_val          = MicroAccuracyBatch(outputs, label_batch, label_info, taxon_hierar).batch_accuracy()   
			global_microacc_data_val    = add_batch_microacc(global_microacc_data_val, micro_accuracy_val)
			val_batch_cnt += 1
		val_loss = val_loss/val_batch_cnt

    
		if val_loss<lowest_val_loss:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'train_loss': train_loss,
				'val_loss':val_loss}, 
				save_path)        
			lowest_val_loss = val_loss
			early_stp_count = 0
		else:
			early_stp_count += 1

		# logging metrics
		wandb.log({'training loss': train_loss, 'validation loss': val_loss, 'epoch': epoch})
    
		final_micro_accuracy_train = final_microacc(global_microacc_data_train)
		final_micro_accuracy_val   = final_microacc(global_microacc_data_val) 
		wandb.log({'train_micro_species_top1': final_micro_accuracy_train['micro_species_top1'], 
				'train_micro_genus_top1': final_micro_accuracy_train['micro_genus_top1'],
				'train_micro_family_top1': final_micro_accuracy_train['micro_family_top1'],
				'val_micro_species_top1': final_micro_accuracy_val['micro_species_top1'], 
				'val_micro_genus_top1': final_micro_accuracy_val['micro_genus_top1'],
				'val_micro_family_top1': final_micro_accuracy_val['micro_family_top1'],
				'epoch': epoch   
              })   
    
		e_time = (time.time()-s_time)/60   # time taken in minutes    
		wandb.log({'time per epoch': e_time, 'epoch': epoch})
    
		if early_stp_count >= early_stop:
			break    

	wandb.log_artifact(save_path, name=mod_name, type='models')

	model.eval()                                          
	global_microacc_data     = None
	global_macroacc_data     = None
	global_confusion_data_sp = None
	global_confusion_data_g  = None
	global_confusion_data_f  = None

	print("Prediction on test data started ...")

	with torch.no_grad():                                 
		for image_batch, label_batch in test_dataloader:  
			image_batch, label_batch = image_batch.to(device), label_batch.to(device)
			predictions              = model(image_batch)
        
			# micro-accuracy calculation
			micro_accuracy           = MicroAccuracyBatch(predictions, label_batch, label_info, taxon_hierar).batch_accuracy()   
			global_microacc_data     = add_batch_microacc(global_microacc_data, micro_accuracy)
        
			# macro-accuracy calculation
			macro_accuracy           = MacroAccuracyBatch(predictions, label_batch, label_info, taxon_hierar).batch_accuracy()
			global_macroacc_data     = add_batch_macroacc(global_macroacc_data, macro_accuracy) 
        
			# confusion matrix
			sp_label_batch, sp_predictions, g_label_batch, g_predictions, f_label_batch, f_predictions = ConfusionDataConvert(predictions, label_batch, label_info, taxon_hierar).converted_data()   
        
			global_confusion_data_sp = confusion_matrix_data(global_confusion_data_sp, [sp_label_batch, sp_predictions])
			global_confusion_data_g  = confusion_matrix_data(global_confusion_data_g, [g_label_batch, g_predictions])
			global_confusion_data_f  = confusion_matrix_data(global_confusion_data_f, [f_label_batch, f_predictions])        

	final_micro_accuracy            = final_microacc(global_microacc_data)
	final_macro_accuracy, taxon_acc = final_macroacc(global_macroacc_data)
	tax_accuracy                    = taxon_accuracy(taxon_acc, label_read)


	# saving evaluation data to file
	confdata_pd_f  = pd.DataFrame({'F_Truth': global_confusion_data_f[0].reshape(-1), 'F_Prediction': global_confusion_data_f[1].reshape(-1)})
	confdata_pd_g  = pd.DataFrame({'G_Truth': global_confusion_data_g[0].reshape(-1), 'G_Prediction': global_confusion_data_g[1].reshape(-1)})
	confdata_pd_sp = pd.DataFrame({'S_Truth': global_confusion_data_sp[0].reshape(-1), 'S_Prediction': global_confusion_data_sp[1].reshape(-1)})
	confdata_pd    = pd.concat([confdata_pd_f, confdata_pd_g, confdata_pd_sp], axis=1)
	confdata_pd.to_csv(mod_save_pth + mod_ver + '_confusion-data.csv', index=False)

	with open(mod_save_pth + mod_ver + '_micro-accuracy.json', 'w') as outfile:
		json.dump(final_micro_accuracy, outfile)

	with open(mod_save_pth + mod_ver + '_macro-accuracy.json', 'w') as outfile:
		json.dump(final_macro_accuracy, outfile)
    
	with open(mod_save_pth + mod_ver + '_taxon-accuracy.json', 'w') as outfile:
		json.dump(tax_accuracy, outfile)

	wandb.log({'final micro accuracy' : final_micro_accuracy})
	wandb.log({'final macro accuracy' : final_macro_accuracy})
	wandb.log({'configuration' : config_data})
	wandb.log({'tax accuracy' : tax_accuracy})

	label_f = tf.keras.utils.to_categorical(global_confusion_data_f[0], num_classes=no_family_cl)
	pred_f  = tf.keras.utils.to_categorical(global_confusion_data_f[1], num_classes=no_family_cl)

	wandb.finish()

def set_random_seed(random_seed):
	"""set random seed for reproducibility"""
	random.seed(random_seed)
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)
	torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_webdataset_url', help = 'path to webdataset tar files for training', required=True)
	parser.add_argument('--val_webdataset_url', help = 'path to webdataset tar files for validation', required=True)
	parser.add_argument('--test_webdataset_url', help = 'path to webdataset tar files for testing', required=True)
	parser.add_argument('--config_file', help = 'path to configuration file containing training information', required=True)
	parser.add_argument('--dataloader_num_workers', help = 'number of cpus available', required=True, type=int)
	parser.add_argument('--random_seed', help = 'random seed for reproducible experiments', default=42, type=int)
	args   = parser.parse_args()
	
	set_random_seed(args.random_seed)
	train_model(args)