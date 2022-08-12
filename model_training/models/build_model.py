"""
Author: Aditya Jain
Date  : August 3, 2022
About : Main function for building a model
"""
from models.resnet50 import Resnet50
from models.efficientnet import EfficientNet

def build_model(config):
	model_name  = config['model']['type']
	
	if model_name == 'resnet50':
		return Resnet50(config)
	elif model_name == 'efficientnetv2-b3':
		return EfficientNet(config).get_model()
	else:
		raise RuntimeError(f'Model {self.model_name} not implemented') 