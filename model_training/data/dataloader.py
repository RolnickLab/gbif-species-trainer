"""
Author       : Aditya Jain
Date Started : July 19, 2022
About        : Data loader functions for training
"""
from torchvision import transforms
import torch
import webdataset as wds


def get_transforms(input_size=224, is_training=True, preprocess_mode='torch'):
	"""transform to be applied to each image"""

	if preprocess_mode == 'torch':
		mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	elif preprocess_mode == 'tf':
		mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
	else:
		mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

	if is_training:
		return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])
	else:
		return transforms.Compose([
                transforms.Resize((input_size, input_size)), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ])


def identity(x):
	return x


def build_webdataset_pipeline(sharedurl, input_size, batch_size, is_training, num_workers, preprocess_mode):
	"""main dataset building function"""

	transform = get_transforms(input_size, is_training, preprocess_mode)
	dataset   = wds.WebDataset(sharedurl, shardshuffle=is_training)  
	if is_training:
		dataset = dataset.shuffle(10000)
  
	dataset = (dataset.decode('pil')
              .to_tuple('jpg', 'cls')
              .map_tuple(transform, identity))

	loader  = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_size=batch_size)

	return loader


