"""
Author: Aditya Jain
Date  : August 3, 2022
About : Description of EfficientNet Model
"""
import timm

class EfficientNet:
	def __init__(self, config):
		"""
        Args:
            config: provides parameters for model generation
        """
		super(Resnet50, self).__init__()
		self.num_classes = config['model']['species_num_classes']
		self.model_name  = config['model']['type']
        
	def get_model(self):
		if self.model_name == 'efficientnetv2-b3':
			return timm.create_model('tf_efficientnetv2_b3',
								pretrained=True,
								num_classes=self.num_classes)
		
		else:
			raise RuntimeError(f'Model {self.model_name} not implemented') 