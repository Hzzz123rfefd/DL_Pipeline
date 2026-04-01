import numpy as np
from torch.utils.data.dataloader import default_collate
import yaml

class OneHot:
    def __init__(self):
        self.dict = {}
        
    def _add(self, key, unique_values):
        self.dict[key] = unique_values
        
    def encode(self, key, value):
        unique_values = self.dict[key]
        one_hot_vector = [0] * len(unique_values)
        if value in unique_values:
            index = unique_values.index(value)
            one_hot_vector[index] = 1
        return one_hot_vector
     
class ValueNormalizer:
    def __init__(self, max_ratio = 1.05):
        self.max_ratio = max_ratio
        self.dict = {}
        

    def  _add(self, key, min_max_value:list):
        min = (float)(min_max_value[0])
        max = (float)(min_max_value[1])
        max = max * self.max_ratio
        self.dict[key] = [min, max]
        
    def encode(self, key, value):
        value = (float)(value)
        min_max_value = self.dict[key]
        return  (value - min_max_value[0])/(min_max_value[1] - min_max_value[0])

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def recursive_collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: recursive_collate_fn([b[key] for b in batch]) for key in batch[0]}
    else:
        return default_collate(batch)
    
def calculate_mse(x1, x2):
    assert x1.shape == x2.shape, "输入的形状必须相同"
    return np.mean((x1 - x2) ** 2)

def calculate_mae(x1, x2):
    assert x1.shape == x2.shape, "输入的形状必须相同"
    return np.mean(np.abs(x1 - x2))

def calculate_r2(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "输入的形状必须相同"
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0 
    return 1 - ss_res / ss_tot