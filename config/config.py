
import torch
import os
from torch import cuda
import sys 
sys.path.append(os.getcwd())
from library.utils.utils import color_map
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



print(os.getcwd())
class config(object):

    model_path = "./model/"
    path = "./datasets/"
    load_model = "./model/state_dict.pt"
    batch = 4
    lr = 0.0001
    epochs = 40
    input_size = (128,128)
    if cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device('cpu')
    code2id, id2code, name2id, id2name = color_map(path+'class_dict.csv')