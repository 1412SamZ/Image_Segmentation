import os
import natsort
from torch.utils.data import Dataset 
from torchvision import datasets, transforms
from PIL import Image 
import torch
import numpy as np
import sys
sys.path.append(os.getcwd())
from library.utils.utils import rgb_to_mask
from config.config import config

CONFIG = config()
id2code = CONFIG.id2code

class cvDataset(Dataset):
    def __init__(self, img_pth, mask_pth, transform):
        super().__init__()
        self.img_pth = img_pth
        self.mask_pth = mask_pth 
        self.transform = transform 
        images = os.listdir(self.img_pth)
        masks = [img_name[:-4] + "_L" + img_name[-4:] for img_name in images]
        self.total_imgs = natsort.natsorted(images)
        self.total_masks = natsort.natsorted(masks) 
        
    def __len__(self):
        return len(self.total_imgs)
    
    def __getitem__(self, index):
        img_loc = os.path.join(self.img_pth, self.total_imgs[index])
        image = Image.open(img_loc).convert("RGB")
        mask_loc = os.path.join(self.mask_pth, self.total_masks[index])
        mask = Image.open(mask_loc).convert("RGB")
        
        output_image, rgb_mask = self.transform(image), self.transform(mask)
        
        output_image = transforms.Compose([transforms.ToTensor()])(output_image)
        rgb_mask = transforms.Compose([transforms.PILToTensor()])(rgb_mask)
        output_mask = rgb_to_mask(torch.from_numpy(np.array(rgb_mask)).permute(1, 2, 0), id2code)
        
        return output_image, output_mask, rgb_mask.permute(0,1,2)
    
class testDataset(Dataset):
    def __init__(self, img_pth, mask_pth, transform):
        super().__init__()
        self.img_pth = img_pth
        self.mask_pth = mask_pth
        self.transform = transform 
        images = os.listdir(self.img_pth)
        masks = [img_name[:-4] + "_L" + img_name[-4:] for img_name in images]
        self.total_imgs = natsort.natsorted(images)
        self.total_masks = natsort.natsorted(masks)
        
    def __len__(self):
        return len(self.total_imgs)
    
    def __getitem__(self, index):
        img_loc = os.path.join(self.img_pth, self.total_imgs[index])
        image = Image.open(img_loc).convert("RGB")
        output_image = self.transform(image)
        
        mask_loc = os.path.join(self.mask_pth, self.total_masks[index])
        mask = Image.open(mask_loc).convert("RGB")
        rgb_mask = self.transform(mask)
        
        return output_image, rgb_mask
    
    
    