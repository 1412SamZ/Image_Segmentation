


import torch
import torch.nn as nn
import numpy as np
from library.utils.iou import *
from library.utils.utils import *
from config.config import config

CONFIG = config()

def Validate(model, validloader, criterion, valid_loss_min, device, model_path):
    valid_loss = 0
    val_iou = []
    val_losses = []
    model.eval()
    for i, val_data in enumerate(validloader):
        inp, masks, _ = val_data
        inp, masks = inp.to(device), masks.to(device)
        out = model(inp)
        val_target = masks.argmax(1)
        val_loss = criterion(out, val_target.long())
        valid_loss += val_loss.item() * inp.size(0)
        iou = iou_pytorch(out.argmax(1), val_target)
        val_iou.extend(iou)    
    miou = torch.FloatTensor(val_iou).mean()
    valid_loss = valid_loss / len(validloader.dataset)
    val_losses.append(valid_loss)
    print(f'\t\t Validation Loss: {valid_loss:.4f},',f' Validation IoU: {miou:.3f}')
    
    if np.mean(val_losses) <= valid_loss_min:
        torch.save(model.state_dict(), model_path+'/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,np.mean(val_losses))+'\n')
        valid_loss_min = np.mean(val_losses)

    return valid_loss, valid_loss_min

def Test_eval(model, testloader, criterion, model_save_pth, device):
      model.load_state_dict(torch.load(model_save_pth))
      model.eval()
      test_loss = 0
      imgs, masks, preds = [], [], []
      for i, test_data in enumerate(testloader):
        img, mask = test_data
        inp, mask = img.to(device), mask.to(device)
        imgs.extend(inp.cpu().numpy())
        masks.extend(mask.cpu().numpy())
        out = model(inp.float())
        preds.extend(out.detach().cpu().numpy())
        target = mask.argmax(1)
        loss = criterion(out, target.long())
        test_loss += loss.item() * inp.size(0)
      test_loss = loss / len(testloader.dataset)
      pred = mask_to_rgb(np.array(preds), CONFIG.id2code)
      print(f"Test loss is: {test_loss:.4f}")
      return np.array(imgs), np.array(masks), np.array(pred)
  
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from library.models.unet import *
from library.loss.loss import FocalLoss
from config import config
from library.utils.utils import *
from library.dataset.cv_dataset import *
from library.utils.iou import *
#from library.eval import *

CONFIG = config()

path = CONFIG.path
batch = CONFIG.batch
lr = CONFIG.lr
epochs = CONFIG.epochs
device = CONFIG.device
print(f"The device being used is: {device}\n")
id2code = CONFIG.id2code
input_size = CONFIG.input_size
model_sv_pth = CONFIG.model_path
load_model_pth = CONFIG.load_model

def train(model, trainloader, validloader, criterion, optimizer, epochs, device, load_pth, model_sv_pth, plot=True, visualize=False, load_model=False):
    if load_model: model.load_state_dict(torch.load(load_pth))
    model.train()
    stats = []
    valid_loss_min = np.Inf
    print('Training Started.....')
    for epoch in range(epochs):
        train_loss = 0
        train_iou = []
        for i, data in enumerate(trainloader):
            inputs, mask, rgb = data
            inputs, mask = inputs.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(inputs.float())
            target = mask.argmax(1)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0) 
            iou = iou_pytorch(output.argmax(1), target)
            train_iou.extend(iou)     
            if visualize and epoch%10==0 and i == 0:
                print('The training images')
                show_batch(inputs.detach().cpu(), size=(8,8))
                print('The original masks')
                show_batch(rgb.detach().cpu(), size=(8,8))
                RGB_mask =  mask_to_rgb(output.detach().cpu(), id2code)
                print('Predicted masks')
                show_batch(torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
        miou = torch.FloatTensor(train_iou).mean()
        train_loss = train_loss / len(trainloader.dataset)
        print('Epoch',epoch,':',f'Lr ({optimizer.param_groups[0]["lr"]})',f'\n\t\t Training Loss: {train_loss:.4f},',f' Training IoU: {miou:.3f},')
        
        with torch.no_grad():
            valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth)
            
        stats.append([train_loss, valid_loss])
        stat = pd.DataFrame(stats, columns=['train_loss', 'valid_loss'])

    print('Finished Training')
    if plot: plotCurves(stat)


if __name__ == "__main__":
    
    #Define transforms for the training data and validation data
    train_transforms = transforms.Compose([transforms.Resize(input_size, 0)])
    valid_transforms = transforms.Compose([transforms.Resize(input_size, 0)])

    #pass transform here-in
    train_data = cvDataset(img_pth = path + 'train/', mask_pth = path + 'train_labels/', transform = train_transforms)
    valid_data = cvDataset(img_pth = path + 'val/', mask_pth = path + 'val_labels/', transform = valid_transforms)

    #data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)

    model = UNet(3, 32).to(device)
    criterion = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    train(model, trainloader, validloader, criterion, optimizer, epochs, device, load_model_pth, model_sv_pth, plot=True, visualize=True, load_model=False)