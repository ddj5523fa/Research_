#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch.nn.functional as F
import torch
from torch import nn
from torchvision.utils import make_grid
import torchvision.models as models

# In[3]:


class TrainerBase(nn.Module):

    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        mse_loss = nn.MSELoss()
        loss = mse_loss(out, targets)
        #loss = 1 - F.cosine_similarity(out,targets).mean()
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)
        mse_loss = nn.MSELoss()
        loss = mse_loss(out, targets)
        #loss = 1 - F.cosine_similarity(out,targets).mean()
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss']))


# In[4]:


class AttriGenModel(TrainerBase):
    def __init__(self,out):
        super().__init__()
        # Use a pretrained model
        #model.load_state_dict(torch.load(model_path))
        self.network = models.resnet101(pretrained=True)
        #self.network = self.load_state_dict(torch.load(resnet_path))
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(nn.Flatten(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm1d(num_ftrs//2),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(in_features=num_ftrs//2,out_features=num_ftrs//4),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm1d(num_ftrs//4),
                                       nn.Dropout(p=0.3), 
                                       nn.Linear(num_ftrs//4,out))  
    def forward(self, xb):
        return self.network(xb)

### Retrained ResNet

#res101=models.resnet101(pretrained=True)

'''
class resnet_retrained_model(TrainerBase):
    def __init__(self):
        super(resnet_retrained_model, self).__init__()
        self.network=models.resnet101(pretrained=True)
        self.network.fc = nn.Sequential(
                           nn.Linear(2048, 1024),
                           nn.ReLU(inplace=True),
                           nn.Linear(1024,40),
                           nn.Softmax(dim=0)
                            )
    def forward(self, x):
        return self.network(x)
###############################################################################
#                            END TODO                                         #
###############################################################################
x = torch.randn(4, 3, 224, 224)
resnet_retrained_model = resnet_retrained_model()
#make_dot(mymodel(x), params=dict(model.named_parameters()))
'''

'''
resnet_retrained_model.load_state_dict(torch.load(resnet_path))
resnet_retrained_model.eval()
class AttriGenModel_RetrainVer(TrainerBase):
    def __init__(self):
        super(AttriGenModel_RetrainVer, self).__init__()
        self.Net=resnet_retrained_model
        num_ftrs = self.Net.network.fc[0].in_features
        self.Net.network.fc = nn.Sequential(nn.Flatten(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(in_features=num_ftrs,out_features=num_ftrs//2),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm1d(num_ftrs//2),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(in_features=num_ftrs//2,out_features=num_ftrs//4),
                                       nn.ReLU(inplace=True),
                                       nn.BatchNorm1d(num_ftrs//4),
                                       nn.Dropout(p=0.3), 
                                       nn.Linear(num_ftrs//4,out_dim))
    def forward(self, x):
        return self.Net(x)
###############################################################################
#                            END TODO                                         #
###############################################################################
#make_dot(mymodel(x), params=dict(model.named_parameters()))
'''