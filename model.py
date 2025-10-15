import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = 2):
        """
        the in channels meaning what the model takes in, it takes in 9 because our point giles are compiled to a 9 field vector
        the classes is how many outputs, only 2 right now for clean or corroded (definitely change later)
        """
        super().__init__()
        # three layers of learning applied to each point 
        self.mlp1 = nn.Linear(in_channels,64) # first layer learns 64 features on each point
        self.mlp2 = nn.Linear(64,128) # second layer expands it to 128 feature
        self.mlp3 = nn.Linear(128,1024) # third layer expands it to a large vector of 1024 features
        
        # now we scale back down to get only two labels which are clean or corroded (could change later)
        self.fc1 = nn.Linear(1088,512) # 1088 from combining the third vector to the initila 64 feature one
        self.fc2 = nn.Linear(512,256) 
        self.fc3 = nn.Linear(256,num_classes) #output only one label per class
        
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        """
        forward the data through the model
        
        x is gonna be the input tensor made up of an B N and C
        B is the batch size, N is the number of points in the cloud
        C is the features per point
        
        this def is supposed to return a tensor where it predicts how likely it is to be corroded or not
        """
        
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        point_features = F.relu(self.mlp3(x)) # making a rich feature vector for every point