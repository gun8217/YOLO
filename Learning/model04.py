import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=32)
        self.fc1_act = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc2_act = nn.ReLU()
        
        self.fc3 = nn.Linear(in_features=16, out_features=4)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_act(x)
        
        x = self.fc2(x)
        x = self.fc2_act(x)
        
        x = self.fc3(x)
        return x