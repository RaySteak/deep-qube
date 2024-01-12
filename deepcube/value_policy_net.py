import torch
from torch import nn as nn

class ValuePolicyNet(nn.Module):
    def __init__(self, value_only = False):
        super().__init__()
        self.value_only = value_only
        
        self.elu = nn.ELU()
        
        self.fc1 = nn.Linear(20 * 24, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        
        self.fc3_value = nn.Linear(2048, 512)
        self.fc4_value = nn.Linear(512, 1)
        
        if not value_only:
            self.fc3_policy = nn.Linear(2048, 512)
            self.fc4_policy = nn.Linear(512, 12)
        
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3_value.weight)
        torch.nn.init.xavier_normal_(self.fc4_value.weight)
        
        if not value_only:
            torch.nn.init.xavier_normal_(self.fc3_policy.weight)
            torch.nn.init.xavier_normal_(self.fc4_policy.weight)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        
        value = self.fc3_value(x)
        value = self.elu(value)
        value = self.fc4_value(value)
        
        if self.value_only:
            return value
        
        policy = self.fc3_policy(x)
        policy = self.elu(policy)
        policy = self.fc4_policy(policy)
        
        return value, policy
