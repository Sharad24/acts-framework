import torch
import torch.nn as nn
import torch.nn.functional as F
import io

class hitClassify(nn.Module):
    def __init__(self):
        super(hitClassify, self).__init__()
        self.fc1 = nn.Linear(8, 15)
        self.fc2 = nn.Linear(15, 5)
        self.fc3 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), 1)
        return x

    
m = torch.jit.script(hitClassify())
torch.jit.save(m, "/tmp/tmp.pb")


