import random
import numpy as np
import torch
import torch.nn as nn
import os
from torchrl.data import ReplayBuffer, ListStorage
import matplotlib.pyplot as plt
from MinesweeperDQN import MinesweeperDiscreetEnv
#eval for DDQN
env = MinesweeperDiscreetEnv(10,9)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model.load_state_dict(torch.load("model.pth",map_location=device)) 
scores = np.array([])
won = np.array([])
import time
start = time.time()
for i in range(1000):
    total = 0
    s = env.reset(True)
    terminated = False
    used_actions = torch.zeros(env.board_size*env.board_size).to(device)
    while not terminated:
        with torch.no_grad():
            qtensor = torch.from_numpy(s).float().to(device).unsqueeze(0).unsqueeze(0)
            qvals = model(qtensor)
            a = torch.argmax(qvals).item()
            if (used_actions[a] < 0):
                qvals+=used_actions
                a = torch.argmax(qvals).item()
            used_actions[a] = -np.inf 
        next_s, reward, terminated, empty = env.step(a)
        total+=reward
        s = next_s
            #print(qvals)
    print(i)
    print(total)
    if (reward == 1):
        won = np.append(won,1)
    else:
        won = np.append(won,0)
    scores = np.append(scores,total)
print("done in ")
end = time.time() - start
print(end)
print(np.average(won))
