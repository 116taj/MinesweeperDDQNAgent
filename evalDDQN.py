import random
import numpy as np
import torch
import torch.nn as nn
import os
from torchrl.data import ReplayBuffer, ListStorage
import matplotlib.pyplot as plt
from environment import MinesweeperDiscreteEnv
from model import Net
#eval for DDQN
env = MinesweeperDiscreteEnv(10,9,3,render_mode='human')

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
    valid = env.valid_actions.copy()
    while not terminated:
        valid_tensor = torch.from_numpy(valid).to(device)
        with torch.no_grad():
            qtensor = torch.from_numpy(s).float().to(device).unsqueeze(0).unsqueeze(0)
            qvals = model(qtensor)
            qvals = qvals.masked_fill(~valid_tensor.unsqueeze(0), -1e9)
            a = torch.argmax(qvals).item()
        next_s, reward, terminated, info = env.step(a)
        total+=reward
        s = next_s
        valid = env.valid_actions.copy()
    if (reward == 1):
        won = np.append(won,1)
    else:
        won = np.append(won,0)
    scores = np.append(scores,total)
print("done in ")
end = time.time() - start
print(end)
print(np.average(won))
