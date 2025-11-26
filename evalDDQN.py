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
env = MinesweeperDiscreteEnv(10,9,3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model.load_state_dict(torch.load("model1life.pth",map_location=device)) 
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
        flattened_s = s.flatten()
        empty = np.where(flattened_s != -2)[0]
        empty = torch.from_numpy(empty).to(device)
        used_actions[empty] = -1e9
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
        env.render()
        print("Took action at row "+str(int(a/10)+1)+" col "+str(a%10+1))
        time.sleep(1)
    if (reward == 1):
        print("agent win")
        won = np.append(won,1)
    else:
        print("agent loss")
        won = np.append(won,0)
    scores = np.append(scores,total)
print("done in ")
end = time.time() - start
print(end)
print(np.average(won))
