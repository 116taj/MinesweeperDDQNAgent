import random
import numpy as np
import torch
import torch.nn as nn
from torchrl.data import PrioritizedReplayBuffer, ListStorage
import matplotlib.pyplot as plt
from MinesweeperDQN import MinesweeperDiscreetEnv

#get env
env = MinesweeperDiscreetEnv(training=True)

#CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 4 conv layers with 128 filters and relu as activation
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
        x = self.conv(x)\
        #flatten
        x = x.view(x.size(0), -1)
        return x

#init device and models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model2 = Net().to(device)

#print(device)
#set hyperparameters
learning_rate = 0.000005
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
model.train()
model2.eval()
#print(model.summary())
discount = 0.95
epsilon = 1
decay = 0.999
epsilon_min = 0.01
crewards = []
scores = []
wins = []
won = []
losses = []
training_episodes = 10000
#declare prioritized replay buffer with hyper params
batch_size = 256
max_size = 256*100
buffer = PrioritizedReplayBuffer(alpha=0.6,beta=0.9,storage=ListStorage(max_size=max_size),batch_size=batch_size)
tau = 0.01
#training
import time
start = time.time()
for i in range(training_episodes):
    #reset stat trackers
    total = 0
    ep_loss = 0
    timesteps = 0
    #reset env
    s = env.reset(True)
    terminated = False
    #reset action mask
    used_actions = torch.zeros(env.board_size*env.board_size).to(device)
    while not terminated:
        timesteps+=1
        #turn s into tensor for torch operations
        s_tensor = torch.from_numpy(s).float().to(device)
        #choose action epsilon greedy
        if torch.rand(1).item() < epsilon:
            a = random.randrange(env.board_size*env.board_size)
        else:
            #if non random choose best according to q val from model
            with torch.no_grad():
                qtensor = s_tensor.unsqueeze(0).unsqueeze(0)
                qvals = model(qtensor)
                a = torch.argmax(qvals).item()
            #if already done then choose 2nd best action (to speed up process)
            if (used_actions[a] < 0):
                qvals[0,used_actions == -torch.inf] = -torch.inf
                a = torch.argmax(qvals).item()
        #mask action
        used_actions[a] = -torch.inf 
        #step in env and turn all useful into tensors
        next_s, reward, terminated, info = env.step(a)
        next_s_tensor = torch.from_numpy(next_s).float().to(device)
        reward_tensor = torch.tensor(reward).float().to(device)
        a_tensor = torch.tensor(a).to(device)
        #add to replay buffer
        buffer.add((s_tensor,a_tensor,reward_tensor,next_s_tensor))
        total+=reward
        #if buffer large enough
        if (len(buffer) > batch_size):
            #sample from buffer
            batch, info = buffer.sample(return_info=True)
            states, actions, rewards, next_states = batch
            #ensure dims are okay and get qvals (with grad since we need loss and optim)
            states = states.unsqueeze(1)
            qvals = model(states)
            qvals = qvals.gather(1,actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                #get target
                next_states = next_states.unsqueeze(1)
                next_qvals = model2(next_states).max(1).values
                target = rewards + discount * next_qvals
            #calculate loss and td error for priority
            loss = nn.functional.smooth_l1_loss(qvals,target)
            ep_loss+=loss.item()
            tderror = torch.abs(qvals - target).detach()
            buffer.update_priority(info["index"],tderror)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #update 2nd model using tau
            policynet = model.state_dict()
            targetnet = model2.state_dict()
            for key in policynet:
                targetnet[key] = policynet[key]*tau + targetnet[key]*(1-tau)
            model2.load_state_dict(targetnet)
            #decay epsilon
            if epsilon > epsilon_min:
                epsilon *= decay
        #go to next state
        s = next_s
    #add to stats
    losses.append(ep_loss/timesteps)
    #print(i)
    #print(total)
    #print(ep_loss/timesteps)
    if (reward == 1):
        wins.append(1)
    else:
        wins.append(0)
    scores.append(total)
    #update moving average for reward and WR
    if (i+1) % 100 == 0:
        won.append(np.mean(wins))
        crewards.append(np.mean(scores))
        wins = []
        scores = []

'''save and plot stats
torch.save(model.state_dict(), "model1life.pth")
torch.save(model2.state_dict(), "model2.pth")
print("done in ")
end = time.time() - start
print(end)
print(won)
print(crewards)
#show data
episodes = list(range(100,10100,100))
won = [0.03, 0.05, 0.12, 0.37, 0.39, 0.53, 0.5, 0.57, 0.59, 0.62, 0.63, 0.59, 0.65, 0.67, 0.77, 0.66, 0.69, 0.74, 0.69, 0.66, 0.77, 0.66, 0.71, 0.67, 0.69, 0.67, 0.74, 0.73, 0.7, 0.7, 0.7, 0.78, 0.67, 0.73, 0.72, 0.78, 0.74, 0.75, 0.69, 0.8, 0.78, 0.67, 0.71, 0.78, 0.72, 0.77, 0.72, 0.79, 0.64, 0.77, 0.74, 0.82, 0.77, 0.7, 0.76, 0.72, 0.76, 0.74, 0.76, 0.74, 0.7, 0.78, 0.78, 0.77, 0.8, 0.8, 0.72, 0.74, 0.75, 0.73, 0.78, 0.73, 0.78, 0.78, 0.8, 0.74, 0.79, 0.74, 0.78, 0.81, 0.79, 0.78, 0.74, 0.76, 0.78, 0.77, 0.69, 0.75, 0.8, 0.76, 0.7, 0.81, 0.74, 0.77, 0.76, 0.74, 0.75, 0.74, 0.79, 0.76]
plt.plot(episodes, won, label='Average Win Rate For Past 100 Episodes')
plt.xlabel('Episode Number')
plt.ylabel('Average Win Rate')
plt.title("Average Win Rate for DDQN")
plt.legend()
plt.show()
plt.plot(crewards, episodes,label='Average Reward For Past 100 Episodes')
plt.xlabel('Episode Number')
plt.ylabel('Average Reward')
plt.title("Average Reward for DDQN")
plt.legend()
plt.show()
plt.plot(losses,label='Smooth L1 Loss')
plt.xlabel('Episode Number')
plt.ylabel('Loss')
plt.title("Loss for DDQN over Episodes")
plt.legend()
plt.show()

'''