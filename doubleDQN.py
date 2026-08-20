import random
import numpy as np
import torch
import torch.nn as nn
from torchrl.data import PrioritizedReplayBuffer, ListStorage
import matplotlib.pyplot as plt
from environment import MinesweeperDiscreteEnv
from model import Net
#get env
env = MinesweeperDiscreteEnv()

#init device and models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)
model2 = Net().to(device)

#set hyperparameters
learning_rate = 0.000005
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
model.train()
model2.eval()
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
best_wr = 0.0
#training
for i in range(training_episodes):
    #reset stat trackers
    total = 0
    ep_loss = 0
    timesteps = 0
    train_steps = 0
    #reset env
    s = env.reset(True)
    terminated = False
    #reset action mask
    valid = env.valid_actions.copy()
    while not terminated:
        timesteps+=1
        #turn s into tensor for torch operations
        s_tensor = torch.from_numpy(s).float().to(device)
        valid_tensor = torch.from_numpy(valid).to(device)
        #choose action epsilon greedy
        if torch.rand(1).item() < epsilon:
            a = int(random.choice(np.flatnonzero(valid)))
        else:
            #if non random choose best according to q val from model
            with torch.no_grad():
                qtensor = s_tensor.unsqueeze(0).unsqueeze(0)
                qvals = model(qtensor)
                qvals = qvals.masked_fill(~valid_tensor.unsqueeze(0), -1e9)
                a = torch.argmax(qvals).item()
        #step in env and turn all useful into tensors
        next_s, reward, terminated, info = env.step(a)
        next_valid = env.valid_actions.copy()
        next_s_tensor = torch.from_numpy(next_s).float().to(device)
        reward_tensor = torch.tensor(reward).float().to(device)
        a_tensor = torch.tensor(a).to(device)
        done_tensor = torch.tensor(float(terminated)).to(device)
        next_valid_tensor = torch.from_numpy(next_valid).to(device)
        #add to replay buffer
        buffer.add((s_tensor,a_tensor,reward_tensor,next_s_tensor,done_tensor,next_valid_tensor))
        total+=reward
        #if buffer large enough
        if (len(buffer) > batch_size):
            #sample from buffer
            batch, info = buffer.sample(return_info=True)
            states, actions, rewards, next_states, dones, next_valids = batch
            #ensure dims are okay and get qvals (with grad since we need loss and optim)
            states = states.unsqueeze(1)
            qvals = model(states)
            qvals = qvals.gather(1,actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                #get target
                next_states = next_states.unsqueeze(1)
                next_actions = model(next_states).masked_fill(~next_valids, -1e9).argmax(1, keepdim=True)
                next_qvals = model2(next_states).gather(1,next_actions).squeeze(1)
                target = rewards + discount * next_qvals * (1 - dones)
            #calculate loss and td error for priority
            loss = nn.functional.smooth_l1_loss(qvals,target)
            ep_loss+=loss.item()
            train_steps+=1
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
        valid = next_valid
    #add to stats
    losses.append(ep_loss/train_steps if train_steps > 0 else 0.0)
    if (reward == 1):
        wins.append(1)
    else:
        wins.append(0)
    scores.append(total)
    #update moving average for reward and WR
    if (i+1) % 100 == 0:
        won.append(np.mean(wins))
        crewards.append(np.mean(scores))
        print(f"ep {i+1:>5}  winrate {np.mean(wins):.3f}  reward {np.mean(scores):>7.2f}  loss {np.mean(losses[-100:]):.4f}  eps {epsilon:.3f}")
        if won[-1] >= best_wr:
            best_wr = won[-1]
            torch.save(model.state_dict(), "model_best.pth")
        wins = []
        scores = []


torch.save(model.state_dict(), "model.pth")
#show data
episodes = list(range(100, 100*len(won)+1, 100))
plt.plot(episodes, won, label='Average Win Rate For Past 100 Episodes')
plt.xlabel('Episode Number')
plt.ylabel('Average Win Rate')
plt.title("Average Win Rate for DDQN")
plt.legend()
plt.show()
plt.plot(episodes, crewards,label='Average Reward For Past 100 Episodes')
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
