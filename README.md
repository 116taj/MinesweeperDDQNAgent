# Minesweeper DDQN Agent

A neural network that learns Minesweeper by playing it, in PyTorch on a
Gymnasium environment.

It is never told the rules. It sees what a player sees, the uncovered numbers
and nothing else, and it clicks a square. Safe clicks pay a little, mines cost,
and over tens of thousands of games it tunes itself toward more reward. What
comes out is a network that takes a board and scores all 100 squares by how good
a click each one looks. The method is a Double Deep Q-Network.

Minesweeper is an awkward fit for it. The mines are never shown, so some
positions cannot be solved at all and the agent has to guess. The legal moves
also shrink as a game goes on, since a square can only be clicked once.

## Results

1000 episodes, greedy policy, no exploration. `evalDDQN.py` against `model.pth`:

| Configuration | Win rate |
| --- | --- |
| 10x10 board, 9 mines, 3 lives | **91%** |

That setup is not standard Minesweeper, so the number does not compare to one:

- 9 mines on 100 cells is 9% density. The classic beginner preset is 10 mines
  on 81 cells, 12.3%.
- The agent gets **3 lives**. A mine costs a life and play continues. The third
  one ends the episode. On a single life the win rate is lower.
- No first-click safety, so the opening move hits mines.

I added the lives so the penalty for a mine could change without also changing
when an episode ends. That way I could see how the cost of a mistake affects how
willing the agent is to take an uncertain cell. `life_count=1` gives the
standard rules back.

## Approach

**Fully convolutional Q-network.** `model.py` is four layers
(1 -> 128 -> 128 -> 128 -> 1): three 3x3 convolutions with padding 1 and ReLU,
then a 1x1 convolution down to a single channel, flattened into 100 Q-values,
one per cell. There is no fully connected layer, so a pattern in the top-left
gets the same response in the bottom-right. Minesweeper deductions are local and
read the same anywhere on the board, and the network cannot learn coordinates.

**Action masking.** The action space is `Discrete(100)`, one action per cell,
and a revealed cell is an illegal move. Invalid logits are set to `-1e9` before
every argmax, both when picking an action and inside the target computation.
Without it the agent spends training re-clicking cells it has already opened.

**Double DQN.** The online network picks the next action and the target network
scores it, which holds down the overestimation vanilla DQN runs into. Early
boards are full of cells that look identical, so there is room for it to build
up.

**Prioritized experience replay.** `torchrl.PrioritizedReplayBuffer`, alpha 0.6,
beta 0.9, capacity 25600, batch size 256. Priorities come from the absolute TD
error after each gradient step. Most transitions here are flood-fill reveals the
network already predicts well, and uniform sampling spends its updates on those.

**Soft target updates.** The target network is blended toward the online one at
tau = 0.01 every training step, instead of being copied across at a fixed
interval.

### Reward function

| Event | Reward |
| --- | --- |
| Win (all safe cells revealed) | +1 |
| Mine hit, lives remaining | -0.1 |
| Mine hit, no lives remaining | -1 |
| Safe reveal | +0.02 |

The small reward per safe reveal keeps it opening cells instead of stalling. It
is much smaller than the mine penalty, so no amount of safe clicking pays for
dying.

### Hyperparameters

| | |
| --- | --- |
| Optimizer | Adam, lr 5e-6 |
| Loss | Smooth L1 (Huber) |
| Discount | 0.95 |
| Epsilon | 1.0 -> 0.01, decay 0.999 per training step |
| Batch size | 256 |
| Training episodes | 10,000 |

## Observation encoding

The observation is the visible 10x10 board, one channel of integers:

```
-2   closed
-1   revealed mine
0-8  number of adjacent mines
```

That goes into the network raw, not one-hot encoded per cell value. One-hot is
the more common choice in published Minesweeper RL work, and it is the first
thing I would try for better accuracy.

## Running

```sh
pip install -r requirements.txt
python doubleDQN.py    # train 10,000 episodes, save model.pth, plot curves
python evalDDQN.py     # greedy evaluation over 1000 episodes, print win rate
```

`doubleDQN.py` plots win rate, average reward and loss once training finishes.
`environment.py` takes `render_mode='human'` for a pygame view of the board and
the lives left, which is what I used to watch where the agent goes wrong.

## Files

```
model.py           Fully convolutional Q-network
environment.py     Gymnasium Minesweeper env, lives, action mask, renderer
doubleDQN.py       Training loop with prioritized replay
evalDDQN.py        Greedy evaluation over 1000 episodes
model.pth          The weights behind the result above
requirements.txt   Dependencies
```

## Caveats

- There is no baseline here to compare against. The number to beat is a naive
  probability solver on the same 10x10, 9-mine, 3-life setup.
- `place_mines` places mines uniformly at random and pays no attention to the
  first move, so some of the losses are not the policy's fault.
- One training run, one seed, no averaging.

## Credits

The Gymnasium environment is adapted from
[aylint/gym-minesweeper](https://github.com/aylint/gym-minesweeper). The lives,
action masking, reward shaping, pygame renderer and the learning pipeline are
mine.
