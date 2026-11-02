# Minesweeper DDQN Agent

A Double Deep Q-Network that learns to play Minesweeper, written in PyTorch on
a Gymnasium environment. It trains from the visible board alone and is never
shown where the mines are.

## Results

1000 episodes, greedy policy, no exploration. `evalDDQN.py` against `model.pth`:

| Configuration | Win rate |
| --- | --- |
| 10x10 board, 9 mines, 3 lives | **91%** |

This is not the standard game, so the number does not compare to one:

- 9 mines on 100 cells is 9% density. The classic beginner preset is 10 mines
  on 81 cells, 12.3%.
- The agent has **3 lives**. A mine costs a life and play continues. The third
  one ends the episode.
- There is no first-click safety, so the opening move can hit a mine.

`life_count=1` runs it under standard rules.

## Approach

**Fully convolutional Q-network.** `model.py` is four layers
(1 -> 128 -> 128 -> 128 -> 1): three 3x3 convolutions with padding 1 and ReLU,
then a 1x1 convolution to a single channel, flattened into 100 Q-values, one per
cell. With no fully connected layer the network is translation-equivariant, so
it cannot key on absolute cell positions.

**Action masking.** The action space is `Discrete(100)`, one action per cell.
Revealed cells are illegal moves, and their logits are set to `-1e9` before
every argmax, in action selection and in the target computation.

**Double DQN.** The online network picks the next action and the target network
scores it, the standard correction for vanilla DQN's overestimation bias.

**Prioritized experience replay.** `torchrl.PrioritizedReplayBuffer`, alpha 0.6,
beta 0.9, capacity 25600, batch size 256. Priorities are updated from the
absolute TD error after every gradient step.

**Soft target updates.** The target network is blended toward the online network
at tau = 0.01 on every training step.

### Reward function

| Event | Reward |
| --- | --- |
| Win (all safe cells revealed) | +1 |
| Mine hit, lives remaining | -0.1 |
| Mine hit, no lives remaining | -1 |
| Safe reveal | +0.02 |

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

It goes into the network as is, not one-hot encoded per cell value.

## Running

```sh
pip install -r requirements.txt
python doubleDQN.py    # train 10,000 episodes, save model.pth, plot curves
python evalDDQN.py     # greedy evaluation over 1000 episodes, print win rate
```

`doubleDQN.py` plots win rate, average reward and loss when training ends.
`environment.py` takes `render_mode='human'` for a pygame view of the board and
the lives remaining.

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

- There is no baseline here to compare the win rate against.
- `place_mines` places mines uniformly at random and ignores the first move.
- One training run, one seed, no averaging.

## Credits

The Gymnasium environment is adapted from
[aylint/gym-minesweeper](https://github.com/aylint/gym-minesweeper). The lives,
action masking, reward shaping, pygame renderer and the learning pipeline are
mine.
