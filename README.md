# Minesweeper DDQN Agent

A neural network that learns to play Minesweeper by playing it, written in
PyTorch on a Gymnasium environment.

Nobody tells it the rules. It sees the board the way a player does, the
uncovered numbers and nothing else, and it clicks a square. Uncovering a safe
square earns it a small reward, hitting a mine costs it, and across tens of
thousands of games it adjusts itself to collect more reward. What comes out is a
network that takes a board and scores every square by how good a click it looks
like. The method is called a Double Deep Q-Network, and the rest of this file is
how it is put together.

The interesting part is not that it plays Minesweeper. It is that most of what
the agent would need is hidden from it: the mines are never shown, so some
positions cannot be settled by reasoning at all. The set of legal moves also
shrinks as a game goes on, since a square can only be clicked once. Both of
those shape the design below.

## Results

Evaluated over **1000 episodes with a greedy policy** (no exploration), using
`evalDDQN.py` against `model.pth`:

| Configuration | Win rate |
| --- | --- |
| 10x10 board, 9 mines, 3 lives | **91%** |

Read that row carefully before comparing it to anything else. This is **not**
standard Minesweeper:

- 9 mines on 100 cells is 9% density, easier than the classic beginner preset
  (10 mines on 81 cells, 12.3%).
- The agent gets **3 lives**. Hitting a mine costs a life and the episode
  continues. Only the third hit ends it. A single-life win rate would be lower
  and is not what this number measures.
- There is no first-click safety, so the opening move can and does hit mines.

The lives mechanic is deliberate rather than a shortcut. It exists so the
mine-hit penalty can be varied independently of episode termination, which makes
it possible to study how the cost of a mistake changes the agent's willingness
to take an uncertain cell. Setting `life_count=1` recovers standard rules.

## Approach

**Fully convolutional Q-network.** `model.py` is four convolutional layers
(1 -> 128 -> 128 -> 128 -> 1): three 3x3 convolutions with padding 1 and ReLU
activations, then a 1x1 convolution that collapses the 128 channels down to one.
The output is flattened to 100 values, one Q-value per cell. There is no fully
connected layer, so the network is translation-equivariant: a pattern in the
top-left produces the same response in the bottom-right. That matters here
because Minesweeper deductions are local and position-independent, and it means
the network cannot memorize board coordinates.

**Action masking.** The action space is `Discrete(100)`, one action per cell, but
already-revealed cells are illegal. Before every argmax, invalid logits are
filled with `-1e9`, during action selection and inside the target computation.
Without this the agent spends most of training re-clicking known cells.

**Double DQN.** The online network picks the next action, the target network
scores it. This is the standard fix for the maximization bias in vanilla DQN,
which matters more than usual here because most cells look alike early in an
episode and overestimation compounds.

**Prioritized experience replay.** `torchrl.PrioritizedReplayBuffer` with
alpha 0.6, beta 0.9, capacity 25600, batch size 256. Priorities are updated from
the absolute TD error after every gradient step. Most transitions in Minesweeper
are uninformative flood-fill reveals, so uniform sampling wastes updates on
states the network already predicts well.

**Soft target updates.** The target network is blended toward the online network
with tau = 0.01 on every training step, rather than copied on a fixed interval.

### Reward function

| Event | Reward |
| --- | --- |
| Win (all safe cells revealed) | +1 |
| Mine hit, lives remaining | -0.1 |
| Mine hit, no lives remaining | -1 |
| Safe reveal | +0.02 |

The small positive step reward is there to keep the agent opening cells rather
than stalling. It is deliberately much smaller than the mine penalty so that
grinding out safe reveals never outweighs a death.

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

The observation is the visible 10x10 board as a single-channel integer grid:

```
-2   closed
-1   revealed mine
0-8  number of adjacent mines
```

This is passed to the network raw rather than one-hot encoded per cell value.
One-hot encoding is the more common choice in published Minesweeper RL work and
would be the first thing to try for a further accuracy gain.

## Running

Requires `torch`, `torchrl`, `gymnasium`, `numpy`, `pygame`, `matplotlib`, `six`.

```
python doubleDQN.py    # train 10,000 episodes, save to model.pth, plot curves
python evalDDQN.py     # greedy evaluation over 1000 episodes, print win rate
```

`doubleDQN.py` plots win rate, average reward, and loss at the end of training.
`environment.py` supports `render_mode='human'` for a pygame view of the board
and remaining lives, which is how the agent's failure modes were inspected.

## Files

```
model.py          Fully convolutional Q-network
environment.py    Gymnasium Minesweeper env, lives mechanic, action mask, render
doubleDQN.py      Double DQN training loop with prioritized replay
evalDDQN.py       Greedy evaluation over 1000 episodes
model.pth         Trained weights used for the reported result
```

## Caveats

Things a reader should know before taking the 91% at face value:

- No baseline comparison is included yet. The number to beat is a naive
  probability solver on the same 10x10 / 9-mine / 3-life configuration; without
  it, 91% has no context.
- Mine placement in `place_mines` samples uniformly at random with rejection,
  and is independent of the agent's first move, so early-move losses are
  unavoidable rather than a policy failure.
- The reported result is from a single training run. No seed averaging.

## Credits

The Gymnasium environment is adapted from
[aylint/gym-minesweeper](https://github.com/aylint/gym-minesweeper). The lives
mechanic, action masking, reward shaping, pygame renderer, and the entire
learning pipeline are mine.
