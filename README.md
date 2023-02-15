# Board game santorini using reinforcement algorithms
This project applies Reinforcement algorithms to the board game Santorini without Godly powers.
The training uses the minimax agents and self-play technique, and
PPO, A2C, and TRPO are applied to this board game.

## Install with python 3.8

```bash
pip install -r requirements.txt
```

## Play
### Play with a random agent:

```bash
python play.py -p2 random
```


### Play with a minimax agent:

```bash
python play.py -p2 minimax1 # depth 1
python play.py -p2 minimax2 # depth 2
python play.py -p2 minimax3 # depth 3
```

### Play with a trained model: 
```bash
python play.py -p2 rl -pt2 zoo/ckpt/TRPO_mixed/lr1.00e-02_run0 -bm init-random
```

### Play with another person:
```bash
python play.py -p2 human
```

### Watch agents playing:
```bash
# 2 players mode: 1 vs 1
python play.py -p1 rl -p2 minimax3 -pt1 zoo/ckpt/TRPO_mixed/lr1.00e-02_run0 -bm init-random
# 3 players mode: 1 vs 1 vs 1
python play.py -p1 minimax3 -p2 minimax3 -p3 minimax3
# 4 players mode: 2 vs 2
python play.py -p1 minimax3 -p2 minimax3 -p3 minimax3 -p4 minimax3
```


## References

### Santorini
https://boardgamegeek.com/boardgame/194655/santorini

http://files.roxley.com/Santorini-Rulebook-Web-2016.08.14.pdf

https://boardgamearena.com/gamepanel?game=santorini

### Reinforcement Learning

https://en.wikipedia.org/wiki/Self-play_(reinforcement_learning_technique)

https://stable-baselines3.readthedocs.io/en/master/

https://sb3-contrib.readthedocs.io/en/master/
