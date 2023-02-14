# Board game santorini using Reinforcement algorithms

## Install with python 3.8

```commandline
pip install -r requirements.txt
```

## Play
### Play with a random agent:

```commandline
python play.py -p2 random
```


### Play with a minimax agent with various depth:

```commandline
python play.py -p2 minimax1 # depth 1
python play.py -p2 minimax2 # depth 2
python play.py -p2 minimax3 # depth 3
```

### Play with a trained model: 
```commandline
python play.py -p2 rl -pt2 zoo/ckpt/TRPO_mixed/lr1.00e-02_run0 -bm init-random
```

### Play with another person:
```commandline
python play.py -p2 human
```

### Watch agents playing:
```commandline
# 1 vs 1
python play.py -p1 rl -p2 minimax3
# 1 vs 1 vs 1
python play.py -p1 minimax3 -p2 minimax3 -p3 minimax3
# 2 vs 2
python play.py -p1 minimax3 -p2 minimax3 -p3 minimax3 -p4 minimax3
```


## References

https://boardgamegeek.com/boardgame/194655/santorini

https://en.wikipedia.org/wiki/Self-play_(reinforcement_learning_technique)