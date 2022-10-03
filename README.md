# DQN-2048

## ***Introduction***

2048 is a single-player sliding tile puzzle video game written by Italian web developer Gabriele Cirulli and published on GitHub. The objective of the game is to slide numbered tiles on a grid to combine them to create a tile with the number 2048; however, one can continue to play the game after reaching the goal, creating tiles with larger numbers.

![](https://upload.wikimedia.org/wikipedia/commons/f/f9/2048_win.png)

### ***Try out the puzzle: [2048](https://play2048.co/)***

## ***Deep Q-Learning***

Reinforcement is a part of machine learning concerned about the action, which an agent in an environment takes to maximize the rewards. Reinforcement Learning differs from supervised learning and unsupervised learning in the sense that it does not need a supervised input/output pair.

In Deep Q-Learning, the user stores all past experiences in memory and the future action defined by the output of  Q-Network. Thus, Q-network gains the Q-value at state st, and at the same time target network (Neural Network) calculates the Q-value for state St+1 (next state) to make the training stabilized and blocks the abruptly increments in Q-value count by copying it as training data on each iterated Q-value of the Q-network.

Read for details: [Deep Reinforcement Learning: Guide to Deep Q-Learning](https://www.mlq.ai/deep-reinforcement-learning-q-learning/)

## ***Implementation***

This project implements DQN in Tensorlfow/Keras to solve 2048.
The GUI of the game is implemented in PyGame.


