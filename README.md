# Harvard CS181 Practical 4
## Arjun Verma, in collaboration with Philippe Noël, Jorma Görns

I was primarily responsible for implementing the Q-Learning agent that learns to play the SwingyMonkey game, which is similar to the popular smartphone game Flappy Bird. The Q-Learning is optimised by a discretisation of the state space, an approximation of gravity, a decaying epsilon-greedy policy and a decaying learning rate based on the number of visits to the state.

The model can be run by downloading the repository and running "python stub.py". The agent will learn to play the game itself within a few minutes and achieve extremely high scores. To play the game yourself, run "python SwingyMonkey.py".
