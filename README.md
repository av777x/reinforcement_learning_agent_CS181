# Harvard CS181 Practical 4
## Arjun Verma, Philippe Noël, Jorma Görns

I was primarily responsible for implementing the Q-Learning agent that learns to play the SwingyMonkey game, which is similar to the popular smartphone game Flappy Bird. Added multiple layers of thinking over Q-Learning, including a discretisation of the state space, an approximation of gravity, a decaying epsilon-greedy policy and a decaying learning rate based on the number of visits to the state.

Run the model by downloading the repository and running "python stub.py". The agent will learn to play the game itself within a few minutes and achieve extremely high scores. To play the game yourself, run "python SwingyMonkey.py".
