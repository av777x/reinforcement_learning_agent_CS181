# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
from math import exp
from SwingyMonkey import SwingyMonkey

width = 600 # given in pdf
height = 400 # given in pdf
sizeOfBin = 50 # An arbitrary number chosen so that there are not too many bins
w_div_bin = int(width / sizeOfBin) # gives the number of bins for width of size sizeOfBin
h_div_bin = int(height / sizeOfBin) # gives the number of bins for height of size sizeOfBin
discount = 0.9 # The discount on future values
epsilon = 0.05 # initializing epsilon (decays later)

class Learner(object):
    '''
    This agent jumps following Q-Learning with decaying epsilon-greedy policy.
    '''
    def __init__(self):
        self.counter = 0
        self.bestScore = 0
        self.lastState  = None
        self.lastAction = None
        self.lastReward = None
        self.gravity = 1
        self.vel0 = 0

        # We will be considering: 2 actions (Up or Down), a screen of width 600 divided into
        # bins of chosen size, a screen of height 400 divided into bins of chosen size,
        # velocity (which we are categorizing into 5 bins based on best performance), gravity (1 or 4).
        self.QTable = np.zeros((2, w_div_bin, h_div_bin, 5, 2)) # initialize QTable with 0's
        self.stateVisits = np.zeros((2, w_div_bin, h_div_bin, 5, 2)) # realised we wanted to
        # keep track of the number of times a state has been visited to decay the
        # learningRate and epsilon accordingly. Hence, initialize this array.

    def reset(self):
        self.lastState  = None # when starting game, lastState is None
        self.lastAction = None # when starting game, there is no lastAction, hence None
        self.lastReward = None # when starting game, there is no lastReward, hence None
        self.counter = 0 # resetting counter for new epoch
        self.gravity = 1

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        if self.counter == 0:
            self.vel0 = state['monkey']['vel']
        if self.counter == 1:
            if (self.vel0 - state['monkey']['vel']) % 4 == 0:
                self.gravity = 0
            else:
                self.gravity = 1

        self.counter += 1 # a counter counting number of actions in an epoch

        distToTree = int(state['tree']['dist'] / sizeOfBin) # binned distance to next tree for current state
        headGap = int((state['tree']['top'] - state['monkey']['top']) / sizeOfBin) # binned gap between monkey's top and tree top for current state
        vel = int(state['monkey']['vel'] / 20)
        # By exploration, we had discovered that the velocity appears to range roughly
        # between -50 and around +30. We explored different number of bins and as mentioned
        # above, decided to go with 5, where we divide the velocity by 20 to bin it.

        # all the cases where distToTree is less than 0 mean the same to us. We
        # really care about the positive values of distToTree. Hence we simply
        # the negative values into one category
        if distToTree < 0:
            distToTree = 0

        if npr.rand() < 0.5:
            nextAction = 1
        else:
            nextAction = 0

        # if there was a lastAction i.e. it was not the beginning of the game
        if self.lastAction is not None:
            # relevant values from previous state
            prevDistToTree = int(self.lastState['tree']['dist'] / sizeOfBin)
            prevHeadGap = int((self.lastState['tree']['top'] - self.lastState['monkey']['top']) / sizeOfBin)
            prevVel = int(self.lastState['monkey']['vel'] / 20)

            bestActionQVal = np.max(self.QTable[:, distToTree, headGap, vel, self.gravity])
            if self.QTable[1][distToTree, headGap, vel, self.gravity] > self.QTable[0][distToTree, headGap, vel, self.gravity]:
                nextAction = 1
            else:
                nextAction = 0

            # for decaying epsilon: reduce chance of choosing random action
            # from states that have already been visited because the max actions
            # from them are likely already suitable
            if self.stateVisits[nextAction][distToTree, headGap, vel, self.gravity] > 0:
                eps = epsilon / self.stateVisits[nextAction][distToTree, headGap, vel, self.gravity]
            else:
                eps = epsilon
            if (npr.rand() < eps):
                if npr.rand() < 0.5:
                    nextAction = 1
                else:
                    nextAction = 0

            # first tried a fixed learning rate 0.2, 0.4 and so on, all gave bad results
            # second tried learningRate = 0.5 * exp(-self.counter/500)
            # realised that we want it to be signed so that the update can be signed. Also
            # wanted to
            # Then created a decaying learning rate, so that the learning rate for
            # a state is inversely proportional to the number of times it has
            # been passed. The reasoning for this is that if a state has already been
            # explored, then the Q Values already represent this and we do not
            # want to change them significantly.
            # It is also signed.
            learningRate = 1 / self.stateVisits[self.lastAction][prevDistToTree, prevHeadGap, prevVel, self.gravity]
            # performing the Q-Value Update using the obtained values above
            self.QTable[self.lastAction][prevDistToTree, prevHeadGap, prevVel, self.gravity] += learningRate * (self.lastReward + discount * bestActionQVal - self.QTable[self.lastAction][prevDistToTree, prevHeadGap, prevVel, self.gravity])

        # making the next action to be taken the determined action
        self.lastAction = nextAction
        # setting the current state to the last state as we move onto the next
        self.lastState = state
        # updating the count of states visited
        self.stateVisits[nextAction][distToTree, headGap, vel, self.gravity] += 1
        return nextAction

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.lastReward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # storing the highest score in the learning
        if  learner.bestScore < learner.lastState['score']:
            learner.bestScore = learner.lastState['score']

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    print ("best score:", learner.bestScore)
    print ("mean score:", np.mean(hist))

    pg.quit()
    return

if __name__ == '__main__':
    # multiple gammas and epsilon values for grid search hyperparameter tuning
    # gammas = [0.9, 0.8, 0.6, 0.4, 0.2]
    # epsilons = [0.2, 0.1, 0.05, 0.01, 0.001]
    # for gamma1 in gammas:
    #    for epsilon1 in epsilons:
            # params for this iteration
    #        discount = gamma1
    #        epsilon = epsilon1
    print("Params: " + str(discount) + ", " + str(epsilon))
	# Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 500, 1) # 500 iterations
    # print line for clarity in visualizing
    print("-------------")

    # Save history.
    np.save('hist', np.array(hist))
