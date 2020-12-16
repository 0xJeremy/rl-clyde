# ghostAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random
from util import manhattanDistance
import numpy as np
import random, util, math


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.99, prob_scaredFlee=0.01, **args):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [
            manhattanDistance(pos, pacmanPosition) for pos in newPositions
        ]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [
            action
            for action, distance in zip(legalActions, distancesToPacman)
            if distance == bestScore
        ]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist


class ExperDirectionalGhost(ReinforcementAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(
        self,
        index=1,
        extractor="GhostExtractor",
        numTraining=100,
        numTesting=100,
        epsilon=0.5,
        alpha=0.5,
        gamma=1,
        **args
    ):
        ReinforcementAgent.__init__(self, **args, index=index)
        self.index = index

        self.featExtractor = util.lookup(extractor, globals())()
        self.qvalues = util.Counter()
        self.weights = util.Counter()
        self.filename = "scores/ghost_out.score"
        self.outfile = open(self.filename, "w")
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.numTraining = numTraining + numTesting
        self.numTesting = numTesting

    def __computeValueFromQValues(self, state):
        actions = self.getLegalActions(state, self.index)
        out = -float("inf")
        for a in actions:
            if self.getQValue(state, a) >= out:
                out = self.getQValue(state, a)

        return out if out != (-float("inf")) else 0.0

    def __computeActionFromQValues(self, state):
        actions = self.getLegalActions(state, self.index)
        out = -float("inf")

        for a in actions:
            if self.getQValue(state, a) > out:
                out = self.getQValue(state, a)
                outa = a
        lis = []
        for a in actions:
            if self.getQValue(state, a) == out:
                lis.append(a)
        if out != -float("inf"):
            outa = random.choice(lis)
        else:
            outa = None
        return outa

    def getQValue(self, state, action):
        return self.weights * self.featExtractor.getFeatures(state, action)

    def getValue(self, state):
        return self.__computeValueFromQValues(state)

    def getPolicy(self, state):
        return self.__computeActionFromQValues(state)

    def getAction(self, state):
        legalActions = self.getLegalActions(state, self.index)
        if len(legalActions) == 0:
            return Directions.STOP

        action = (
            random.choice(legalActions)
            if util.flipCoin(self.epsilon)
            else self.__computeActionFromQValues(state)
        )

        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        featureVector = self.featExtractor.getFeatures(state, action)

        maxQFromNextState = self.__computeValueFromQValues(nextState)
        actionQValue = self.getQValue(state, action)

        for feature in featureVector:
            self.weights[feature] += (
                self.alpha
                * (reward + self.discount * maxQFromNextState - actionQValue)
                * featureVector[feature]
            )

    def final(self, state):
        # print("Score: {}".format(state.getScore(self.index)))
        ReinforcementAgent.final(self, state)
        # if self.episodesSoFar == self.numTraining:
        self.outfile.write("{}\n".format(state.getScore(self.index)))

        if self.episodesSoFar == self.numTesting:
            msg = "Training Done (turning off epsilon and alpha). Beginning Testing..."
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning
            self.outfile.write("\n")

        if self.episodesSoFar == self.numTraining:
            msg = "Testing Done."
            print("%s\n%s" % (msg, "-" * len(msg)))
