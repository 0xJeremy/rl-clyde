from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
import random, util, math

class ExperimentalAgent(ReinforcementAgent):
    def __init__(self, extractor="SimpleExtractor", **args):
        ReinforcementAgent.__init__(self, **args)
        self.featExtractor = util.lookup(extractor, globals())()
        self.qvalues = util.Counter()
        self.weights = util.Counter()
        self.outfile = open('scores/out.score', 'w')

    def __computeValueFromQValues(self, state):
        actions = self.getLegalActions(state)
        out = -float("inf")
        for a in actions:
            if self.getQValue(state, a) >= out:
                out = self.getQValue(state, a)

        return out if out != (-float("inf")) else 0.0

    def __computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
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
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None

        action = random.choice(legalActions) if util.flipCoin(self.epsilon) else self.__computeActionFromQValues(state)

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
        print("Score: {}".format(state.getScore()))
        ReinforcementAgent.final(self, state)
        # if self.episodesSoFar == self.numTraining:
        self.outfile.write('{}\n'.format(state.getScore()))
        
