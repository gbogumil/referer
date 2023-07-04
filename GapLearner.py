import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow import keras
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping
from collections import deque
import math
import collections
import heapq
import base64
from Memory import Memory
from Memory import Observation
import logging
import json
from typing import Union
from sklearn.metrics import mean_squared_error

class GapGroup:
    logger = logging.getLogger('gaplogger')
    # the set of peers and learners
    def __init__(self, distance = 1):
        self.peers = []
        self.memories = Memory(5)
        # create a peer set with initial distance specified
        self.peers.append(GapPeers(self, distance))

    def predict(self, observation, parentPrediction = None):
        # make sure the observation is usable
        if isinstance(observation, bytearray):
            GapGroup.logger.info(f'predicting obs len {len(observation)}')
            observation = Observation(observation, self.memories.current_time)
            GapGroup.logger.info(f'created observation {len(observation.observation)}')
        elif not isinstance(observation, Observation):
            GapGroup.logger.error(f"Invalid observation type: {type(observation)}")
            return None

        self.memories.add_observation(observation, 0)
        self.memories.tick()

        # check for old memories so n be followed
        maxpeer = max(self.peers, key=lambda peer: peer.distance)
        return maxpeer.predict(observation, parentPrediction)

    def train(self):
        # review all memories and using differences stored against models which predict that distance
        # update the memory observation ranks during each storage
        GapGroup.logger.info(f'training: memory count {len(self.memories.memory)}')
        # get a list of distances between each memory
        ages = [o[0].age for o in GapGroup.memories.memory]
        GapGroup.logger.info(f'ages {ages}')

        learnerdistances = list(set([learner.distance 
                            for peer in self.peers 
                            for learner in peer.learners]))
        GapGroup.logger.info(f'learner distances {learnerdistances}')
        # get the set of data that might be useful
        observationdistances = [(o1[0].age - o2[0].age, o1, o2) 
                                for o1 in self.memories.memory 
                                for o2 in self.memories.memory
                                if o1[0].age - o2[0].age in learnerdistances]
        GapGroup.logger.info(f'distance matches {len(observationdistances)}')
        distances = [d for (d, o1, o2) in observationdistances]
        GapGroup.logger.info(f'{distances}')
        # evaluate each memory pair as a training set for each model
        GapGroup.logger.info(f'peers {len(self.peers)}')
        for peer in self.peers:
            GapGroup.logger.info(f'fitting peer group {peer.distance}')
            observations = np.array([(o1[0].observation, o2[0].observation) for _, o1, o2 in peer.distance])
            GapGroup.logger.info(f'\nobs {observations}')
            o1, o2 = map(np.array, zip(*observations))
            for learner in peer.learners:
                early_stopping = EarlyStopping(monitor='val_loss', patience=3)
                learner.model.fit(o1, o2, validation_split=0.2, epochs=1500, batch_size=8*peer.distance, callbacks=[early_stopping])

    def stats(self):
        # starts with a set of peers, recursively calls each peer
        statinfo = self._stats_internal()
        GapGroup.logger.info(f'stats: {statinfo}')
        return json.dumps(statinfo, indent=2)

    def _stats_internal(self):
        return {
            'peers count': len(self.peers),
            'observation_size': self.observation_size,
            'memory size': len(self.memories),
            'peerStats': [
                {
                    'distance': peer.distance,
                    'numLearners': len(peer.learners),
                    'childDistance': -1 if not peer.child else peer.child.peers.distance,
                    'learnerStats': [
                        {
                            'rank': learner.rank,
                            'confidence': learner.confidence_amount
                        }
                        for learner in peer.learners
                    ]
                }
                for peer in self.peers
            ]
        }

    def grow(self):
        # build out the network where it needs support
        # and grow it where it is doing well
        for peer in self.peers:
            for learner in peer.learners:
                GapGroup.logger.debug(f'learner confidence {learner.confidence_amount}')
                if learner.confidence_amount > learner.high_limit:
                    result = learner.grow_deeper()
                    if not result[0]:
                        result = learner.grow_taller()
                    self.peers.append(result[1])
                if learner.confidence_amount < learner.low_limit:
                    learner.grow_wider()
    def prune(self):
        pass

class GapPeers:
    # the set of peers in a given distance level
    def __init__(self, group, distance, learner = None):
        self.group = group
        self.distance = distance
        self.learners = []
        if learner == None:
            learner = GapLearner(self)
        self.learners.append(learner)

    def predict(self, observation, parentPrediction = None):
        GapGroup.logger.debug(f'peer predict {observation} {parentPrediction}')
        # if there is no parent observation then try getting one
        past_memory = self.group.memories.get_memory(self.distance)
        # update the observation ranking based on similarity and time
        accuracies = [(learner, learner.evaluate(observation, past_memory[0], parentPrediction)) 
                      for learner in self.learners]

        o1age = self.group.memories.get_age(observation)
        o2age = self.group.memories.get_age(past_memory[0])
        GapGroup.logger.debug(f'age 1 & 2 {o1age} & {o2age}')
        bin = Memory.get_bin(o2age - o1age)
        GapGroup.logger.debug(f'bin {bin}')

        for accuracy in accuracies:
            past_memory[0].rank.add_rank(bin, accuracy[1][0])

        best_peer, (best_accuracy, best_prediction) = max(accuracies, key=lambda item: item[1][0])

        GapGroup.logger.debug(f'best peer child {best_peer.child}')
        if best_peer.child:
            return best_peer.child.peers.predict(observation, parentPrediction = best_prediction)
        else:
            return best_peer._predict_internal(observation, parentPrediction)
    
class GapLearner:
    # GapLearner is a dynamic NN model designed to predict the future
    # at varying distances.
    # it is dynamic in shape as it can grow specialized NNs of varying distance
    # the shape can expand in 3 ways
    # 1. width - when the current model is insufficient to predict the future, 
    # the width is increased
    # this means that an additional model of the same distance is created
    # 2. depth - when the current model is sufficient
    # 3. height - when the current model is sufficient
    # contraction is handled by use and fitness of models
    # seems reasonable that when training
    # 1. the models that aren't right often are removed
    # 2. the remainder is trained
    # 3. the memory is pruned
    
    # a shared view of memories for all learners to draw upon
    logger = logging.getLogger('gaplogger')
    
    def __init__(self, peers, model=None, observation_size=10):
        GapGroup.logger = logging.getLogger('gaplogger')
        self.peers = peers
        self.model = model if model is not None else self.create_model(
            input_size=observation_size*2, output_size=observation_size)
        self.rank = 0
        self.observation_size = observation_size
        # GapPeers reference smaller distance peers
        self.childPeers = None
        # the amount of confidence for this learner
        self.confidence_amount = 0.5
        # the level of accuracy needed to flip confidence effect
        self.confidence_factor = 0.4
        # how fast confidence increases with good predictions
        self.confidence_increase = 0.2
        # how fast confidence decreases with bad predictions
        self.confidence_decrease = 0.1
        # the confidence level needed to birth a child and double the current distance
        self.high_limit = 0.6
        # the confidence lower bound to spawn an additional GapLearner of the same distance
        self.low_limit = 0.2

    def create_model(self, input_size, output_size):
        GapGroup.logger.info(f'create model: {input_size} {output_size}')
        model = keras.Sequential([
            # input layer is always 2x the observation size
            # this allows for passing the best parent prediction of the future to the more proximal
            # predictor
            keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(output_size), #second to last is actually the output desired
        ])

        model.compile(optimizer='adam', loss='mse')

        return model
    
    def clone_model(self):
        # Clone the current model architecture
        new_model = tf.keras.models.clone_model(self.model)
        # Copy the weights from the current model to the new model
        new_model.set_weights(self.model.get_weights())
        new_model.compile(optimizer='adam', loss='mse')
        return new_model
    
    def _predict_internal(self, observation, parentPrediction = None):
        # check the model to see what shape it expects
        # it should either expect the shape of observation
        # or the shape of observation * 2
        # if it expects observation * 2 then parentPrediction
        # will be used and if empty padded to make up the difference

        GapGroup.logger.debug(f'observation: {observation.observation}')
        GapGroup.logger.debug(f'prediction: {parentPrediction}')
        modelinputshape = self.model.layers[0].input_shape[1]
        GapGroup.logger.debug(f'model input layer shape: {modelinputshape}')
        if parentPrediction == None:
            parentPrediction = [[] for _ in range(len(observation.observation))]
        input = np.concatenate((observation.observation, parentPrediction))
        input = np.expand_dims(input, axis=0)
        GapGroup.logger.debug(f'evaluate shape: {input.shape}')
        prediction = self.model.predict(input)
        GapGroup.logger.debug(f'predicted {prediction}')
        return prediction[0]

    def evaluate(self, observation, pastObservation, parentPrediction = None):
        # models except the root model take input 2x observation size
        
        prediction = self._predict_internal(pastObservation, None)
        # see how far off the predictor would be from the previous state
        rank = mean_squared_error(np.squeeze(observation.observation), prediction)
        GapGroup.logger.debug(f'adding rank {rank} {observation.rank}')
        pastObservation.rank.add_rank(
            self.memories.get_bin(self.memories.get_age(observation)), 
            rank)
        GapGroup.logger.debug(f'obs rank {observation.rank}')
        self.rank = rank
        GapGroup.logger.info(f'adusting confidence_amount {self.confidence_amount}')
        confidence_hold = self.confidence_amount
        if rank < self.confidence_factor:
            self.confidence_amount -= self.confidence_decrease
        else:
            self.confidence_amount += self.confidence_increase
        GapGroup.logger.info(f'adjusted confidence_amount by {self.confidence_amount - confidence_hold} from {confidence_hold} to {self.confidence_amount}')

        return rank, prediction[0]
    
    

