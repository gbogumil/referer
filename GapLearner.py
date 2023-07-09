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
    def __init__(self, distance = 1, width=10):
        self.peers = []
        self.width = width
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

        self.memories.add_observation(observation)
        self.memories.tick()

        # check for old memories so n be followed
        maxpeer = max(self.peers, key=lambda peer: peer.distance)
        return maxpeer.predict(observation, parentPrediction)

    def train(self):
        # review all memories and using differences stored against models which predict that distance
        # update the memory observation ranks during each storage
        GapGroup.logger.info(f'training: memory count {len(self.memories.memory)}')
        # get a list of distances between each memory
        births = [o.birth for o in self.memories.memory]
        GapGroup.logger.info(f'births {births}')

        learnerdistances = list(set([peer.distance 
                            for peer in self.peers]))
        GapGroup.logger.info(f'learner distances {learnerdistances}')
        # get the set of data that might be useful
        observationdistances = [(o1.birth - o2.birth, o1, o2) 
                                for o1 in self.memories.memory 
                                for o2 in self.memories.memory
                                if o1.birth - o2.birth in learnerdistances]
        GapGroup.logger.info(f'distance matches {len(observationdistances)}')
        distances = list(set([d for (d, o1, o2) in observationdistances]))
        GapGroup.logger.info(f'{distances}')
        # evaluate each memory pair as a training set for each model
        GapGroup.logger.info(f'peers {len(self.peers)}')
        for peer in self.peers:
            GapGroup.logger.info(f'fitting peer group {peer.distance}')
            training_data = []
            labels = []
            for _, o1, o2 in observationdistances:
                zeros = np.zeros_like(o1.observation)
                training_data.append(np.concatenate((o1.observation, zeros)))
                labels.append(o1.observation)
            training_data = np.array(training_data)
            labels = np.array(labels)
            GapGroup.logger.info(f'\nobs {training_data}')
            for learner in peer.learners:
                early_stopping = EarlyStopping(monitor='val_loss', patience=3)
                learner.model.fit(training_data, labels, validation_split=0.2, epochs=1500, batch_size=8*peer.distance, callbacks=[early_stopping])

    def evolve(self):
        # all this means is to examine the current structure and adjust
        # loop through each learner in the each peer group
        # if confidence is low then
        # if there is not a child add a child
        # if there is a child add a peer
        # if confidence is high then
        # if there is already a parent then increase the parent confidence
        # else add a parent
        sortedpeers = sorted(self.peers, key=lambda peer:peer.distance)
        for peer in sortedpeers:
            newchildren = []
            newparents = []
            newlearners = []
            for learner in peer.learners:
                newpeer = None
                newpeerlearner = None
                newparent = None
                if self.shouldaddpeerlearner(learner):
                    newpeerlearner = self.create_peerlearner(learner)
                if self.shouldaddchild(learner):
                    newpeer = self.create_peer(peer.distance/2)
                if self.shouldaddparent(learner):
                    newparent = self.create_peer(peer.distance*2)
                if self.shouldadjustconfidence(learner):
                    self.adjustconfidence(learner)
                if newpeerlearner:
                    peer.learners.append(newpeerlearner)
                    newlearners.append(newpeerlearner)
                if newpeer:
                    learner.childPeers = newpeer
                    self.peers.append(newpeer)
                    newchildren.append(newpeer)
                if newparent:
                    self.peers.append(newparent)
                    newparents.append(newparent)

    def shouldaddchild(self, learner):
        return (learner.peers.distance > 1 and
                learner.confidence_amount < learner.low_limit and
                learner.childPeers is None)
    
    def create_peer(self, distance):
        return GapPeers(self, distance)
    
    def create_parent(self, learner):
        peer = self.create_peer(learner)
    
    def shouldaddpeerlearner(self, learner):
        return (learner.confidence_amount < learner.low_limit and
                learner.childPeers is not None)
    
    def create_peerlearner(self, learner):
        learner = GapLearner(learner.peers)
        return learner
    
    def shouldaddparent(self, learner):
        return learner.confidence_amount > learner.high_limit
    
    def shouldadjustconfidence(self, learner):
        return (self.shouldaddchild(learner) or
                self.shouldaddchild(learner) or
                self.shouldaddpeerlearner(learner))

    def adjustconfidence(self, learner):
        learner.confidence_amount = (learner.low_limit + learner.high_limit) / 2

    def prune(self):
        self.memories.prune()

    def stats(self):
        # starts with a set of peers, recursively calls each peer
        statinfo = self._stats_internal()
        GapGroup.logger.info(f'stats: {statinfo}')
        return statinfo

    def _stats_internal(self):
        return {
            'peerLen': len(self.peers),
            'peerStats': [
                self._stats_peer(peer)
                for peer in self.peers],
            'memoryStats': self._stats_memory(self.memories)
        }
    
    def _stats_memory(self, memory):
        return {
            'depth': memory.depth,
            'current_time': memory.current_time,
            'memorylen': len(memory.memory),
            'memoryinfo': [
                {
                    'memorypos': i,
                    'birth': m.birth,
                    'rank': m.rank.rank
                } for i, m in enumerate(memory.memory)
            ],
            'binlen': len(memory.memory_bins),
            'bininfo': [
                {
                    'binpos': i,
                    'binlen': len(bin),
                    'binObservations': [
                        {
                            # 'observation': o.observation,
                            'birth': o.birth,
                            'rank': o.rank.rank
                        } for o in bin
                    ]
                } for i, bin in enumerate(memory.memory_bins)
            ]
    }
    
    
    def _stats_peer(self, peer):
        if not peer:
            return None
        return {
            'distance': peer.distance,
            'numLearners': len(peer.learners),
            'learnerStats': [
                self._stats_learner(learner) for learner in peer.learners
            ]
        }
            
    def _stats_learner(self, learner):
        return {
            'rank': learner.rank,
            'confidence': learner.confidence_amount,
            'learnerChildPeers': self._stats_peer(learner.childPeers)
        }

class GapPeers:
    # the set of peers in a given distance level
    def __init__(self, group, distance, learner = None):
        self.group = group
        self.distance = distance
        self.learners = []
        if learner == None:
            learner = GapLearner(self, observation_size=self.group.width)
        self.learners.append(learner)

     def predict(self, observation, parentPrediction = None):
        GapGroup.logger.debug(f'peer predict {observation} {parentPrediction}')
        past_memory = self.group.memories.get_memory(self.distance)
        if past_memory == None:
            # we can't evaluate, so just sort on rank and evaluate the max
            GapGroup.logger.debug(f'no memories - {len(self.learners)}')
            best_peer = max(self.learners, key=lambda l: l.rank)
            best_prediction = best_peer._predict_internal(observation, parentPrediction)
        else:
            # update the observation ranking based on similarity and time
            accuracies = [(learner, learner.evaluate(observation, past_memory, parentPrediction)) 
                        for learner in self.learners]

            o1age = self.group.memories.get_age(observation.birth)
            o2age = self.group.memories.get_age(past_memory.birth)
            GapGroup.logger.debug(f'age 1 & 2 {o1age} & {o2age}')
            bin = Memory.get_bin(o2age - o1age)
            GapGroup.logger.debug(f'bin {bin}')

            for accuracy in accuracies:
                past_memory.rank.add_rank(bin, accuracy[1][0])

            best_peer, (_, best_prediction) = max(accuracies, key=lambda accuracy: accuracy[1][0])

        GapGroup.logger.debug(f'best peer child {best_peer.childPeers}')
        if best_peer.childPeers:
            return best_peer.childPeers.predict(observation, parentPrediction = best_prediction)
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
        self.rank = 0
        self.peers = peers
        self.observation_size = observation_size
        self.model = model if model is not None else self.create_model(
            input_size=observation_size*2, output_size=observation_size)
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
            parentPrediction = [0.0 for _ in range(len(observation.observation))]
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
        memories = self.peers.group.memories
        pastObservation.rank.add_rank(memories.get_bin(memories.get_age(observation.birth)), rank)
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
    
    

