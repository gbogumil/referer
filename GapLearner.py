import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow import keras
from tensorflow.keras.layers import Dense
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

class GapLearner:
    
    # a shared view of memories for all learners to draw upon
    memories = Memory(5)
    training_counter = 10
    training_counter_reset = training_counter
    learners = [] # the total collection of learners in the learning tree
    logger = logging.getLogger('gaplogger')
    
    def __init__(self, distance = 1, model=None, observation_size=10):
        self.logger = logging.getLogger('gaplogger')
        GapLearner.learners.append(self)
        self.model = model if model is not None else self.create_model(input_size=observation_size, output_size=observation_size)
        self.rank = 0
        self.observation_size = observation_size
        # the distance of future predictions to predict toward
        self.distance = distance
        # the parent of the current layer - only one node is None
        self.parent = None
        # the child of the current learner
        self.child = None
        # the maximum peers that can be associated to this learner
        self.maxpeers = 4 # arbitrary.  should be dynamic somehow
        # the list of peer learners of the same distance
        # # and the resulting observation is passed back as the result to the outer predict function
        self.peers = []
        self.peers.append(self)
        # the amount of numbness for this learner
        self.numbness_amount = 0.3
        # the level of accuracy needed to flip numbness effect
        self.numbing_factor = 0.8
        # how fast numbing increases with good predictions
        self.numbing_increase = 0.01
        # how fast numbing decreases with bad predictions
        self.numbing_decrease = 0.001
        # the numbness level needed to birth a child and double the current distance
        self.high_limit = 0.99
        # the numbness lower bound to spawn an additional GapLearner of the same distance
        self.low_limit = 0.2

    @classmethod
    def train(cls):
        # review all memories and using differences stored against models which predict that distance
        # update the memory observation ranks during each storage
        GapLearner.logger.debug(f'memory count {len(GapLearner.memories.memory)}')
        # get a list of distances between each memory
        observationdistances = [(o1[0].age - o2[0].age, o1, o2) 
                                for o1 in GapLearner.memories.memory 
                                for o2 in GapLearner.memories.memory
                                if o1[0].age - o2[0].age > 0]
        GapLearner.logger.debug(f'distance matches {len(observationdistances)}')
        # evaluate each memory pair as a training set for each model
        GapLearner.logger.debug(f'learners {len(GapLearner.learners)}')
        for learner in GapLearner.learners:
            observations = np.array([(o1[0].observation, o2[0].observation) for d, o1, o2 in observationdistances 
                                     if d == learner.distance])
            o1, o2 = map(np.array, zip(*observations))
            GapLearner.logger.debug(f'fitting {len(o1)}, {len(o2)}')
            learner.model.fit(o1, o2, epochs=5, batch_size=32)

        # build out the network where it needs support
        # and grow it where it is doing well
        for learner in GapLearner.learners:
            GapLearner.logger.debug(f'learner numbness {learner.numbness_amount}')
            if learner.numbness_amount > learner.high_limit:
                learner.set_child()
                learner.distance *= 2
            if learner.numbness_amount < learner.low_limit:
                learner.add_peer()
        GapLearner.memories.prune()

    def predict(self, observation, parentObservation = None, addObservation = False):
        if isinstance(observation, bytearray):
            observation = Observation(observation, self.memories.current_time)
        elif not isinstance(observation, Observation):
            self.logger.error(f"Invalid observation type: {type(observation)}")
            return None

        if addObservation:
            GapLearner.memories.tick()

            if GapLearner.training_counter <= 0:
                self.logger.info(f'training {GapLearner.memories.current_time}')
                GapLearner.train()
                GapLearner.training_counter = GapLearner.training_counter_reset
                self.logger.info('training complete')
            else:
                GapLearner.training_counter -= 1
                self.logger.debug(f'training_counter {GapLearner.training_counter}')
                
            self.logger.debug(f'adding observation {len(GapLearner.memories)}')
            GapLearner.memories.add_observation(observation, self.memories.current_time)

        
        past_observation = GapLearner.memories.get_observation(self.distance)
        self.logger.debug(f'distance {self.distance} found {past_observation}')
        # the past observation should be evaluated by the current model
        # the accuracy of the model to the current observation indicates the accuracy
        # evaluation happens for each peer,
        # the best peer is followed until a child produces a lower quality result
        # once the child is chosen, it and its parents get a boost in training
        if past_observation is None:
            if self.child is None:
                return None
            self.logger.debug(f'predicting with child {observation} {parentObservation}')
            return self.child.predict(observation, parentObservation)
        self.logger.debug('evaluating peers')
        best_peer, (best_accuracy, best_prediction) = self.evaluatePeers(observation, past_observation[0], parentObservation)

        self.logger.debug(f'best peer child {best_peer.child}')
        if best_peer.child is None:
            self.logger.debug(f'observation: {observation.observation}')
            self.logger.debug(f'prediction: {best_prediction}')
            modelinputshape = best_peer.model.layers[0].input_shape[1]
            self.logger.debug(f'model input layer shape: {modelinputshape}')
            if len(observation.observation) == modelinputshape:
                input = np.expand_dims(observation.observation, axis=0)
            elif len(observation.observation) + len(best_prediction) == modelinputshape:
                input = np.concatenate((observation.observation, best_prediction))
                input = np.expand_dims(input, axis=0)
            else:
                raise ValueError(f'input shape does not match {modelinputshape} - {len(observation.observation)} - {len(best_prediction)}')
            prediction = best_peer.model.predict(input)
            self.logger.debug(f'evaluate shape: {input.shape}')
            prediction = best_peer.model.predict(input)
            # might want to consider adding the prediction also.
            return prediction
        else:
            return best_peer.child.predict(observation, best_prediction)

    def evaluatePeers(self, observation, prevObservation, parentObservation = None):
        # update the observation ranking to inform it how it works for each age range
        accuracies = [(peer, peer.evaluate(observation, prevObservation, parentObservation)) 
                      for peer in self.peers]

        o1age = GapLearner.memories.get_age(observation)
        o2age = GapLearner.memories.get_age(prevObservation)
        self.logger.debug(f'age 1 & 2 {o1age} & {o2age}')
        bin = Memory.get_bin(o1age - o2age)
        self.logger.debug(f'bin {bin}')

        for accuracy in accuracies:
            prevObservation.rank.add_rank(bin, accuracy[1][0])

        best_peer, (best_accuracy, best_prediction) = max(accuracies, key=lambda item: item[1][0])
        return best_peer, (best_accuracy, best_prediction)

    def evaluate(self, observation, prevObservation, parentObservation = None):
        # models except the root model take input 2x observation size
        prevFloats = prevObservation.observation
        parentFloats = parentObservation.observation if parentObservation else []
        input = np.concatenate((prevFloats, parentFloats))
        self.logger.debug(f'evaluate shape: {input.shape}')
        self.logger.debug(f'model input layer shape: {self.model.layers[0].input_shape[1]}')
        self.logger.debug(f'{input[np.newaxis, :]}')
        prediction = self.model.predict(input[np.newaxis, :])
        rank = mean_squared_error(np.squeeze(prediction), prevObservation.observation)
        self.rank = rank
        self.logger.debug(f'adusting numbness_amount {self.numbness_amount}')
        if rank < self.numbing_factor:
            self.numbness_amount -= self.numbing_decrease
        elif rank > self.numbing_factor:
            self.numbness_amount += self.numbing_increase
        self.logger.debug(f'adjusted numbness_amount {self.numbness_amount}')

        return rank, prediction[0]

    def set_child(self, distance = -1):
        if not self.child is None:
            return
        new_model = self.clone_model()
        
        # Create a new GapLearner with the cloned model
        distance = self.distance if distance == -1 else distance
        self.logger.info(f'creating child peers, distance, child distance ({len(self.peers)}, {self.distance}, {distance})')
        self.child = GapLearner(distance = distance, model = new_model)
        self.numbness_amount = (self.low_limit + self.high_limit) / 2.0
    
    def add_peer(self):
        # when adding a new peer
        # Clone the current model architecture
        self.logger.info(f'creating peer of peer, distance ({len(self.peers)}, {self.distance})')
        new_model = self.create_model(self.model, input_size=self.observation_size, output_size=self.observation_size)
        # Create a new GapLearner with the cloned model
        new_peer = GapLearner(distance = self.distance, model = new_model, observation_size=self.observation_size)
        # Increase numbness to median value between peer limit and set_child limit
        self.peers.append(new_peer)
        self.numbness_amount = (self.low_limit + self.high_limit) / 2.0

    def create_model(self, input_size = 10, output_size=10):
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
    
    def stats(self):
        # starts with a set of peers, recursively calls each peer
        statinfo = self._stats_internal()
        self.logger.info(f'stats: {statinfo}')
        return json.dumps(statinfo, indent=2)

    def _stats_internal(self):
        return {
            'numPeers': len(self.peers),
            'peerStats': [
                {
                    'numbnessAmount': peer.numbness_amount,
                    'rank': "{:.2f}".format(peer.rank),
                    'childStats': None if not peer.child else peer.child._stats_internal() 
                }
                for peer in self.peers
            ]
        }
    