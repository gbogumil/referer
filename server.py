import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from collections import deque
import gym
import math
import collections
import heapq
import base64

class GapLearnerLayer:
    
    def __init__(self, learner):
        self.learners = []
        self.learners.append(learner)

    def add_learner(self, learner):
        pass


class GapLearner:
    
    # a shared view of memories for all learners to draw upon
    memories = Memory(5)
    learners = []

    def __init__(self, env, distance = 1, model=None):
        GapLearner.learners.append(self)
        self.env = env
        self.model = model if model is not None else self.create_model()
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
    def train(cls, observation, past_observation, best_accuracy):
        # review all memories and using differences stored against models which predict that distance
        # update the memory observation ranks during each storage

        # get a list of distances between each memory
        observationdistances = [(o1.age - o2.age, o1, o2) 
                                for o1 in GapLearner.memories.memory 
                                for o2 in GapLearner.memories.memory
                                if o1.age - o2.age > 0]
        # evaluate each memory pair as a training set for each model
        for learner in GapLearner.learners:
            observations = np.array([(o1, o2) for d, o1, o2 in observationdistances if d == learner.distance])
            o1, o2 = map(np.array, zip(*observations))
            learner.model.fit(o1, o2, epochs=5, batch_size=32)

        # build out the network where it needs support
        # and grow it where it is doing well
        for learner in GapLearner.learners:
            if learner.numbness_amount > learner.high_limit:
                learner.set_child()
                learner.distance *= 2
            if learner.numbness_amount < learner.low_limit:
                learner.add_peer()
        GapLearner.memories.prune()

    def predict(self, observation, parentobservation = None):
        past_observation = GapLearner.memories.get_observation(self.distance)
        # the past observation should be evaluated by the current model
        # the accuracy of the model to the current observation indicates the accuracy
        # evaluation happens for each peer,
        # the best peer is followed until a child produces a lower quality result
        # once the child is chosen, it and its parents get a boost in training
        best_peer, (best_accuracy, best_prediction) = self.evaluatePeers(observation, past_observation, parentobservation)

        if best_peer.child is None:
            prediction = best_peer.model.predict(np.concatenate([np.array(observation), np.array(best_prediction)]))
            # might want to consider adding the prediction also.
            GapLearner.memories.add_observation(Observation(observation))
            return prediction
        else:
            return best_peer.child.predict(observation, best_prediction)

    def evaluate(self, observation, prevObservation, parentobservation = None):
        # models except the root model take input 2x observation size
        prediction = self.model.predict(np.array([prevObservation]))
        rank = np.sqrt(np.mean(np.square(prediction - prevObservation)))
        if rank < self.numbing_factor:
            self.numbness_amount -= self.numbing_decrease
        if rank > self.numbing_factor:
            self.numbness_amount += self.numbing_increase
        return rank, prediction

    def evaluatePeers(self, observation, prevObservation):
        accuracies = [(peer, peer.evaluate(observation, prevObservation)) for peer in self.peers]
        best_peer, (best_accuracy, best_prediction) = max(accuracies, key=lambda item: item[1][0])
        return best_peer, (best_accuracy, best_prediction)

    def set_child(self, distance = -1):
        if not self.child is None:
            return
        new_model = self.clone_model()
        
        # Create a new GapLearner with the cloned model
        distance = self.distance if distance == -1 else distance
        self.child = GapLearner(env = self.env, distance = distance, model = new_model)
        self.numbness_amount = (self.low_limit + self.high_limit) / 2.0
    
    def add_peer(self):
        # when adding a new peer
        # Clone the current model architecture
        new_model = self.create_model()
        # Create a new GapLearner with the cloned model
        new_peer = GapLearner(env = self.env, distance = self.distance, model = new_model)
        # Increase numbness to median value between peer limit and set_child limit
        self.peers.append(new_peer)
        self.numbness_amount = (self.low_limit + self.high_limit) / 2.0

    def create_model(self, observation_size = 10):
        model = keras.Sequential([
            # input layer is always 2x the observation size
            # this allows for passing the best parent prediction of the future to the more proximal
            # predictor
            keras.layers.Dense(64, activation='relu', input_shape=(2 * observation_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(observation_size), #second to last is actually the output desired
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

class Memory:
    def __init__(self, depth=5):
        self.depth = 15
        self.current_time = 0
        self.memory = []

    def add_observation(self, observation, rank):
        self.memory.append((observation, self.current_time, rank))

    def get_observation(self, distance):
        bin_heap = []
        target_bin = self.get_bin(distance)
        
        for observation, time_added, position in self.memory:
            if self.get_bin(self.current_time - time_added + position) == target_bin:
                rank = observation.rank
                if len(bin_heap) < self.depth:
                    heapq.heappush(bin_heap, (-rank, observation))
                else:
                    heapq.heappushpop(bin_heap, (-rank, observation))
        # now filter the observation returned
        return max(bin_heap)[1]
    
    def tick(self):
        self.current_time += 1
    
    def prune(self):
        # Create a heap for each bin
        bin_heaps = collections.defaultdict(list)

        # Assign each observation to a bin
        for class_reference, time_added, distance in self.memory:
            current_age = (self.current_time - time_added) + distance
            bin = self.get_bin(current_age)
            if len(bin_heaps[bin]) < self.depth:
                heapq.heappush(bin_heaps[bin], (-current_age, class_reference))
            else:
                heapq.heappushpop(bin_heaps[bin], (-current_age, class_reference))
        
        # Replace our memory with the top n elements of each bin
        self.memory = [(class_ref, time_added, -rank) for bin_heap in bin_heaps.values() for rank, class_ref in bin_heap]
        
class Observation:
    def __init__(self, observation):
        self.observation = observation
        self.age = 0
        # an array of time dependent ranking results
        # this represents the ranking of this observation by time gap learners of different distances
        # slot 0 has usage for distance 1, slot 2 has rank for distance 2, slot 3 for distance 3 & 4
        # slot 4 for distance 5 through 8, slot 6for distance 9 through 16 and so on
        self.rank_over_time = np.array([])
        # calculated when rank_over_time is updated
        # represents the average of the values
        self.rank = 0
    def add_rank(self, distance, rank):
        bin = Memory.get_bin(distance)
        if len(self.rank_over_time) < bin:
            # expand array
            np.append(self.rank_over_time, np.zeros(bin - len(self.rank_over_time)))
            self.rank_over_time[bin] = (1, rank)
        else:
            self.rank_over_time[bin] = self.update_rank(self.rank_over_time[bin], rank)
        self.rank = np.mean(self.rank_over_time[bin])

    def update_rank(self, countandrank, rank):
        newcount = countandrank[0] + 1
        newrank = ((countandrank[0]*countandrank[1])+rank) / newcount
        return (newcount, newrank)

def get_bin(age):
    # Calculate which bin the age should go in
    if age == 0:
        return 0
    return int(math.log2(age))

