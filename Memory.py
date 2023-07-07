import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from collections import deque
import math
import collections
import heapq
import base64
import logging


class Memory:
    def __init__(self, depth=5):
        self.logger = logging.getLogger('gaplogger')
        self.depth = 15
        self.current_time = 0
        self.memory = []
        self.memory_bins = []
        self.get_memory = self.get_memory_preprune

    def __len__(self):
        return len(self.memory)

    def add_observation(self, observation, rank):
        self.logger.debug(f'adding observation \nmemory length: {len(self.memory)} cur time: ({self.current_time}) ({rank})\nobs len ({len(observation.observation)})')
        self.memory.append((observation, self.current_time, rank))
        observation.age = self.current_time

    def get_memory_preprune(self, distance):
        if len(self.memory) == 0:
            return None
        if not self.memory_bins or len(self.memory_bins) == 0:
            self.prune()
            self.get_memory = self.get_memory_postprune
            return self.get_memory(distance)

    def get_memory_postprune(self, distance):
        target_bin = Memory.get_bin(distance)
        self.logger.debug(f'distance {distance} target_bin {target_bin} bins {len(self.memory_bins)}')
        self.logger.debug(f'cur time {self.current_time} memory 0 age {self.memory[0][0].age}')
        # if there is a binned memory then get the youngest one in the 0th position
        # and subtract that from current time to compare to distance
        if self.current_time - self.memory[0][0].age > distance :
            self.logger.debug('checking for memory matches')
            matches = [memory for memory in self.memory if self.memorylistfilter(memory, distance)]
        else:
            self.logger.debug('checking for bin matches')
            # search the bins based on the closes power of 2 distance
            bins = self.get_bins(distance)
            matches = [memory for memory in self.memory_bins[bins[0]] + self.memory_bins[bins[1]]
                       if abs(distance - (self.current_time - memory[1])) < 1e-7]
        self.logger.debug(f'found {len(matches)} matches')
        if len(matches) == 0:
            return None
        # get max memory based on rank which is the first element of the tripple tuple returned
        # of rank, position, obseration
        return max(matches, key=lambda memory: memory[2])
    
    def tick(self):
        self.current_time += 1
        self.logger.info(f'ticked {self.current_time}')
    
    def prune(self):
        self.logger.info(f'pruning {len(self.memory)}')
        ages = [memory[0].age for memory in self.memory]
        self.logger.info(f'ages ({self.current_time}) [{ages}]')
        # Create a heap for each bin
        if len(self.memory) < 1:
            return
        # bin_count represents the power 2 is raised to to get the maximum age
        # the maximum age is the first element in self.memory's age subtracted from current_time
        # then the ceil log of that value is taken
        max_age = Memory.get_bin(self.current_time - self.memory[0][1])
        bin_count = 1 if max_age <= 0 else math.ceil(math.log2(max_age))
        self.logger.info(f'max_age {max_age}\nbin_count {bin_count}')

        bin_heaps = [[] for _ in range(bin_count)]
        self.logger.debug(f'prune: creating bins {bin_count}')

        # rebase the memories and reset current_time
        for memory in self.memory:
            memory[0].age += self.current_time
        self.current_time = 0
        ages = [memory[0].age for memory in self.memory]
        self.logger.info(f'ages ({self.current_time}) [{ages}]')

        # Assign each observation to a bin
        for i, (observation, time_added, rank) in enumerate(self.memory):
            bin = Memory.get_bin(observation.age)
            while bin >= len(bin_heaps):
                bin_heaps.append([])
            if len(bin_heaps[bin]) < self.depth:
                heapq.heappush(bin_heaps[bin], (observation.rank.rank, observation.age, i, observation))
            else:
                heapq.heappushpop(bin_heaps[bin], (observation.rank.rank, observation.age, i, observation))
        # reframe the heaps in terms of the memories so they're all the same when matching
        bin_heaps = [sorted([(observation, observation.age, rank) 
                      for rank, age, pos, observation in bin_heap], key=lambda key:key[1])
                      for bin_heap in bin_heaps]
        self.logger.info(f'new heaps {len(bin_heaps)}')
        # Replace our memory with the top n elements of each bin
        self.memory = [(observation, observation.age, observation.rank.rank) 
                       for bin_heap in bin_heaps 
                       for observation, age, rank in bin_heap]
        self.logger.info(f'memories {len(self.memory)}')
        
        self.memory_bins = bin_heaps
        self.logger.info(f'{len(self.memory)}')
        self.logger.info(f'{len(self.memory[-1])}')
        self.newest_memory_time = self.memory[-1][1]
        self.oldest_memory_time = self.memory[0][1]
        self.logger.info(f'oldest & newest {self.oldest_memory_time} & {self.newest_memory_time}')

    
    def get_bins(self, distance):
        if distance <= 0:
            return (0, 0)
        target_bin = math.log2(distance)
        return (math.floor(target_bin), math.ceil(target_bin))
    
    def get_age(self, observation):
        return observation.age + self.current_time
    
    def memorylistfilter(self, memory, distance):
        memoryage = memory[0].age + self.current_time
        #self.logger.debug(f'memorylistfilter: {memory[0].age} ~ {distance}')
        return memory[0].age == distance
    
    @classmethod
    def get_bin(cls, age):
        # Calculate which bin the age should go in
        if age <= 0:
            return 0
        return int(math.log2(age))
    

    
class Observation:
    def __init__(self, o, age = 0):
        
        self.observation = self.cleanObservation(o)
        self.age = age
        self.rank = ObservationRank()

    def cleanObservation(self, o, nan_substitute=float('inf')):
        floats = np.nan_to_num(o, nan=nan_substitute)
        # this is where the observation might be sized propertly.. 
        # but can be handed by the user of memory so it's not fixed
        return floats


class ObservationRank:
    def __init__(self):
        # an array of time dependent ranking results
        # this represents the ranking of this observation by time gap learners of different distances
        # slot 0 has usage for distance 1, slot 2 has rank for distance 2, slot 3 for distance 3 & 4
        # slot 4 for distance 5 through 8, slot 6 for distance 9 through 16 and so on
        self.rank_over_bin = [ExponentialMovingAverage() for _ in range(64)]
        # calculated when rank_over_bin is updated
        # represents the average of the values
        self.rank = 0

    def add_rank(self, bin, rank):
        self.rank_over_bin[bin].update(rank)
        self.rank = np.mean([ranking.average if ranking.average is not None else 0 
                             for ranking in self.rank_over_bin])

    def update_rank(self, countandrank, rank):
        newcount = countandrank[0] + 1
        newrank = ((countandrank[0]*countandrank[1])+rank) / newcount
        return (newcount, newrank)

class ExponentialMovingAverage:
    def __init__(self, smoothing_factor=0.1):
        self.smoothing_factor = smoothing_factor
        self.average = None

    def update(self, value):
        if self.average is None:
            self.average = value
        else:
            self.average = (1 - self.smoothing_factor) * self.average + self.smoothing_factor * value

    def get_average(self):
        return self.average
