# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:19:09 2018

@author: Mika Sipilä
"""
import math
import os
from numpy import *
import sympy
from random import randint

class AI():
    
    def __init__(self, nhneuron, params, ninput, noutput):
        
        params = array(params)
        self.params = params
        self.ninput = ninput
        self.noutput = noutput
        self.hlayer1_weights = sympy.Matrix(self.ninput, nhneuron, params[0:(nhneuron * self.ninput)])

        self.hlayer1_bias = sympy.Matrix(params[len(self.hlayer1_weights): 
            nhneuron + len(self.hlayer1_weights)])
        self.outp_weights = sympy.Matrix(nhneuron, self.noutput, params[len(self.hlayer1_weights) + len(self.hlayer1_bias):
            len(self.hlayer1_weights) + len(self.hlayer1_bias) + self.noutput * nhneuron])
        self.outp_bias = sympy.Matrix(params[len(self.hlayer1_weights) + len(self.hlayer1_bias) + len(self.outp_weights):
            len(self.hlayer1_weights) + len(self.hlayer1_bias) + len(self.outp_weights) + self.noutput])
    
    def output(self, inputvec):
        inpvec = sympy.Matrix(inputvec).T
        hlayer1 = AI.sigmoid(inpvec*self.hlayer1_weights + self.hlayer1_bias.T)
        outp = AI.tanh(hlayer1*self.outp_weights + self.outp_bias.T)
        return int(round(outp[0][0]))
    
    def sigmoid(x1):
        x1 = array(x1)
        return 1/(1 + exp(-x1.astype(float)))
    
    def tanh(x1):
        x1 = array(x1).astype(float)
        return (exp(x1) - exp(-x1))/(exp(x1) + exp(-x1))
        

class GA():
    
    def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            fitness = array(fitness)
            max_fitness_idx = where(fitness == max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -9999
        return parents
    
    def crossover(parents, offspring_size):
        offspring = empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = int(round(random.uniform(low=0, high=offspring_size[1])))
        
        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring
 
    def mutation(offspring_crossover, p):
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            for i in range(offspring_crossover.shape[1]):
                val = random.uniform()    
                if val <= p:
                    random_value = random.uniform(-2.0, 2.0)
                    offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value
        return offspring_crossover