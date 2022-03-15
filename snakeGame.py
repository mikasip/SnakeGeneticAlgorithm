# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 13:13:45 2018

@author: Mika Sipilä
"""

import math
import os
from numpy import *
import sympy
from random import randint
from collections import deque

import pygame
from pygame.locals import *
from AI import *
import copy

FPS = 8
BLOCK_SIZE = 20
MAP_SIZE = 20
MAP_WIDTH = MAP_SIZE*BLOCK_SIZE
MAX_MOVES = 100
INITIAL_WORM_SIZE = 4

class Population():
    
    def __init__(self, pop, nhneuron, ninput, noutput, worm_size = 4, worm_pos_random = False):
        self.dead = 0
        self.worms = deque()
        self.best = None
        if random:
            for i in range(pop.shape[0]):
                self.worms.append(Worm(
                        neurons = pop[i,], 
                        nhneuron = nhneuron, ninput = ninput, 
                        noutput = noutput, size = worm_size, random_pos = worm_pos_random))
            self.best = self.worms[0]
    
    
    
        

class Worm():
    
    def __init__(self, neurons, nhneuron, ninput, noutput, size = 4, random_pos = False, state=array([0,1]), enemy=False):
        self.moves = 0
        self.dead = False
        self.score = 0
        self.state = state
        self.pieces = deque()
        self.enemy = False
        self.neurons = neurons
        if not random_pos:
            if not enemy:
                self.start_pos = array([int(floor(MAP_SIZE)/2), int(floor(MAP_SIZE)/2) + 1])
            else: 
                self.start_pos = array([int(floor(MAP_SIZE)/2), 1])
        else:
            pos_x = floor(random.uniform(low=0, high=20))
            pos_y = floor(random.uniform(low=0, high=20))
            self.start_pos = array([int(pos_x), int(pos_y)])
        self.brain = WormBrain(nhneuron, neurons, ninput, noutput)
        self.treat = Treat()
        self.pieces.append(WormPiece(array(self.start_pos), first=True))
        last_pos = self.start_pos
        for i in range(1,size):
            if not random_pos:
                self.pieces.append(WormPiece(array(self.start_pos - (state*i))))
            else:
                pos_x = last_pos[0]
                pos_y = last_pos[1]
                int1 = floor(random.uniform(low=0, high=2))
                int2 = floor(random.uniform(low=0, high=2))
                int3 = floor(random.uniform(low=0, high=2))
                if int1 == 0:
                    if int2 == 0 and last_pos[0] - 1 >= 0:
                        pos_x = last_pos[0] - 1
                    elif last_pos[0] + 1 < MAP_SIZE:
                        pos_x = last_pos[0] + 1
                if pos_x == last_pos[0]:
                    if int3 == 0 and last_pos[1] - 1 >= 0:
                        pos_y = last_pos[1] - 1
                    elif last_pos[1] + 1 < MAP_SIZE:
                        pos_y = last_pos[1] + 1
                    if pos_y == last_pos[1]:
                        break
                self.pieces.append(WormPiece(array([pos_x, pos_y])))
                last_pos = array([pos_x, pos_y])
                
                    
                
    def update(self, state, eat = False):
        self.moves += 1
        self.state = state
        last_piece_coor = copy.copy(self.pieces[-1].place)
        last_piece_color = copy.copy(self.pieces[-1].color)
        prev_place = copy.copy(self.pieces[0].place)
        self.pieces[0].place += array(self.state) 
        for i in range(1,len(self.pieces)):
            t_place = copy.copy(self.pieces[i].place)
            self.pieces[i].place = prev_place
            prev_place = copy.copy(t_place)
        if eat:
            self.moves = 0
            self.pieces.append(WormPiece(place = last_piece_coor, color = last_piece_color))
    
    def set_enemy(self, enemy):
        self.enemy = enemy
        
    def get_input(self):
      
        f1 = 0
        l1 = 0
        r1 = 0
        
        if self.state[0] == 0 and self.state[1] == 1:
            if self.hit([0,1]):
                f1 = 1
            #f = y2/(MAP_SIZE - 1)
            if self.hit([1,0]):
                l1 = 1
            #l = x1/(MAP_SIZE - 1)
            if self.hit([-1,0]):
                r1 = 1
            #r = x2/(MAP_SIZE - 1)
        if self.state[0] == 0 and self.state[1] == -1:
            if self.hit([0,-1]):
                f1 = 1
            #f = y1/(MAP_SIZE - 1)
            if self.hit([-1,0]):
                l1 = 1
            #l = x2/(MAP_SIZE - 1)
            if self.hit([1,0]):
                r1 = 1
            #r = x1/(MAP_SIZE - 1)
        if self.state[0] == 1 and self.state[1] == 0:
            if self.hit([1,0]):
                f1 = 1
            #f = x1/(MAP_SIZE - 1)
            if self.hit([0,-1]):
                l1 = 1
            #l = y1/(MAP_SIZE - 1)
            if self.hit([0,1]):
                r1 = 1
            #r = y2/(MAP_SIZE - 1)
        if self.state[0] == -1 and self.state[1] == 0:
            if self.hit([-1,0]):
                f1 = 1
            #f = x2/(MAP_SIZE - 1)
            if self.hit([0,1]):
                l1 = 1
            #l = y1/(MAP_SIZE - 1)
            if self.hit([0,-1]):
                r1 = 1
            #r = y2/(MAP_SIZE - 1)
        
        d = self.dist_to_treat()
        return [f1, l1, r1 , d[0], d[1]]
            
    def hit(self, move):
        next_pos = array(self.pieces[0].place) + array(move)
        if next_pos[0] >= MAP_SIZE or next_pos[1] < 0 or next_pos[1] >= MAP_SIZE or next_pos[0] < 0:
            return True
        for piece in self.pieces:
            if piece.place[0] == next_pos[0] and piece.place[1] == next_pos[1]:
                return True
        if self.enemy:
            for piece in self.enemy.pieces:
                if piece.place[0] == next_pos[0] and piece.place[1] == next_pos[1]:
                    return True
        return False
    
    def dist_to_treat(self):
        
        f_dist = 0
        l_dist = 0
        first_piece = self.pieces[0].place
        if self.state[0] == 0 and self.state[1] == 1:
            f_dist = (self.treat.place[1] - first_piece[1])/(MAP_SIZE - 1)
            l_dist = (self.treat.place[0] - first_piece[0])/(MAP_SIZE - 1)
        if self.state[0] == 0 and self.state[1] == -1:
            f_dist = (first_piece[1] - self.treat.place[1])/(MAP_SIZE - 1)
            l_dist = (first_piece[0] - self.treat.place[0])/(MAP_SIZE - 1)
        if self.state[0] == 1 and self.state[1] == 0:
            f_dist = (self.treat.place[0] - first_piece[0])/(MAP_SIZE - 1)
            l_dist = (first_piece[1] - self.treat.place[1])/(MAP_SIZE - 1)
        if self.state[0] == -1 and self.state[1] == 0:
            f_dist = (first_piece[0] - self.treat.place[0])/(MAP_SIZE - 1)
            l_dist = (self.treat.place[1] - first_piece[1])/(MAP_SIZE - 1)
        
        return [f_dist, l_dist]
    
    def eat(self):
        if self.pieces[0].place[0] == self.treat.place[0] and self.pieces[0].place[1] == self.treat.place[1]:
            self.treat = Treat()
            self.score += 1
            return True
        return False
    
    def update_color(self, color):
        for piece in self.pieces:
            if not piece.is_first:
                piece.color = color
                piece.update_color(color)
    
    def point_to_obstacle(self):
        point = copy.copy(self.pieces[0].place)
        treat = copy.copy(self.treat)
        c = 0
        while ( c <= MAP_SIZE*2):
            for i in range(1,len(self.pieces)):
                if self.pieces[i].place[0] == point[0] and self.pieces[i].place[1] == point[1]:
                    return 1
            if (point[0] == treat.place[0] and point[1] == treat.place[1]):
                return 0
            dir1 = 0
            dir2 = 0
            if ( treat.place[0] - point[0] >= 0): dir1 = 1
            else: dir1 = -1 
            if (treat.place[1] - point[1] >= 0): dir2 = 1
            else: dir2 = -1
            if (abs(point[0] - treat.place[0]) >= abs(point[1] - treat.place[1])):
                point += [dir1,0]
            else:
                point += [0,dir2]
            
            c += 1
    
    def sim_fit(worm, n_tot, n):
        if n <= -1: return 0
        states = [[0,1], [0,-1], [1,0], [-1,0]]
        for i in states:
            if i[0]*(-1) == worm.state[0] and i[1]*(-1) == worm.state[1]: states.remove(i)
        fits = [0,0,0]
        for i in range(len(states)):
            val = worm.fitness(worm.pieces[0].place, states[i])
            fits[i] += val
            if not val==-5 and not val==3:
                new_worm = Worm([0] * 100, 10, 6, 1)
                for k in range(len(new_worm.pieces)):
                    new_worm.pieces[k] = WormPiece(place=copy.copy(worm.pieces[k].place))
                for k in range(len(new_worm.pieces),len(worm.pieces)):
                    new_worm.pieces.append(WormPiece(place=copy.copy(worm.pieces[k].place)))
                new_worm.treat = copy.copy(worm.treat)
                new_worm.update(states[i])
                fits[i] += Worm.sim_fit(new_worm, n_tot, n-1)
        return max(fits)
    
    # If foresight is executed, the worm foresees all possible n move combinations and chooses the best
    def foresight(self, n):
        states = [[0,1], [0,-1], [1,0], [-1,0]]
        for i in states:
            if i[0]*(-1) == self.state[0] and i[1]*(-1) == self.state[1]: states.remove(i)
        fits = [0,0,0]
        for i in range(len(states)):
            worm = Worm([0] * 100, 10, 6, 1)
            for k in range(len(worm.pieces)):
                worm.pieces[k] = WormPiece(place=copy.copy(self.pieces[k].place))
            for k in range(len(worm.pieces),len(self.pieces)):
                worm.pieces.append(WormPiece(place=copy.copy(self.pieces[k].place)))
            worm.treat = copy.copy(self.treat)
            val = worm.fitness(worm.pieces[0].place, states[i])
            fits[i] += val
            if not val==3 and not val==-5:
                worm.update(states[i])
                fits[i] += Worm.sim_fit(worm, n, n-1)

        if self.state[0] == 0 and self.state[1] == 1:
            if fits[0] > fits[1] and fits[0] > fits[2]: return [0,1]
            if fits[1] >= fits[0] and fits[1] >= fits[2]: return [1,0]
            return [-1,0]
        if self.state[0] == 0 and self.state[1] == -1:
            if fits[0] > fits[1] and fits[0] > fits[2]: return [0,-1]
            if fits[1] >= fits[0] and fits[1] >= fits[2]: return [1,0]
            return [-1,0]
        if self.state[0] == 1 and self.state[1] == 0:
            if fits[2] > fits[1] and fits[2] > fits[0]: return [1,0]
            if fits[0] >= fits[1] and fits[0] >= fits[2]: return [0,1]
            return [0,-1]
        if self.state[0] == -1 and self.state[1] == 0:
            if fits[2] > fits[1] and fits[2] > fits[0]: return [-1,0]
            if fits[1] >= fits[0] and fits[1] >= fits[2]: return [0,-1]
            return [0,1]
        
        
    def fitness(self, point, move):
        if (self.hit(move)):
            return -5
        if (Worm.dist(point + move, self.treat.place) >= Worm.dist(point, self.treat.place)):
            return -0.2
        if ( point[0] + move[0] == self.treat.place[0] and point[1] + move[1] == self.treat.place[1]):
            return 3
        else: return 0.1
    
    def dist(point1, point2):
        return( sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2))
            

def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    bird-wingup: An image of the bird with its wing pointing upward.
        Use this and bird-wingdown to create a flapping bird.
    bird-wingdown: An image of the bird with its wing pointing downward.
        Use this and bird-wingup to create a flapping bird.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (./images/).  All images are converted before being returned to
        speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        file_name = os.path.join(os.path.dirname(__file__),'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'treat': load_image('treat.png')}
            
class WormBrain():
    
    def __init__(self, nhneuron, neurons, ninput, noutput):
        self.neural_net = AI(nhneuron, neurons, ninput, noutput)
    
    def output(self, inp, state):
        outp = self.neural_net.output(inp)
        # if outp == -1, change direction to left, if outp == 0, no changes, if outp == 1, right
        if outp == 0:
            return state
        if state[0] == 0 and state[1] == 1:
            if outp == -1:
                return array([1,0])
            if outp == 1:
                return array([-1,0])
        if state[0] == 0 and state[1] == -1:
            if outp == -1:
                return array([-1,0])
            if outp == 1:
                return array([1,0])
        if state[0] == 1 and state[1] == 0:
            if outp == -1:
                return array([0,-1])
            if outp == 1:
                return array([0,1])
        if state[0] == -1 and state[1] == 0:
            if outp == -1:
                return array([0,1])
            if outp == 1:
                return array([0,-1])
            
        
class WormPiece():
    
    def __init__(self, place = array([int(floor(MAP_SIZE)), int(floor(MAP_SIZE)) + 1]), first = False, color = (255, 255, 255)):
        self.place = place
        self.color = color
        self.is_first = first
        self.image = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), SRCALPHA)
        self.image.convert()   # speeds up blitting
        if first:
            self.image.fill((200,200,200))
        else:
            self.image.fill(color)
        
    def update(self, state):
        self.place = array(self.place) + array(state)
        
    def rect(self):
        return pygame.Rect((self.place[0]*BLOCK_SIZE),(self.place[1]*BLOCK_SIZE), BLOCK_SIZE, BLOCK_SIZE)
        
    def update_color(self, color):
        self.image.fill(color)
        
class Treat():
    
    def __init__(self):
        self.place = random_vec()
        self.image = load_images()['treat']
    
    def rect(self):
        return Rect(self.place[0]*BLOCK_SIZE, self.place[1]*BLOCK_SIZE, BLOCK_SIZE,BLOCK_SIZE)

def random_vec(low=0, high=MAP_SIZE - 1):
    c1 = random.randint(low, high)
    c2 = random.randint(low, high)
    return array([c1,c2])

    
def play_genalg(population, gen, nhneuron, ninput, noutput):

    pygame.init()
    display_surface = pygame.display.set_mode((MAP_WIDTH, MAP_WIDTH))
    pygame.display.set_caption('Snake')

    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    frame_clock = 0
    background_img = pygame.Surface((1040,1040), True, (0,0))
    background_img.fill((0,0,0))
    population = Population(population, nhneuron, ninput, noutput, worm_size = INITIAL_WORM_SIZE, worm_pos_random = True)
    #enemy = Worm([0]*100, 1,1,1, enemy=True)
    #enemy.update_color((100,100,0))
    #for worm in population.worms:
    #    worm.set_enemy(enemy)
    
    done = False
    
    while not done:# and gen%10 == 0:
        clock.tick(FPS)
        for x in (0, MAP_WIDTH / 2):
                display_surface.blit(background_img, (x, 0))
        #display_surface.blit(img, (BLOCK_SIZE*19, BLOCK_SIZE*19))
        for worm in population.worms:
            if worm.score > population.best.score:
                population.best = worm
                population.best.update_color((255, 0, 0))
            if worm is not population.best:
                worm.update_color((255,255,255))
            
            if not worm.dead:
                for piece in worm.pieces:
                    display_surface.blit(piece.image, (piece.place[0]*BLOCK_SIZE, piece.place[1]*BLOCK_SIZE))
                display_surface.blit(worm.treat.image, (worm.treat.place[0]*BLOCK_SIZE, worm.treat.place[1]*BLOCK_SIZE))
                inp = worm.get_input()
                outp = worm.brain.output(inp, worm.state)
                eat = worm.eat()
                if worm.hit(outp) or worm.moves >= MAX_MOVES:
                    worm.dead = True
                    population.dead += 1
                else:
                    worm.update(outp, eat)
                    
        #if  not enemy.dead:
        #    for piece in enemy.pieces:
        #        display_surface.blit(piece.image, (piece.place[0]*BLOCK_SIZE, piece.place[1]*BLOCK_SIZE))
        #    display_surface.blit(enemy.treat.image, (enemy.treat.place[0]*BLOCK_SIZE, enemy.treat.place[1]*BLOCK_SIZE))
        #    outp = enemy.foresight(2)
        #    eat = enemy.eat()
        #    if enemy.hit(outp) or enemy.moves >= MAX_MOVES:
        #        enemy.dead = True
        #        population.dead += 1
        #    else:
        #        enemy.update(outp, eat)
                
        
        if population.dead == len(population.worms):
            done = True
        
        if done:
            print(population.best.pieces[0].place)
            inp = population.best.get_input()
            print(inp)
            print(population.best.brain.output(inp, population.best.state))
            
        gen_surface = score_font.render("Generation:" + str(gen), True, (255, 255, 255))
        score_surface = score_font.render("Best score:" + str(population.best.score), True, (255, 255, 255))
        gen_x = MAP_WIDTH/2 - gen_surface.get_width()/2
        display_surface.blit(score_surface, (gen_x, gen_surface.get_height() + 15))
        display_surface.blit(gen_surface, (gen_x, 10))

        pygame.event.pump()
        pygame.display.flip()

        frame_clock += 1
        
    while not done and not gen%10 == 0:
        for worm in population.worms:
            if not worm.dead:
                inp = worm.get_input()
                outp = worm.brain.output(inp, worm.state)
                eat = worm.eat()
                if worm.hit(outp) or worm.moves >= MAX_MOVES:
                    worm.dead = True
                    population.dead += 1
                else:
                    worm.update(outp, eat)
                    
        if population.dead == len(population.worms):
            done = True
    
   
    return population
        
def play_foresight():
    pygame.init()
    display_surface = pygame.display.set_mode((MAP_WIDTH, MAP_WIDTH))
    pygame.display.set_caption('Snake')

    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    frame_clock = 0
    img = load_images()['worm']
    background_img = load_images()['background']
    worm = Worm([0]*100, 1,1,1)
    
    done = False
    
    while not done:# and gen%10 == 0:
        clock.tick(FPS)
        for x in (0, MAP_WIDTH / 2):
                display_surface.blit(background_img, (x, 0))
        #display_surface.blit(img, (BLOCK_SIZE*19, BLOCK_SIZE*19))
        for piece in worm.pieces:
            display_surface.blit(piece.image, (piece.place[0]*BLOCK_SIZE, piece.place[1]*BLOCK_SIZE))
        display_surface.blit(worm.treat.image, (worm.treat.place[0]*BLOCK_SIZE, worm.treat.place[1]*BLOCK_SIZE))
        outp = worm.foresight(3)
        eat = worm.eat()
        if worm.hit(outp) or worm.moves >= MAX_MOVES:
            done = True
        else:
            worm.update(outp, eat)
        
        score_surface = score_font.render("Best score:" + str(worm.score), True, (255, 255, 255))
        score_x = MAP_WIDTH/2 - score_surface.get_width()/2
        display_surface.blit(score_surface, (score_x, 15))

        pygame.event.pump()
        pygame.display.flip()

        frame_clock += 1
        
def main():
    """The application's entry point.

    If someone executes this module (instead of importing it, for
    example), this function is called."""
    
    ninput = 5
    noutput = 1
    nhneuron = 16
    sol_per_pop = 100
    param_size = ninput*nhneuron + nhneuron + nhneuron*noutput + noutput
    # Defining the population size.

    pop_size = (sol_per_pop,param_size)
    # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

    #Creating the initial population.
    new_population = random.uniform(low=-4.0, high=4.0, size=pop_size)
    #new_population[0,] = [2.65788526,-0.28586704,-12.11243499,1.49806307,0.37706115,
  #-2.77640279,-4.89834013,1.75455785,-1.47743984,3.24675565,
  # 9.25893938,4.95792497,0.92421614,1.54219728,1.3221732,
  # 0.09988809,   0.203137,     1.46250542,   1.31234543,  -2.59589696,
  # 2.71554472,  -0.1359064,    2.64599773,  -8.09835741,  5.28159519,
  # 3.91780505,  -2.36047743, -12.85039548,  -3.95595393,  10.57089806,
  #-3.18495393,  -2.03925022,  -3.63258485,   7.65617045,  -4.57891004,
  #-5.3285765,   -2.29223479,  -2.09878869,   7.1110834,    5.45267432,
  #-1.2901836,    0.65468538,  16.56683689,  -3.06738816,  -2.07500671,
  # 2.14476024,  -2.984351,    -1.742256,     1.07028518,   1.37429585,
  # 1.107146,    11.64306812,   0.45591337,  -3.76973915,  -2.65104678,
  # 4.15555215,   6.35019955,   1.81884411,   6.34252231,  10.94451854,
  # 4.49946409,  -3.78356016,  -5.08067837,   0.63244935, -1.95015939,
  # 0.6137593,   -2.45897404,   5.67895412,   0.85297382,  -0.71323908,
  # 0.65911773]
    
    num_generations = 100

    num_parents_mating = 2
    gen = 1
    best_score = -1
    best_unit = None
    
    for generation in range(num_generations):
        # Measuring the fitness of each chromosome in the population.
        population = play_genalg(new_population, gen, nhneuron, ninput, noutput)
        gen += 1
        fitness = [0] * len(population.worms)
        for i in range(len(population.worms)):
            fitness[i] = population.worms[i].score
        print(max(fitness))
        if ( population.best.score > best_score):
            best_score = population.best.score
            best_unit = population.best
        # Selecting the best parents in the population for mating.
        parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)
        parents_with_best_unit = empty((num_parents_mating + 1, new_population.shape[1]))
        parents_with_best_unit[0:2, :] = parents[0:2, : ]
        parents_with_best_unit[2, :] = best_unit.neurons
 
        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(parents_with_best_unit, (pop_size[0]-parents_with_best_unit.shape[0], param_size))
 
        # Adding some variations to the offsrping using mutation.
        p = 0.2
        if max(fitness) <= 1:
            p = 1
        offspring_mutation = GA.mutation(offspring_crossover, p)
        # Creating the new population based on the parents and offspring.
        new_population[0:parents_with_best_unit.shape[0], :] = parents_with_best_unit
        new_population[parents_with_best_unit.shape[0]:, :] = offspring_mutation
            
    print(best_unit.brain.neural_net.params)

    """play_foresight()"""
    
if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    main()
    
    