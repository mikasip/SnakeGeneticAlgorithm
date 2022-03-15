# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:24:59 2019

@author: Mika Sipilä
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:11:37 2019

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

class Worm():
    
    def __init__(self, neurons, nhneuron, ninput, noutput, size = 4, state=array([0,1])):
        self.moves = 0
        self.dead = False
        self.score = 0
        self.state = state
        self.pieces = deque()
        self.start_pos = array([int(floor(MAP_SIZE)/2), int(floor(MAP_SIZE)/2) + 1])
        self.treat = Treat(self)
        self.pieces.append(WormPiece(array(self.start_pos), first=True))
        self.mat = ones((MAP_SIZE + 1, MAP_SIZE + 1))
        for i in range(1,size):
            self.pieces.append(WormPiece(array(self.start_pos - (state*i))))
        for i in range(0,size):
            self.mat[self.pieces[i].place[0], self.pieces[i].place[1]] = -10
        self.mat[self.treat.place[0], self.treat.place[1]] = 10
    
    def update(self, state, eat = False):
        self.moves += 1
        self.state = state
        last_piece_coor = copy.copy(self.pieces[-1].place)
        last_piece_color = copy.copy(self.pieces[-1].color)
        prev_place = copy.copy(self.pieces[0].place)
        self.pieces[0].place += array(self.state)
        #self.mat[self.treat.place[0], self.treat.place[1]] = 10
        self.mat[self.pieces[0].place[0], self.pieces[0].place[1]] = -10
        self.mat[last_piece_coor[0], last_piece_coor[1]] = 1
        for i in range(1,len(self.pieces)):
            t_place = copy.copy(self.pieces[i].place)
            self.pieces[i].place = prev_place
            self.mat[self.pieces[i].place[0], self.pieces[i].place[1]] = -10
            prev_place = copy.copy(t_place)
        if eat:
            self.moves = 0
            self.pieces.append(WormPiece(place = last_piece_coor, color = last_piece_color))
            self.mat[last_piece_coor[0], last_piece_coor[1]] = -10
                        
        
    def hit(self, move):
        next_pos = array(self.pieces[0].place) + array(move)
        if next_pos[0] >= MAP_SIZE or next_pos[1] < 0 or next_pos[1] >= MAP_SIZE or next_pos[0] < 0:
            return True
        for piece in self.pieces:
            if piece.place[0] == next_pos[0] and piece.place[1] == next_pos[1]:
                return True
        return False
    
    def eat(self):
        if self.pieces[0].place[0] == self.treat.place[0] and self.pieces[0].place[1] == self.treat.place[1]:
            self.mat[self.treat.place[0], self.treat.place[1]] = 1
            self.treat = Treat(self)
            self.score += 1
            self.mat[self.treat.place[0], self.treat.place[1]] = 10
            return True
        return False
    
    def sim_fit(self, point, state, n_tot, n):
        if n <= -1: return 0
        states = [[0,1], [0,-1], [1,0], [-1,0]]
        for i in states:
            if i[0]*(-1) == state[0] and i[1]*(-1) == state[1]: states.remove(i)
        fits = [0,0,0]
        for i in range(len(states)):
            val = self.fitness(point, states[i])
            fits[i] += val
            if not val==-100 and not val==3:
                new_point = point + states[i]
                fits[i] += self.sim_fit(new_point, states[i], n_tot, n-1)
        return max(fits)
    
    # If foresight is executed, the worm foresees all possible n move combinations and chooses the best
    def foresight(self, n):
        states = [[0,1], [0,-1], [1,0], [-1,0]]
        for i in states:
            if i[0]*(-1) == self.state[0] and i[1]*(-1) == self.state[1]: states.remove(i)
        fits = [0,0,0]
        for i in range(len(states)):
            val = self.fitness(self.pieces[0].place , states[i])
            fits[i] += val
            if not val==-100 and not val==3:
                point = self.pieces[0].place + states[i]
                fits[i] += self.sim_fit(point, states[i], n, n-1)

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
        if point[0] + move[0] >= MAP_SIZE or point[0] + move[0] < 0 or point[1] + move[1] >= MAP_SIZE or point[1] + move[1] < 0:
            return -100
        if (self.mat[point[0] + move[0], point[1] + move[1]] == -10): 
            return -100
        if (Worm.dist(point + move, self.treat.place) >= Worm.dist(point, self.treat.place)):
            return -0.2
        if ( self.mat[point[0] + move[0], point[1] + move[1]] == 10 ):
            return 3
        else:
            return 0.1
    
    def dist(point1, point2):
        return( sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2))
        
    def point_to_obstacle(self, point1):
        point = copy.copy(point1)
        c = 0
        while ( c <= MAP_SIZE*2):
            if self.mat[point[0], point[1]] == -10:
                return True
            if self.mat[point[0], point[1]] == 10:
                return False
            dir1 = 0
            dir2 = 0
            if ( self.treat.place[0] - point[0] >= 0): dir1 = 1
            else: dir1 = -1 
            if (self.treat.place[1] - point[1] >= 0): dir2 = 1
            else: dir2 = -1
            if (abs(point[0] - self.treat.place[0]) >= abs(point[1] - self.treat.place[1])):
                point += [dir1,0]
            else:
                point += [0,dir2]
            
            c += 1    
            
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
        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'worm': load_image('pipe_body.png'),
            'treat': load_image('bird_wing_up.png')}
    
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
    
    def __init__(self, worm):
        self.place = random_vec()
        for piece in worm.pieces:
           if piece.place[0] == self.place[0] and piece.place[1] == self.place[1]:
               self.new_pos(worm)
        self.image = load_images()['treat']
    
    def rect(self):
        return Rect(self.place[0]*BLOCK_SIZE, self.place[1]*BLOCK_SIZE, BLOCK_SIZE,BLOCK_SIZE)
    
    def new_pos(self, worm):
        self.place = random_vec()
        for piece in worm.pieces:
           if piece.place[0] == self.place[0] and piece.place[1] == self.place[1]:
               self.new_pos(worm)
        
        
def random_vec(low=0, high=MAP_SIZE - 1):
    c1 = random.randint(low, high)
    c2 = random.randint(low, high)
    return array([c1,c2])

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
        #display_surface.blit(img, (19*BLOCK_SIZE, 19*BLOCK_SIZE))
        for piece in worm.pieces:
            display_surface.blit(piece.image, (piece.place[0]*BLOCK_SIZE, piece.place[1]*BLOCK_SIZE))
        display_surface.blit(worm.treat.image, (worm.treat.place[0]*BLOCK_SIZE, worm.treat.place[1]*BLOCK_SIZE))
        outp = worm.foresight(6)
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
        
        if done:
            print(worm.mat.T)
        
def main():
    
    play_foresight()
    
if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    main()