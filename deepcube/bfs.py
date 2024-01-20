import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from tqdm import tqdm
from value_policy_net import ValuePolicyNet
from resnet_model import ResnetModel
from cube import Cube
import numpy as np
import torch
import heapq

class Node():
    def __init__(self, facelets, parent, action):
        self.facelets = facelets
        self.parent = parent
        self.action = action

class BFS():
    def __init__(self, num_iter = 10_000, show_progress = False):
        self.num_iter = num_iter
        self.show_progress = show_progress
        
        self.action_encode = {
            'F': 0,
            'F\'': 1,
            'B': 2,
            'B\'': 3,
            'L': 4,
            'L\'': 5,
            'R': 6,
            'R\'': 7,
            'U': 8,
            'U\'': 9,
            'D': 10,
            'D\'': 11,
        }
        self.action_decode = {encoding: action for action,
                        encoding in self.action_encode.items()}
        
        self.cube = Cube()
        self.edges_corners = self.cube.edges_corners

    def bfs(self, root, num_iter):
        q = [root]
        vis = set()
        
        for i in tqdm(range(num_iter), disable = not self.show_progress):
            cur_node = q.pop(0)
            
            self.cube.facelets = cur_node.facelets
            if self.cube.is_solved(use_tracked = False):
                return cur_node, i
            
            for a in self.action_encode:
                self.cube.facelets = np.copy(cur_node.facelets)
                self.cube.rotate_code(a)

                if self.cube.facelets.tobytes() not in vis:
                    vis.add(self.cube.facelets.tobytes())
                    q.append(Node(self.cube.facelets, cur_node, a))
        
        return None, None

    def solve(self, cube, num_iter = None, return_steps_taken = False):
        if num_iter is None:
            num_iter = self.num_iter
        
        root = Node(np.copy(cube.facelets), None, None)
        solved_node, steps_taken = self.bfs(root, num_iter)
        
        if solved_node is None:
            return None if not return_steps_taken else (None, None)
        
        solution = []
        while solved_node.parent is not None:
            solution.append(solved_node.action)
            solved_node = solved_node.parent
            
        solution = ' '.join([a for a in reversed(solution)])
        
        return solution if not return_steps_taken else (solution, steps_taken)
        
if __name__ == '__main__':
    bfs = BFS(num_iter = 100_000, show_progress = True)
    
    cube = Cube()
    
    user_input = input("Input number of scramble moves (press enter to input own scramble): ")
    if user_input == '':
        scramble_str = input("Input your own scramble: ")
    else:
        num_moves = int(user_input)
        scramble_str = cube.get_scramble(num_moves)
        print("Scramble: ", scramble_str)
    cube.animate(scramble_str, interval=0.1, block=False)

    print("Solving...")
    solution = bfs.solve(cube)

    if solution is None:
        print("No solution found")
    else:
        print("Solution: ", solution)
        cube.animate(solution, 3.0)
