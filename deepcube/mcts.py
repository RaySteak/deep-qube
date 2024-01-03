import sys
sys.path.append('../')

import torch
from cube import Cube
from value_policy_net import ValuePolicyNet


vp_net = ValuePolicyNet()
vp_net.eval()

cube = Cube()

num_moves = int(input("Input number of scramble moves: "))
cube.scramble(num_moves)


class TreeNode:
    def __init__(self, state, agent_index, parent=None, action=None, reward=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.reward = reward
        self.agent_index = agent_index
        self.children = []
        self.visited_actions = []
        self.visits = 0
        self.value = 0

    def add_child(self, child):
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1
        self.value = self.reward / self.visits

def get_not_fully_expanded(node):
    pass

def rollout(node):
    pass

def backpropagate(node, reward):
    pass

def mcts(root, num_it):
    while num_it > 0:
        unvisited_node = get_not_fully_expanded(root)
        reward = rollout(unvisited_node)
        backpropagate(unvisited_node, reward)
        
        num_it -= 1
    
    return root