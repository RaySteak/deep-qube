import sys
sys.path.append('../')

import torch
import numpy as np
from cube import Cube
from value_policy_net import ValuePolicyNet
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


c = 5
nu = 0.0
num_iter = 10_000


action_encode = {
    'F': 0,
    'F\'': 1,
    'B' : 2,
    'B\'': 3,
    'L' : 4,
    'L\'': 5,
    'R' : 6,
    'R\'': 7,
    'U' : 8,
    'U\'': 9,
    'D' : 10,
    'D\'': 11,
}
action_decode = {encoding : action for action, encoding in action_encode.items()}

vp_net = ValuePolicyNet().to(device)
vp_net.eval()

cube = Cube()
edges = cube.edges
corners = cube.corners

num_moves = int(input("Input number of scramble moves: "))
scramble_str = cube.get_scramble(num_moves)
print("Scramble: ", scramble_str)
cube.animate(scramble_str, interval = 0.1, block = False)

def encode_state(tracked, edges, corners):
    encoded = np.zeros((20, 24))
    for f in range(6):
        for i in range(3):
            for j in range(3):
                is_edge = (i == 1) or (j == 1)
                if tracked[f, i, j] != -1:
                    pos_value = edges[f, i, j] if is_edge else corners[f, i, j]
                    encoded[tracked[f, i, j], pos_value] = 1
    
    return encoded


class TreeNode:
    def __init__(self, facelets, tracked, parent=None, action=None):
        self.facelets = facelets
        self.tracked = tracked
        
        self.parent = parent
        self.action = action

        self.W = np.zeros(12)
        self.N = np.zeros(12)
        self.L = np.zeros(12)
        
        self.visited = False
        self.children = []

    def add_child(self, child):
        self.children.append(child)

def mcts_simulate(node):
    if not node.visited:
        with torch.no_grad():
            node.V, node.P = vp_net(torch.Tensor(encode_state(node.tracked, edges, corners)).to(device)[None, :])
            node.V = node.V.item()
            node.P = torch.nn.Softmax(dim = 1)(node.P)[0].cpu().numpy()
        
        node.visited = True  
        for a in range(12):
            cube.facelets = np.copy(node.facelets)
            cube.tracked = np.copy(node.tracked)
            cube.rotate_code(action_decode[a])
            
            node.add_child(TreeNode(cube.facelets, cube.tracked, node, a))
        return node
    
    U = c * node.P * np.sqrt(np.sum(node.N)) / (1 + node.N)
    Q = node.W - node.L
    
    A = np.argmax(Q + U)
    
    unvisited_node = mcts_simulate(node.children[A])
    
    node.W[A] = np.max([unvisited_node.V, node.W[A]])
    node.N[A] += 1
    node.L -= nu
    
    return unvisited_node

def mcts(root, num_it):
    for _ in tqdm(range(num_it)):
        mcts_simulate(root)
        
        num_it -= 1
    
    return root

def bfs(root):
    queue = [root]
    while len(queue) > 0:
        node = queue.pop(0)
        
        cube.facelets = node.facelets
        if cube.is_solved():
            return node
        
        queue.extend(node.children)
    return None

root = TreeNode(np.copy(cube.facelets), np.copy(cube.tracked))
print("Solving...")
mcts(root, num_iter)
solved_node = bfs(root)

if solved_node is None:
    print("No solution found")
else:
    cube = Cube()
    cube.rotate_code_sequence(scramble_str)
    
    solution = []
    while solved_node.parent is not None:
        solution.append(solved_node.action)
        solved_node = solved_node.parent
    solution = ' '.join([action_decode[a] for a in reversed(solution)])
    print("Solution: ", solution)
    
    cube.animate(solution, 0.5)