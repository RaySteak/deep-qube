import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from tqdm import tqdm
from value_policy_net import ValuePolicyNet
from cube import Cube
import numpy as np
import torch

# Just in case
sys.setrecursionlimit(100_000)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TreeNode:
    def __init__(self, tracked, parent=None, action=None):
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

class MCTS():
    def __init__(self, c = 5, nu = 0.1, num_iter = 10_000, show_progress = False):
        self.c = c
        self.nu = nu
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

        self.vp_net = ValuePolicyNet(value_only = False).to(device)
        self.vp_net.load_state_dict(torch.load('vp_net.pt'))
        self.vp_net.eval()
        
        self.cube = Cube()
        self.edges_corners = self.cube.edges_corners


    def mcts_simulate(self, node):
        if not node.visited:
            with torch.no_grad():
                node.V, node.P = self.vp_net(torch.Tensor(Cube.encode_state(
                    node.tracked, self.edges_corners)).to(device)[None, :])
                node.V = node.V.item()
                node.P = torch.nn.Softmax(dim=1)(node.P)[0].cpu().numpy()

            node.visited = True
            for a in range(12):
                cube.tracked = np.copy(node.tracked)
                cube.rotate_code(self.action_decode[a])

                node.add_child(TreeNode(cube.tracked, node, a))
            return node

        U = self.c * node.P * np.sqrt(np.sum(node.N)) / (1 + node.N)
        Q = node.W - node.L

        A = np.argmax(Q + U)

        unvisited_node = self.mcts_simulate(node.children[A])

        node.W[A] = np.max([unvisited_node.V, node.W[A]])
        node.N[A] += 1
        node.L -= self.nu

        return unvisited_node

    def mcts(self, root, num_iter, show_progress):
        for _ in tqdm(range(num_iter), disable = not show_progress):
            self.mcts_simulate(root)
        return root
    
    def bfs(self, root):
        queue = [root]
        while len(queue) > 0:
            node = queue.pop(0)

            cube.tracked = node.tracked
            if cube.is_solved(use_tracked = True):
                return node

            queue.extend(node.children)
        return None

    def solve(self, cube, num_iter = None):
        if num_iter is None:
            num_iter = self.num_iter
        
        root = TreeNode(np.copy(cube.tracked))
        root = self.mcts(root, num_iter, self.show_progress)
        solved_node = self.bfs(root)
        
        if solved_node is None:
            return None
        
        solution = []
        while solved_node.parent is not None:
            solution.append(solved_node.action)
            solved_node = solved_node.parent
            
        solution = ' '.join([self.action_decode[a] for a in reversed(solution)])
        
        return solution

if __name__ == '__main__':
    mcts = MCTS(num_iter = 10_000, show_progress = True)
    
    cube = Cube()
    num_moves = int(input("Input number of scramble moves: "))
    scramble_str = cube.get_scramble(num_moves)
    print("Scramble: ", scramble_str)
    cube.animate(scramble_str, interval=0.1, block=False)

    print("Solving...")
    solution = mcts.solve(cube)

    if solution is None:
        print("No solution found")
    else:
        print("Solution: ", solution)
        cube.animate(solution, 0.5)
