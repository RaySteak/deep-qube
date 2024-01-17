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

class PriorityQueue:
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class Node():
    def __init__(self, tracked, path_cost, parent, action):
        self.tracked = tracked
        self.path_cost = path_cost
        self.parent = parent
        self.action = action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Astar():
    def __init__(self, lam = 1, num_iter = 10_000, network_type = 'normal', show_progress = False):
        self.lam = lam
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

        self.v_net = ValuePolicyNet(value_only = True).to(device) if network_type == 'normal' else ResnetModel(20, 24, 5000, 1000, 4, 1, True).to(device)
        self.v_net.load_state_dict(torch.load('v_net_normal.pt' if network_type == 'normal' else 'v_net_resnet.pt'))
        self.v_net.eval()

        self.cube = Cube()
        self.edges_corners = self.cube.edges_corners

    def heuristic(self, tracked):
        self.cube.tracked = tracked
        if self.cube.is_solved(use_tracked = True):
            return 0
        else:
            return self.v_net(torch.Tensor(Cube.encode_state(tracked, self.edges_corners)).to(device)[None, :]).item()

    def astar(self, root, num_iter):
        q = PriorityQueue()
        vis = {}

        vis[self.cube.tracked.tobytes()] = self.heuristic(self.cube.tracked)
        q.push(root, self.heuristic(self.cube.tracked))

        solved_node = None
        for _ in tqdm(range(num_iter), disable = not self.show_progress):
            cur_node = q.pop()
            
            for a in self.action_encode:
                if cur_node.action is not None and a[0] == cur_node.action[0] and len(a) != len(cur_node.action):
                    continue
                self.cube.tracked = np.copy(cur_node.tracked)
                self.cube.rotate_code(a)
                
                path_cost = cur_node.path_cost + 1
                
                if self.cube.is_solved(use_tracked = True):
                    solved_node = Node(self.cube.tracked, path_cost, cur_node, a)
                    break

                total_cost = self.lam * path_cost + self.heuristic(self.cube.tracked)

                if self.cube.tracked.tobytes() not in vis or vis[self.cube.tracked.tobytes()] > total_cost:
                    vis[self.cube.tracked.tobytes()] = total_cost
                    q.push(Node(self.cube.tracked, path_cost, cur_node, a), total_cost)
            if solved_node is not None:
                break
        
        return solved_node

    def solve(self, cube, num_iter = None):
        if num_iter is None:
            num_iter = self.num_iter
        
        root = Node(np.copy(cube.tracked), 0, None, None)
        solved_node = self.astar(root, num_iter)
        
        if solved_node is None:
            return None
        
        solution = []
        while solved_node.parent is not None:
            solution.append(solved_node.action)
            solved_node = solved_node.parent
            
        solution = ' '.join([a for a in reversed(solution)])
        
        return solution
        
if __name__ == '__main__':
    astar = Astar(num_iter = 10_000, show_progress = True, network_type = 'resnet')
    
    cube = Cube()
    num_moves = int(input("Input number of scramble moves: "))
    scramble_str = cube.get_scramble(num_moves)
    print("Scramble: ", scramble_str)
    cube.animate(scramble_str, interval=0.1, block=False)

    print("Solving...")
    solution = astar.solve(cube)

    if solution is None:
        print("No solution found")
    else:
        print("Solution: ", solution)
        cube.animate(solution, 3.0)
