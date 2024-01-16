import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from tqdm import tqdm
from value_policy_net import ValuePolicyNet
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

lam = 1
iterations = 100_000

action_encode = {
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
action_decode = {encoding: action for action,
                 encoding in action_encode.items()}

v_net = ValuePolicyNet(value_only = True).to(device)
v_net.load_state_dict(torch.load('v_net.pt'))
v_net.eval()

cube = Cube()
edges_corners = cube.edges_corners

num_moves = int(input("Input number of scramble moves: "))
scramble_str = cube.get_scramble(num_moves)
print("Scramble: ", scramble_str)
cube.animate(scramble_str, interval=0.1, block=False)


def heuristic(tracked):
    cube.tracked = tracked
    if cube.is_solved(use_tracked = True):
        return 0
    else:
        return v_net(torch.Tensor(Cube.encode_state(tracked, edges_corners)).to(device)[None, :]).item()

q = PriorityQueue()
vis = {}

vis[cube.tracked.tobytes()] = heuristic(cube.tracked)
q.push(Node(np.copy(cube.tracked), 0, None, None), heuristic(cube.tracked))

solved_node = None
for i in tqdm(range(iterations)):
    cur_node = q.pop()
    
    for a in action_encode:
        if cur_node.action is not None and a[0] == cur_node.action[0] and len(a) != len(cur_node.action):
            continue
        cube.tracked = np.copy(cur_node.tracked)
        cube.rotate_code(a)
        
        path_cost = cur_node.path_cost + 1
        
        if cube.is_solved(use_tracked = True):
            solved_node = Node(cube.tracked, path_cost, cur_node, a)
            break

        total_cost = lam * path_cost + heuristic(cube.tracked)

        if cube.tracked.tobytes() not in vis or vis[cube.tracked.tobytes()] > total_cost:
            vis[cube.tracked.tobytes()] = total_cost
            q.push(Node(cube.tracked, path_cost, cur_node, a), total_cost)
    if solved_node is not None:
        break

if solved_node is None:
    print("No solution found")
    exit()

cube = Cube()
cube.rotate_code_sequence(scramble_str)

solution = []
while solved_node.parent is not None:
    solution.append(solved_node.action)
    solved_node = solved_node.parent
solution = ' '.join([a for a in reversed(solution)])
print("Solution: ", solution)

cube.animate(solution, 0.5)
