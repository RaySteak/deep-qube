import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import numpy as np


class Cube:
    def __init__(self, facelet_size=2):
        # Cube faces are represented as 3x3 matrices with the orange face in the front,
        # the white face on top, and the cube unraveled as follows:
        #   B
        #   W
        # O G R
        #   Y
        # Facelets in a face are then numbered in row-major order starting from the top left
        # in the unraveled format

        self.facelet_size = facelet_size
        self.face2num = {"F": 0, "B": 1, "R": 2, "L": 3, "U": 4, "D": 5}
        self.num2face = {f: n for n, f in self.face2num.items()}

        self.face2col = {"F": "G", "B": "B", "R": "R", "L": "O", "U": "W", "D": "Y"}
        self.num2col = {n: self.face2col[self.num2face[n]] for n in self.num2face}

        self.col2plt = {
            "W": "white",
            "Y": "yellow",
            "G": "green",
            "B": "blue",
            "R": "red",
            "O": "orange",
        }

        self.facelets = np.zeros((6, 3, 3), dtype=np.int8)

        for i in range(6):
            self.facelets[i] = i
        
        # A more compact format like in the DeepCube paper is also employed.
        # Therefore, the tracked stickers are selected as follows (marked with numbers):
        #
        #           -- -- --
        #           0  -- 1
        #           -- -- --
        #
        #           2  3  4
        #           5  -- 6 
        #           7  8  9
        #
        # -- -- --  -- -- --  -- -- --
        # -- -- --  10 -- 11  -- -- --
        # -- -- --  -- -- --  -- -- --
        #        
        #           12 13 14
        #           15 -- 16
        #           17 18 19
        #
        self.tracked = np.zeros((6, 3, 3), dtype = np.int8) - 1
        self.tracked[self.face2num['B'], 1, 0] = 0
        self.tracked[self.face2num['B'], 1, 2] = 1
        self.tracked[self.face2num['U'], 0, 0] = 2
        self.tracked[self.face2num['U'], 0, 1] = 3
        self.tracked[self.face2num['U'], 0, 2] = 4
        self.tracked[self.face2num['U'], 1, 0] = 5
        self.tracked[self.face2num['U'], 1, 2] = 6
        self.tracked[self.face2num['U'], 2, 0] = 7
        self.tracked[self.face2num['U'], 2, 1] = 8
        self.tracked[self.face2num['U'], 2, 2] = 9
        self.tracked[self.face2num['F'], 1, 0] = 10
        self.tracked[self.face2num['F'], 1, 2] = 11
        self.tracked[self.face2num['D'], 0, 0] = 12
        self.tracked[self.face2num['D'], 0, 1] = 13
        self.tracked[self.face2num['D'], 0, 2] = 14
        self.tracked[self.face2num['D'], 1, 0] = 15
        self.tracked[self.face2num['D'], 1, 2] = 16
        self.tracked[self.face2num['D'], 2, 0] = 17
        self.tracked[self.face2num['D'], 2, 1] = 18
        self.tracked[self.face2num['D'], 2, 2] = 19

        # The edge locations are numbered as follows:
        #
        #           -- 0  --
        #           1  -- 2
        #           -- 3  --
        #
        #           -- 4  --
        #           5  -- 6
        #           -- 7  --
        #
        # -- 8  --  -- 12 --  -- 16 --
        # 9  -- 10  13 -- 14  17 -- 18
        # -- 11 --  -- 15 --  -- 19 --
        #        
        #           -- 20 --
        #           21 -- 22
        #           -- 23 --
        #
        self.edges = np.zeros((6, 3, 3), dtype = np.int8) - 1
        self.edges[self.face2num['B'], 0, 1] = 0
        self.edges[self.face2num['B'], 1, 0] = 1
        self.edges[self.face2num['B'], 1, 2] = 2
        self.edges[self.face2num['B'], 2, 1] = 3
        self.edges[self.face2num['U'], 0, 1] = 4
        self.edges[self.face2num['U'], 1, 0] = 5
        self.edges[self.face2num['U'], 1, 2] = 6
        self.edges[self.face2num['U'], 2, 1] = 7
        self.edges[self.face2num['L'], 0, 1] = 8
        self.edges[self.face2num['L'], 1, 0] = 9
        self.edges[self.face2num['L'], 1, 2] = 10
        self.edges[self.face2num['L'], 2, 1] = 11
        self.edges[self.face2num['F'], 0, 1] = 12
        self.edges[self.face2num['F'], 1, 0] = 13
        self.edges[self.face2num['F'], 1, 2] = 14
        self.edges[self.face2num['F'], 2, 1] = 15
        self.edges[self.face2num['R'], 0, 1] = 16
        self.edges[self.face2num['R'], 1, 0] = 17
        self.edges[self.face2num['R'], 1, 2] = 18
        self.edges[self.face2num['R'], 2, 1] = 19
        self.edges[self.face2num['D'], 0, 1] = 20
        self.edges[self.face2num['D'], 1, 0] = 21
        self.edges[self.face2num['D'], 1, 2] = 22
        self.edges[self.face2num['D'], 2, 1] = 23

        # The corner locations are numbered as follows:
        #
        #           0  -- 1
        #           -- -- --
        #           2  -- 3
        #
        #           4  -- 5
        #           -- -- --
        #           6  -- 7
        #
        # 8  -- 9   12 -- 13  16 -- 17
        # -- -- --  -- -- --  -- -- --
        # 10 -- 11  14 -- 15  18 -- 19
        #        
        #           20 -- 21
        #           -- -- --
        #           22 -- 23
        #
        self.corners = np.zeros((6, 3, 3), dtype = np.int8)
        self.corners[self.face2num['B'], 0, 0] = 0
        self.corners[self.face2num['B'], 0, 2] = 1
        self.corners[self.face2num['B'], 2, 0] = 2
        self.corners[self.face2num['B'], 2, 2] = 3
        self.corners[self.face2num['U'], 0, 0] = 4
        self.corners[self.face2num['U'], 0, 2] = 5
        self.corners[self.face2num['U'], 2, 0] = 6
        self.corners[self.face2num['U'], 2, 2] = 7
        self.corners[self.face2num['L'], 0, 0] = 8
        self.corners[self.face2num['L'], 0, 2] = 9
        self.corners[self.face2num['L'], 2, 0] = 10
        self.corners[self.face2num['L'], 2, 2] = 11
        self.corners[self.face2num['F'], 0, 0] = 12
        self.corners[self.face2num['F'], 0, 2] = 13
        self.corners[self.face2num['F'], 2, 0] = 14
        self.corners[self.face2num['F'], 2, 2] = 15
        self.corners[self.face2num['R'], 0, 0] = 16
        self.corners[self.face2num['R'], 0, 2] = 17
        self.corners[self.face2num['R'], 2, 0] = 18
        self.corners[self.face2num['R'], 2, 2] = 19
        self.corners[self.face2num['D'], 0, 0] = 20
        self.corners[self.face2num['D'], 0, 2] = 21
        self.corners[self.face2num['D'], 2, 0] = 22
        self.corners[self.face2num['D'], 2, 2] = 23

    def rotate_array(self, arr, face, clockwise):
        f = arr
        f2n = self.face2num

        if face == "F":
            if clockwise:
                f[f2n["F"]] = np.rot90(f[f2n["F"]], 3)
                up = np.copy(f[f2n["U"], 2, :])
                f[f2n["U"], 2, :] = np.flip(f[f2n["L"], :, 2])
                f[f2n["L"], :, 2] = f[f2n["D"], 0, :]
                f[f2n["D"], 0, :] = np.flip(f[f2n["R"], :, 0])
                f[f2n["R"], :, 0] = up
            else:
                f[f2n["F"]] = np.rot90(f[f2n["F"]], 1)
                up = np.copy(f[f2n["U"], 2, :])
                f[f2n["U"], 2, :] = f[f2n["R"], :, 0]
                f[f2n["R"], :, 0] = np.flip(f[f2n["D"], 0, :])
                f[f2n["D"], 0, :] = f[f2n["L"], :, 2]
                f[f2n["L"], :, 2] = np.flip(up)
        elif face == "B":
            if clockwise:
                f[f2n["B"]] = np.rot90(f[f2n["B"]], 3)
                up = np.copy(f[f2n["U"], 0, :])
                f[f2n["U"], 0, :] = f[f2n["R"], :, 2]
                f[f2n["R"], :, 2] = np.flip(f[f2n["D"], 2, :])
                f[f2n["D"], 2, :] = f[f2n["L"], :, 0]
                f[f2n["L"], :, 0] = np.flip(up)
            else:
                f[f2n["B"]] = np.rot90(f[f2n["B"]], 1)
                up = np.copy(f[f2n["U"], 0, :])
                f[f2n["U"], 0, :] = np.flip(f[f2n["L"], :, 0])
                f[f2n["L"], :, 0] = f[f2n["D"], 2, :]
                f[f2n["D"], 2, :] = np.flip(f[f2n["R"], :, 2])
                f[f2n["R"], :, 2] = up
        elif face == "R":
            if clockwise:
                f[f2n["R"]] = np.rot90(f[f2n["R"]], 3)
                up = np.copy(f[f2n["U"], :, 2])
                f[f2n["U"], :, 2] = f[f2n["F"], :, 2]
                f[f2n["F"], :, 2] = f[f2n["D"], :, 2]
                f[f2n["D"], :, 2] = f[f2n["B"], :, 2]
                f[f2n["B"], :, 2] = up
            else:
                f[f2n["R"]] = np.rot90(f[f2n["R"]], 1)
                up = np.copy(f[f2n["U"], :, 2])
                f[f2n["U"], :, 2] = f[f2n["B"], :, 2]
                f[f2n["B"], :, 2] = f[f2n["D"], :, 2]
                f[f2n["D"], :, 2] = f[f2n["F"], :, 2]
                f[f2n["F"], :, 2] = up
        elif face == "L":
            if clockwise:
                f[f2n["L"]] = np.rot90(f[f2n["L"]], 3)
                up = np.copy(f[f2n["U"], :, 0])
                f[f2n["U"], :, 0] = f[f2n["B"], :, 0]
                f[f2n["B"], :, 0] = f[f2n["D"], :, 0]
                f[f2n["D"], :, 0] = f[f2n["F"], :, 0]
                f[f2n["F"], :, 0] = up
            else:
                f[f2n["L"]] = np.rot90(f[f2n["L"]], 1)
                up = np.copy(f[f2n["U"], :, 0])
                f[f2n["U"], :, 0] = f[f2n["F"], :, 0]
                f[f2n["F"], :, 0] = f[f2n["D"], :, 0]
                f[f2n["D"], :, 0] = f[f2n["B"], :, 0]
                f[f2n["B"], :, 0] = up
        elif face == "U":
            if clockwise:
                f[f2n["U"]] = np.rot90(f[f2n["U"]], 3)
                front = np.copy(f[f2n["F"], 0, :])
                f[f2n["F"], 0, :] = f[f2n["R"], 0, :]
                f[f2n["R"], 0, :] = np.flip(f[f2n["B"], 2, :])
                f[f2n["B"], 2, :] = np.flip(f[f2n["L"], 0, :])
                f[f2n["L"], 0, :] = front
            else:
                f[f2n["U"]] = np.rot90(f[f2n["U"]], 1)
                front = np.copy(f[f2n["F"], 0, :])
                f[f2n["F"], 0, :] = f[f2n["L"], 0, :]
                f[f2n["L"], 0, :] = np.flip(f[f2n["B"], 2, :])
                f[f2n["B"], 2, :] = np.flip(f[f2n["R"], 0, :])
                f[f2n["R"], 0, :] = front
        elif face == "D":
            if clockwise:
                f[f2n["D"]] = np.rot90(f[f2n["D"]], 3)
                front = np.copy(f[f2n["F"], 2, :])
                f[f2n["F"], 2, :] = f[f2n["L"], 2, :]
                f[f2n["L"], 2, :] = np.flip(f[f2n["B"], 0, :])
                f[f2n["B"], 0, :] = np.flip(f[f2n["R"], 2, :])
                f[f2n["R"], 2, :] = front
            else:
                f[f2n['D']] = np.rot90(f[f2n['D']], 1)
                front = np.copy(f[f2n['F'], 2, :])
                f[f2n['F'], 2, :] = f[f2n['R'], 2, :]
                f[f2n['R'], 2, :] = np.flip(f[f2n['B'], 0, :])
                f[f2n['B'], 0, :] = np.flip(f[f2n['L'], 2, :])
                f[f2n['L'], 2, :] = front
    
    def rotate(self, face, clockwise):
        self.rotate_array(self.facelets, face, clockwise)
        self.rotate_array(self.tracked, face, clockwise)
    
    def rotate_code(self, rotation_code):
        face = rotation_code[0]
        if face not in self.face2num:
            raise ValueError("Invalid rotation format")

        clockwise = True
        double_rotation = False
        if len(rotation_code) != 1:
            if len(rotation_code) > 2 or rotation_code[1] not in ["'", "2"]:
                raise ValueError("Invalid rotation format")

            clockwise = rotation_code[1] != "'"
            double_rotation = rotation_code[1] == "2"

        self.rotate(face, clockwise)
        if double_rotation:
            self.rotate(face, clockwise)
    
    def rotate_code_sequence(self, seq):
        for rot_code in seq.split(' '):
            self.rotate_code(rot_code)
    
    def num_correct_facelets(self):
        n = -6

        for f in range(6):
            n += np.sum(self.facelets[f] == f)

        return n

    def num_correct_sides(self):
        n = 0
        for f in range(6):
            n += (self.facelets[f] == f).all()

        return n
    
    def rotate_code_get_reward(self, rotation_code, reward_type = 'dqn'):
        if reward_type == 'dqn':
            cur_correct_facelets = self.num_correct_facelets()
            cur_correct_sides = self.num_correct_sides()
            
            self.rotate_code(rotation_code)
            
            next_correct_facelets = self.num_correct_facelets()
            next_correct_sides = self.num_correct_sides()
            
            if self.is_solved():
                return 100
                
            reward = -0.1
            reward += next_correct_facelets - cur_correct_facelets
            reward += (next_correct_sides - cur_correct_sides) * 10
        elif reward_type == 'deepcube':
            self.rotate_code(rotation_code)
            reward = 1 if self.is_solved() else -1

        return reward
    
    def get_scramble(self, num_rotations):
        prev_face = None
        scramble_str = ''
        for _ in range(num_rotations):
            while True:
                face = np.random.choice(list(self.face2num.keys()))
                if face != prev_face:
                    break
            rotation_type = np.random.choice(['', '\'', '2'])
            scramble_str += face + rotation_type + ' '
            prev_face = face
        return scramble_str.strip()
    
    def scramble(self, num_rotations = 20):
        scramble_str = self.get_scramble(num_rotations)
        self.rotate_code_sequence(scramble_str)
        return scramble_str

    def is_solved_state(self, state):
        for f in range(6):
            if not np.all(state[f] == f):
                return False
        return True

    def is_solved(self):
        return self.is_solved_state(self.facelets)
    
    def draw_to_axis(self, ax):
        fx = np.array([-self.facelet_size / 2, self.facelet_size / 2, self.facelet_size / 2, -self.facelet_size / 2])
        fy = np.array([-self.facelet_size / 2, -self.facelet_size / 2, self.facelet_size / 2, self.facelet_size / 2])
        fz = np.array([0, 0, 0, 0])

        for f in range(6):
            for i in range(3):
                for j in range(3):
                    if self.num2face[f] == "F":
                        draw_x = fz + self.facelet_size * 1.5
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fy - self.facelet_size * (i - 1)
                    elif self.num2face[f] == "B":
                        draw_x = fz - self.facelet_size * 1.5
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fy + self.facelet_size * (i - 1)
                    elif self.num2face[f] == "R":
                        draw_x = fx - self.facelet_size * (j - 1)
                        draw_y = fz + self.facelet_size * 1.5
                        draw_z = fy - self.facelet_size * (i - 1)
                    elif self.num2face[f] == "L":
                        draw_x = fx + self.facelet_size * (j - 1)
                        draw_y = fz - self.facelet_size * 1.5
                        draw_z = fy - self.facelet_size * (i - 1)
                    elif self.num2face[f] == "U":
                        draw_x = fy + self.facelet_size * (i - 1)
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fz + self.facelet_size * 1.5
                    elif self.num2face[f] == "D":
                        draw_x = fy - self.facelet_size * (i - 1)
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fz - self.facelet_size * 1.5

                    verts = [list(zip(draw_x, draw_y, draw_z))]
                    ax.add_collection3d(Poly3DCollection(verts, facecolors = self.col2plt[self.num2col[self.facelets[f, i, j]]]))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(-2 * self.facelet_size, 2 * self.facelet_size)
        ax.set_ylim(-2 * self.facelet_size, 2 * self.facelet_size)
        ax.set_zlim(-2 * self.facelet_size, 2 * self.facelet_size)
    
    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        
        self.draw_to_axis(ax)
        
        plt.show()


if __name__ == "__main__":
    # Test
    rotation_str = input()
    cube = Cube()
        
    if rotation_str != '':
        for rot_code in rotation_str.split(' '):
            cube.rotate_code(rot_code)
        cube.show()
    else:
        scramble = cube.get_scramble(20)
        cube.animate(scramble)