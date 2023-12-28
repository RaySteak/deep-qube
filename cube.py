import matplotlib.pyplot as plt
import matplotlib.collections as collections
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class Cube():
    def __init__(self, facelet_size = 2):
        # Cube faces are represented as 3x3 matrices with the orange face in the front,
        # the white face on top, and the cube unraveled as follows:
        #   B
        #   W
        # O G R
        #   Y
        # Facelets in a face are then numbered in row-major order starting from the top left
        # in the unraveled format
        
        self.facelet_size = facelet_size
        self.face2num = {
            'F': 0,
            'B': 1,
            'R': 2,
            'L': 3,
            'U': 4,
            'D': 5
        }
        self.num2face = {f : n for n, f in self.face2num.items()}
        
        self.face2col = {
            'F': 'G',
            'B': 'B',
            'R': 'R',
            'L': 'O',
            'U': 'W',
            'D': 'Y'
        }
        self.num2col = {n : self.face2col[self.num2face[n]] for n in self.num2face}
        
        self.col2plt = {
            'W': 'white',
            'Y': 'yellow',
            'G': 'green',
            'B': 'blue',
            'R': 'red',
            'O': 'orange'
        }
        
        self.facelets = np.zeros((6, 3, 3), dtype = np.int8)
        
        for i in range(6):
            self.facelets[i] = i
    
    def rotate(self, face, clockwise):
        f = self.facelets
        f2n = self.face2num
        
        if face == 'F':
            if clockwise:
                f[f2n['F']] = np.rot90(f[f2n['F']], 3)
                up = np.copy(f[f2n['U'], 2, :])
                f[f2n['U'], 2, :] = np.flip(f[f2n['L'], :, 2])
                f[f2n['L'], :, 2] = f[f2n['D'], 0, :]
                f[f2n['D'], 0, :] = np.flip(f[f2n['R'], :, 0])
                f[f2n['R'], :, 0] = up
            else:
                f[f2n['F']] = np.rot90(f[f2n['F']], 1)
                up = np.copy(f[f2n['U'], 2, :])
                f[f2n['U'], 2, :] = f[f2n['R'], :, 0]
                f[f2n['R'], :, 0] = np.flip(f[f2n['D'], 0, :])
                f[f2n['D'], 0, :] = f[f2n['L'], :, 2]
                f[f2n['L'], :, 2] = np.flip(up)
        elif face == 'B':
            if clockwise:
                f[f2n['B']] = np.rot90(f[f2n['B']], 3)
                up = np.copy(f[f2n['U'], 0, :])
                f[f2n['U'], 0, :] = f[f2n['R'], :, 2]
                f[f2n['R'], :, 2] = np.flip(f[f2n['D'], 2, :])
                f[f2n['D'], 2, :] = f[f2n['L'], :, 0]
                f[f2n['L'], :, 0] = np.flip(up)
            else:
                f[f2n['B']] = np.rot90(f[f2n['B']], 1)
                up = np.copy(f[f2n['U'], 0, :])
                f[f2n['U'], 0, :] = np.flip(f[f2n['L'], :, 0])
                f[f2n['L'], :, 0] = f[f2n['D'], 2, :]
                f[f2n['D'], 2, :] = np.flip(f[f2n['R'], :, 2])
                f[f2n['R'], :, 2] = up
        elif face == 'R':
            if clockwise:
                f[f2n['R']] = np.rot90(f[f2n['R']], 3)
                up = np.copy(f[f2n['U'], :, 2])
                f[f2n['U'], :, 2] = f[f2n['F'], :, 2]
                f[f2n['F'], :, 2] = f[f2n['D'], :, 2]
                f[f2n['D'], :, 2] = f[f2n['B'], :, 2]
                f[f2n['B'], :, 2] = up
            else:
                f[f2n['R']] = np.rot90(f[f2n['R']], 1)
                up = np.copy(f[f2n['U'], :, 2])
                f[f2n['U'], :, 2] = f[f2n['B'], :, 2]
                f[f2n['B'], :, 2] = f[f2n['D'], :, 2]
                f[f2n['D'], :, 2] = f[f2n['F'], :, 2]
                f[f2n['F'], :, 2] = up
        elif face == 'L':
            if clockwise:
                f[f2n['L']] = np.rot90(f[f2n['L']], 3)
                up = np.copy(f[f2n['U'], :, 0])
                f[f2n['U'], :, 0] = f[f2n['B'], :, 0]
                f[f2n['B'], :, 0] = f[f2n['D'], :, 0]
                f[f2n['D'], :, 0] = f[f2n['F'], :, 0]
                f[f2n['F'], :, 0] = up
            else:
                f[f2n['L']] = np.rot90(f[f2n['L']], 1)
                up = np.copy(f[f2n['U'], :, 0])
                f[f2n['U'], :, 0] = f[f2n['F'], :, 0]
                f[f2n['F'], :, 0] = f[f2n['D'], :, 0]
                f[f2n['D'], :, 0] = f[f2n['B'], :, 0]
                f[f2n['B'], :, 0] = up
        elif face == 'U':
            if clockwise:
                f[f2n['U']] = np.rot90(f[f2n['U']], 3)
                front = np.copy(f[f2n['F'], 0, :])
                f[f2n['F'], 0, :] = f[f2n['R'], 0, :]
                f[f2n['R'], 0, :] = np.flip(f[f2n['B'], 2, :])
                f[f2n['B'], 2, :] = np.flip(f[f2n['L'], 0, :])
                f[f2n['L'], 0, :] = front
            else:
                f[f2n['U']] = np.rot90(f[f2n['U']], 1)
                front = np.copy(f[f2n['F'], 0, :])
                f[f2n['F'], 0, :] = f[f2n['L'], 0, :]
                f[f2n['L'], 0, :] = np.flip(f[f2n['B'], 2, :])
                f[f2n['B'], 2, :] = np.flip(f[f2n['R'], 0, :])
                f[f2n['R'], 0, :] = front
        elif face == 'D':
            if clockwise:
                f[f2n['D']] = np.rot90(f[f2n['D']], 3)
                front = np.copy(f[f2n['F'], 2, :])
                f[f2n['F'], 2, :] = f[f2n['L'], 2, :]
                f[f2n['L'], 2, :] = np.flip(f[f2n['B'], 0, :])
                f[f2n['B'], 0, :] = np.flip(f[f2n['R'], 2, :])
                f[f2n['R'], 2, :] = front
            else:
                f[f2n['D']] = np.rot90(f[f2n['D']], 1)
                front = np.copy(f[f2n['F'], 2, :])
                f[f2n['F'], 2, :] = f[f2n['R'], 2, :]
                f[f2n['R'], 2, :] = np.flip(f[f2n['B'], 0, :])
                f[f2n['B'], 0, :] = np.flip(f[f2n['L'], 2, :])
                f[f2n['L'], 2, :] = front
    
    def rotate_code(self, rotation_code):
        face = rotation_code[0]
        if face not in self.face2num:
            raise ValueError('Invalid rotation format')
        
        clockwise = True
        double_rotation = False
        if len(rotation_code) != 1:
            if len(rotation_code) > 2 or rotation_code[1] not in ['\'', '2']:
                raise ValueError('Invalid rotation format')

            clockwise = rotation_code[1] != '\''
            double_rotation = rotation_code[1] == '2'
        
        self.rotate(face, clockwise)
        if double_rotation:
            self.rotate(face, clockwise)

    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        
        fx = np.array([-self.facelet_size / 2, self.facelet_size / 2, self.facelet_size / 2, -self.facelet_size / 2])
        fy = np.array([-self.facelet_size / 2, -self.facelet_size / 2, self.facelet_size / 2, self.facelet_size / 2])
        fz = np.array([0, 0, 0, 0])
        
        for f in range(6):            
            for i in range(3):
                for j in range(3):
                    if self.num2face[f] == 'F':
                        draw_x = fz + self.facelet_size * 1.5
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fy - self.facelet_size * (i - 1)
                    elif self.num2face[f] == 'B':
                        draw_x = fz - self.facelet_size * 1.5
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fy + self.facelet_size * (i - 1)
                    elif self.num2face[f] == 'R':
                        draw_x = fx - self.facelet_size * (j - 1)
                        draw_y = fz + self.facelet_size * 1.5
                        draw_z = fy - self.facelet_size * (i - 1)
                    elif self.num2face[f] == 'L':
                        draw_x = fx + self.facelet_size * (j - 1)
                        draw_y = fz -self.facelet_size * 1.5
                        draw_z = fy - self.facelet_size * (i - 1)
                    elif self.num2face[f] == 'U':
                        draw_x = fy + self.facelet_size * (i - 1)
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fz + self.facelet_size * 1.5
                    elif self.num2face[f] == 'D':
                        draw_x = fy - self.facelet_size * (i - 1)
                        draw_y = fx + self.facelet_size * (j - 1)
                        draw_z = fz -self.facelet_size * 1.5
                    
                    verts = [list(zip(draw_x, draw_y, draw_z))]
                    ax.add_collection3d(Poly3DCollection(verts, facecolors = self.col2plt[self.num2col[self.facelets[f, i, j]]]))
                    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()

if __name__ == '__main__':
    rotation_str = input()

    cube = Cube()

    for rot_code in rotation_str.split(' '):
        cube.rotate_code(rot_code)

    cube.draw()
