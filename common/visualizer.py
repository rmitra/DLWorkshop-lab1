import numpy as np
import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

mpii_edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
              [6, 8], [8, 9]]

h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]

# def show_2d(img, points, c, edges):
#     num_joints = points.shape[0]
#     points = ((points.reshape(num_joints, -1))).astype(np.int32)
#     for j in range(num_joints):
#         cv2.circle(img, (points[j, 0], points[j, 1]), 3, c, -1)
#     for e in edges:
#         if points[e].min() > 0:
#             cv2.line(img, (points[e[0], 0], points[e[0], 1]),
#                      (points[e[1], 0], points[e[1], 1]), c, 2)
#     return img


class Visualize(object):
    def __init__(self, edges=mpii_edges):

        self.plt = plt

        oo = 1e10
        self.xmax, self.ymax, self.zmax = -oo, -oo, -oo
        self.xmin, self.ymin, self.zmin = oo, oo, oo
        self.pts_2d = None
        self.pts_3d = None
        self.edges = edges
        # self.data_count = 0
        # self.pts_3d_count = 0
        # self.pts_2d_count = 0

    def add_pt_3d(self, pt_3d):
        pt_3d = pt_3d.reshape(-1, 3)
        pt_3d = pt_3d[h36m_to_mpii, :]

        # self.pts_3d_count = self.pts_3d_count + 1

        self.pts_3d = pt_3d

    def add_pt_2d(self, pt_2d):
        pt_2d = pt_2d.reshape(-1, 2)
        pt_2d = pt_2d[h36m_to_mpii, :]
        # self.pts_2d_count = self.pts_2d_count + 1

        self.pts_2d = pt_2d

    def plot_3D(self, ax, pt_3d):
        self.xmax = 1000
        self.ymax = 1000
        self.zmax = 1000
        self.xmin = -1000
        self.ymin = -1000
        self.zmin = -1000

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_zlim(self.zmin, self.zmax)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # ax.grid(True)

        c = 'b'
        marker = 'o'

        x, y, z = np.zeros((3, pt_3d.shape[0]))

        for j in range(pt_3d.shape[0]):
            x[j] = pt_3d[j, 0].copy()
            y[j] = pt_3d[j, 1].copy()
            z[j] = pt_3d[j, 2].copy()

        ax.scatter(x, y, z, s=20, c=c, marker=marker)
        for e in self.edges:
            ax.plot(x[e], y[e], z[e], c=c)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.view_init(elev=-50, azim=-90)

    def plot_2D(self, ax, pt_2d):
        self.xmax = 1.0
        self.ymax = 1.0
        self.xmin = -1.0
        self.ymin = -1.0

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.grid(True)

        c = 'b'
        marker = 'o'

        # root relative 2d good for viz
        pt_2d = pt_2d - pt_2d[6:7, :]

        x, y = np.zeros((2, pt_2d.shape[0]))

        for j in range(pt_2d.shape[0]):
            x[j] = pt_2d[j, 0].copy()
            y[j] = -pt_2d[j, 1].copy()

        ax.scatter(x, y, s=20, c=c, marker=marker)
        for e in self.edges:
            ax.plot(x[e], y[e], c=c)

        ax.set_xticks([])
        ax.set_yticks([])

    def show(self):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(1, 2)

        ax_1 = fig.add_subplot(gs[0, 0])  # , projection='3d')
        self.plot_2D(ax_1, self.pts_2d)

        ax_2 = fig.add_subplot(gs[0, 1], projection='3d')
        self.plot_3D(ax_2, self.pts_3d)

        plt.show(block=True)


def show_pred_pose(poses_2d, poses_3d, n_joints=17):

    poses_2d = poses_2d.reshape(-1, n_joints, 2)
    poses_3d = poses_3d.reshape(-1, n_joints, 3)

    for i in range(poses_2d.shape[0]):
        viz = Visualize(edges=mpii_edges)

        viz.add_pt_2d(poses_2d[i])
        viz.add_pt_3d(poses_3d[i])

        viz.show()
