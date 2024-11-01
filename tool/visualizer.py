import numpy as np
import matplotlib.pyplot as plt
import math


def axis2d(x=[-1, 1], y=[-1, 1], grid=True) :
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlim([x[0], x[1]]); ax.set_ylim([y[0], y[1]])
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.axhline(0, color='gray', linewidth=5, alpha = 0.25)
    plt.axvline(0, color='gray', linewidth=5, alpha = 0.25)
    ax.grid(grid)
    return ax

def draw_vec2d(axis, v, color='r', start_from = None, alpha = 1.0, label=None):
    if v.shape == (2, ):
        if start_from is None:
            start_from = np.zeros(2) # 원점 (0, 0)
        axis.quiver(*start_from, *v, color=color,
                angles='xy', scale_units='xy', scale=1, alpha=alpha)

        if label is not None:
            axis.text(*(start_from + v/2), label, fontsize=10)

def axis3d(x=[-1,1], y=[-1,1], z=[-1,1]) :
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')
    ax.set_xlim(x); ax.set_ylim(y); ax.set_zlim(z)
    ax.grid(False)
    origin = np.array([0,0,0])
    u0 = np.array([x[0], 0, 0])
    u = np.array([x[1]-x[0], 0, 0])
    v0 = np.array([0, y[0], 0])
    v = np.array([0, y[1]-y[0], 0])
    w0 = np.array([0, 0, z[0]])
    w = np.array([0, 0, z[1]-z[0]])
    ax.quiver(*u0, *u, color='red', alpha=0.2, arrow_length_ratio=0)
    ax.text(*([x[1], 0, 0]), 'x', fontsize=10)
    ax.quiver(*v0, *v, color='green', alpha=0.2, arrow_length_ratio=0)
    ax.text(*([0, y[1], 0]), 'y', fontsize=10)
    ax.quiver(*w0, *w, color='blue', alpha=0.2, arrow_length_ratio=0)
    ax.text(*([0, 0, z[1]]), 'z', fontsize=10)

    ax.view_init(elev=155, azim=-155, roll=-80)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return ax

def setCam(ax3d, cam_loc):
    dist = math.sqrt(cam_loc[0]**2 + cam_loc[1]**2 + cam_loc[2]**2)
    azim = 180*math.atan(cam_loc[1]/cam_loc[0])/math.pi
    elev = 90-180*math.acos(cam_loc[2]/dist)/math.pi
    roll = 120
    ax3d.view_init(elev, azim, roll)

def draw_vec3d(axis, v, color='r', start_from=None, alpha=1.0, label=None):
    xrange = axis.get_xlim()
    dx = xrange[1] - xrange[0]
    if v.shape == (3, ):
        if start_from is None:
            start_from = np.zeros(3) # 원점 (0, 0, 0)
        vector_length = np.linalg.norm(v)
        if alpha > 0.4:
            arrow_length_ratio=alpha*dx/(20*vector_length)
        else:
            arrow_length_ratio = 0.0
        axis.quiver(*start_from, *v, color=color,
                    arrow_length_ratio=arrow_length_ratio, alpha=alpha)

        if label is not None:
            axis.text(*(start_from + v/2), label, fontsize=10)

def draw_points_in_matrix22(axis, M, color='red') :
    x = M[0, :]
    y = M[1, :]
    axis.scatter(x, y, color=color) 

def draw_points(my_axis, points_list, labels=None, color='red'):
    for i in range(len(points_list)):
        my_axis.scatter(points_list[i][0], points_list[i][1], color=color)        
        my_axis.text(*(points_list[i]), labels[i], fontsize=10)

def draw_space_mat22(ax2d, M, label=None, color='gray'):
    draw_mat22(ax2d, M, label=label)
    u, v = M[:, 0], M[:, 1]

    for i in range(-10, 10):
        draw_vec2d(ax2d, v*20, color=color, start_from = u*i+v*(-10), alpha=0.2)
        draw_vec2d(ax2d, u*20, color=color, start_from = v*i+u*(-10), alpha=0.2)



from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_polygons(ax, polygon_list, facecolors, edgecolors='black', alpha=0.8):
    polys = Poly3DCollection(polygon_list, facecolors=facecolors, edgecolors=edgecolors, alpha=alpha)
    ax.add_collection3d(polys)

def draw_mat22(ax2d, M, label=None):
    # 2x2 행렬 시각화
    # 열벡터 가시화
    u, v = M[:, 0], M[:, 1]
    draw_vec2d(ax2d, u, color='r')
    draw_vec2d(ax2d, v, color='g')
    draw_vec2d(ax2d, u, color='gray', start_from = v, alpha=0.2)
    draw_vec2d(ax2d, v, color='gray', start_from = u, alpha=0.2)

    if label is not None:
        ax2d.text(*(u+v), label, fontsize=10)

def draw_mat33(ax3d, M, label=None):
    # 3x3 행렬 시각화
    # 열벡터 가시화
    u, v, w = M[:, 0], M[:, 1], M[:, 2]
    draw_vec3d(ax3d, u, color='r')
    draw_vec3d(ax3d, v, color='g')
    draw_vec3d(ax3d, w, color='b')

    draw_vec3d(ax3d, u, color='gray', start_from = v, alpha=0.2)
    draw_vec3d(ax3d, u, color='gray', start_from = w, alpha=0.2)
    draw_vec3d(ax3d, u, color='gray', start_from = v+w, alpha=0.2)

    draw_vec3d(ax3d, v, color='gray', start_from = u, alpha=0.2)
    draw_vec3d(ax3d, v, color='gray', start_from = w, alpha=0.2)
    draw_vec3d(ax3d, v, color='gray', start_from = u+w, alpha=0.2)

    draw_vec3d(ax3d, w, color='gray', start_from = u, alpha=0.2)
    draw_vec3d(ax3d, w, color='gray', start_from = v, alpha=0.2)
    draw_vec3d(ax3d, w, color='gray', start_from = u+v, alpha=0.2)

    if label is not None:
        ax3d.text(*(u+v+w), label, fontsize=10)


def draw_circle(ax, center=[0,0], radius=1.0, color='blue'):
  """
  Draws a circle on the given axes.

  Args:
    ax: The matplotlib axes object.
    center: A tuple containing the x and y coordinates of the center.
    radius: The radius of the circle.
    color: The color of the circle.
  """
  circle = plt.Circle(center, radius, color=color, fill=False)
  ax.add_patch(circle)

