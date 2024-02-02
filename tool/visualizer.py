import numpy as np
import matplotlib.pyplot as plt
import math

def axis2d(x=[-1, 1], y=[-1, 1]) :
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlim([x[0], x[1]]); ax.set_ylim([y[0], y[1]])
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.axhline(0, color='gray', linewidth=5, alpha = 0.25)
    plt.axvline(0, color='gray', linewidth=5, alpha = 0.25)
    ax.grid(True)
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