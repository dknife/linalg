import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#####################################################
# Axes
#####################################################

def axis2d(x=[-1, 1], y=[-1, 1], grid=True) :
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlim([x[0], x[1]]); ax.set_ylim([y[0], y[1]])
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.axhline(0, color='gray', linewidth=5, alpha = 0.25)
    plt.axvline(0, color='gray', linewidth=5, alpha = 0.25)
    ax.grid(grid)
    return ax

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

def setup_axes(ax, title=''):
    """Common 3D axes setup (labels, title, equal aspect)."""
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_aspect('equal')


#####################################################
# Camera (LookAt)
#####################################################

def setCam(ax, eye, target, up=(0, 0, 1)):
    """
    Set camera using LookAt method.
    Constructs forward/right/up axes (same as a LookAt matrix),
    then converts to matplotlib's elev, azim, roll parameters.
    Args:
        ax     : 3D axes object
        eye    : (x, y, z) camera position
        target : (x, y, z) look-at point
        up     : (x, y, z) world up vector (default: Z-up)
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)

    forward = eye - target
    dist = np.linalg.norm(forward)
    forward = forward / dist

    azim = np.degrees(np.arctan2(forward[0], forward[1]))
    elev = np.degrees(np.arcsin(np.clip(forward[2], -1, 1)))

    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm > 1e-6:
        right = right / right_norm
    camera_up = np.cross(forward, right)

    default_up = np.array([-np.sin(np.radians(elev)) * np.sin(np.radians(azim)),
                           -np.sin(np.radians(elev)) * np.cos(np.radians(azim)),
                            np.cos(np.radians(elev))])
    default_right = np.cross(default_up, forward)
    d_norm = np.linalg.norm(default_right)
    if d_norm > 1e-6:
        default_right = default_right / d_norm

    cos_roll = np.clip(np.dot(camera_up, default_up), -1, 1)
    sin_roll = np.clip(np.dot(camera_up, default_right), -1, 1)
    roll = np.degrees(np.arctan2(sin_roll, cos_roll))

    ax.view_init(elev=elev, azim=azim, roll=roll)


#####################################################
# Vectors
#####################################################

def draw_vec2d(axis, v, color='r', start_from=None, alpha=1.0, label=None):
    if v.shape == (2, ):
        if start_from is None:
            start_from = np.zeros(2)
        axis.quiver(*start_from, *v, color=color,
                angles='xy', scale_units='xy', scale=1, alpha=alpha)

        if label is not None:
            axis.text(*(start_from + v/2), label, fontsize=10)

def draw_vec3d(axis, v, color='r', start_from=None, alpha=1.0, label=None):
    xrange = axis.get_xlim()
    dx = xrange[1] - xrange[0]
    if v.shape == (3, ):
        if start_from is None:
            start_from = np.zeros(3)
        vector_length = np.linalg.norm(v)
        if alpha > 0.4:
            arrow_length_ratio = alpha * dx / (20 * vector_length)
        else:
            arrow_length_ratio = 0.0
        axis.quiver(*start_from, *v, color=color,
                    arrow_length_ratio=arrow_length_ratio, alpha=alpha)

        if label is not None:
            axis.text(*(start_from + v/2), label, fontsize=10)


#####################################################
# Points
#####################################################

def draw_points_in_matrix(axis, M, color='red'):
    if M.shape[0] == 2:
        x = M[0, :]
        y = M[1, :]
        axis.scatter(x, y, color=color)
    elif M.shape[0] == 3:
        x = M[0, :]
        y = M[1, :]
        z = M[2, :]
        axis.scatter(x, y, z, color=color)

def draw_points(my_axis, points_list, labels=None, color='red'):
    for i in range(len(points_list)):
        if points_list[0].shape[0] == 2:
            my_axis.scatter(points_list[i][0], points_list[i][1], color=color)
        elif points_list[0].shape[0] == 3:
            my_axis.scatter(points_list[i][0], points_list[i][1], points_list[i][2], color=color)
        my_axis.text(*(points_list[i]), labels[i], fontsize=10)


#####################################################
# Matrix visualization
#####################################################

def draw_mat22(ax2d, M, label=None):
    u, v = M[:, 0], M[:, 1]
    draw_vec2d(ax2d, u, color='r')
    draw_vec2d(ax2d, v, color='g')
    draw_vec2d(ax2d, u, color='gray', start_from=v, alpha=0.2)
    draw_vec2d(ax2d, v, color='gray', start_from=u, alpha=0.2)

    if label is not None:
        ax2d.text(*(u+v), label, fontsize=10)

def draw_mat33(ax3d, M, label=None):
    u, v, w = M[:, 0], M[:, 1], M[:, 2]
    draw_vec3d(ax3d, u, color='r')
    draw_vec3d(ax3d, v, color='g')
    draw_vec3d(ax3d, w, color='b')

    draw_vec3d(ax3d, u, color='gray', start_from=v, alpha=0.2)
    draw_vec3d(ax3d, u, color='gray', start_from=w, alpha=0.2)
    draw_vec3d(ax3d, u, color='gray', start_from=v+w, alpha=0.2)

    draw_vec3d(ax3d, v, color='gray', start_from=u, alpha=0.2)
    draw_vec3d(ax3d, v, color='gray', start_from=w, alpha=0.2)
    draw_vec3d(ax3d, v, color='gray', start_from=u+w, alpha=0.2)

    draw_vec3d(ax3d, w, color='gray', start_from=u, alpha=0.2)
    draw_vec3d(ax3d, w, color='gray', start_from=v, alpha=0.2)
    draw_vec3d(ax3d, w, color='gray', start_from=u+v, alpha=0.2)

    if label is not None:
        ax3d.text(*(u+v+w), label, fontsize=10)

def draw_space_mat22(ax2d, M, label=None, color='gray'):
    draw_mat22(ax2d, M, label=label)
    u, v = M[:, 0], M[:, 1]

    for i in range(-10, 10):
        draw_vec2d(ax2d, v*20, color=color, start_from=u*i+v*(-10), alpha=0.2)
        draw_vec2d(ax2d, u*20, color=color, start_from=v*i+u*(-10), alpha=0.2)


#####################################################
# Shapes
#####################################################

def draw_polygons(ax, polygon_list, facecolors, edgecolors='black', alpha=0.8):
    polys = Poly3DCollection(polygon_list, facecolors=facecolors, edgecolors=edgecolors, alpha=alpha)
    ax.add_collection3d(polys)

def draw_circle(ax, center=[0,0], radius=1.0, color='blue'):
    """Draw a 2D circle outline."""
    circle = plt.Circle(center, radius, color=color, fill=False)
    ax.add_patch(circle)


#####################################################
# Bounding Objects
#####################################################

def draw_bounding_sphere(ax, center, radius, color='blue', alpha=0.3):
    """
    Draw a bounding sphere.
    Args:
        ax     : 3D axes object
        center : (x, y, z) center of the sphere
        radius : radius of the sphere
        color  : surface color
        alpha  : surface transparency (0=opaque, 1=transparent)
    """
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='gray', linewidth=0.3)
    ax.scatter(*center, color='red', s=30, zorder=5)
    ax.text(center[0], center[1], center[2] + radius + 0.3,
            f'C={center}\nr={radius}', fontsize=8, ha='center')


def draw_aabb(ax, min_pt, max_pt, color='green', alpha=0.2):
    """
    Draw an Axis-Aligned Bounding Box (AABB).
    Args:
        ax     : 3D axes object
        min_pt : (xmin, ymin, zmin)
        max_pt : (xmax, ymax, zmax)
        color  : surface color
        alpha  : surface transparency
    """
    min_pt = np.array(min_pt)
    max_pt = np.array(max_pt)

    vertices = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],  # 0
        [max_pt[0], min_pt[1], min_pt[2]],  # 1
        [max_pt[0], max_pt[1], min_pt[2]],  # 2
        [min_pt[0], max_pt[1], min_pt[2]],  # 3
        [min_pt[0], min_pt[1], max_pt[2]],  # 4
        [max_pt[0], min_pt[1], max_pt[2]],  # 5
        [max_pt[0], max_pt[1], max_pt[2]],  # 6
        [min_pt[0], max_pt[1], max_pt[2]],  # 7
    ])

    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]

    poly = Poly3DCollection(
        [vertices[f] for f in faces],
        alpha=alpha, facecolor=color, edgecolor='black', linewidth=1
    )
    ax.add_collection3d(poly)

    center = (min_pt + max_pt) / 2
    ax.scatter(*center, color='red', s=30, zorder=5)
    ax.text(center[0], center[1], max_pt[2] + 0.3,
            f'min={tuple(min_pt)}\nmax={tuple(max_pt)}', fontsize=8, ha='center')


def draw_obb(ax, center, half_extents, rotation, color='orange', alpha=0.2):
    """
    Draw an Oriented Bounding Box (OBB).
    Args:
        ax           : 3D axes object
        center       : (x, y, z) center of the OBB
        half_extents : (hx, hy, hz) half-size along each local axis
        rotation     : 3x3 rotation matrix (columns = local axes)
        color        : surface color
        alpha        : surface transparency
    """
    center = np.array(center)
    half_extents = np.array(half_extents)
    R = np.array(rotation)

    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ])

    vertices = np.array([center + R @ (s * half_extents) for s in signs])

    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]

    poly = Poly3DCollection(
        [vertices[f] for f in faces],
        alpha=alpha, facecolor=color, edgecolor='black', linewidth=1
    )
    ax.add_collection3d(poly)

    ax.scatter(*center, color='red', s=30, zorder=5)

    axis_names = ['u', 'v', 'w']
    axis_colors = ['red', 'green', 'blue']
    for i in range(3):
        tip = center + R[:, i] * half_extents[i]
        ax.plot([center[0], tip[0]], [center[1], tip[1]], [center[2], tip[2]],
                color=axis_colors[i], linewidth=2, label=f'{axis_names[i]}-axis')
