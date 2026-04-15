import numpy as np
import plotly.graph_objects as go


#####################################################
# Figure creation
#####################################################

def figure3d(x=[-1,1], y=[-1,1], z=[-1,1], title='', width=600, height=600):
    """
    Create a 3D plotly figure with equal-aspect axes.
    Returns a go.Figure that you can add traces to and then call .show().
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        width=width, height=height,
        scene=dict(
            xaxis=dict(range=x, title='x'),
            yaxis=dict(range=y, title='y'),
            zaxis=dict(range=z, title='z'),
            aspectmode='cube',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def setCam(fig, eye, target=(0,0,0), up=(0,0,1)):
    """
    Set camera using eye position, look-at target, and up vector.
    """
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)

    # plotly expects eye in scene-coordinate scale
    fig.update_layout(scene_camera=dict(
        eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        center=dict(x=target[0], y=target[1], z=target[2]),
        up=dict(x=up[0], y=up[1], z=up[2]),
    ))


#####################################################
# Vectors
#####################################################

def _cone_mesh(tip, base, radius=0.06, n=12):
    """Generate cone mesh vertices/faces for an arrowhead."""
    direction = tip - base
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None, None, None, None, None, None

    d = direction / length
    if abs(np.dot(d, np.array([1,0,0]))) < 0.9:
        perp = np.cross(d, np.array([1,0,0]))
    else:
        perp = np.cross(d, np.array([0,1,0]))
    perp = perp / np.linalg.norm(perp)
    perp2 = np.cross(d, perp)

    theta = np.linspace(0, 2*np.pi, n+1)[:-1]
    ring = np.array([base + radius*(np.cos(t)*perp + np.sin(t)*perp2) for t in theta])

    # vertices: ring points + tip + base center
    verts = np.vstack([ring, [tip], [base]])
    tip_idx = n
    base_idx = n + 1

    i_list, j_list, k_list = [], [], []
    for idx in range(n):
        nxt = (idx + 1) % n
        # side face
        i_list.append(idx); j_list.append(nxt); k_list.append(tip_idx)
        # bottom face
        i_list.append(idx); j_list.append(nxt); k_list.append(base_idx)

    return verts[:,0], verts[:,1], verts[:,2], i_list, j_list, k_list


def draw_vec3d(fig, v, color='red', start_from=None, alpha=1.0, label=None,
               cone_ratio=0.12, cone_radius=0.06):
    """
    Draw a 3D vector arrow (line + cone arrowhead).
    Args:
        fig        : plotly Figure
        v          : vector (array-like, length 3)
        color      : color string
        start_from : origin point (default: [0,0,0])
        alpha      : opacity
        label      : text label at midpoint
        cone_ratio : fraction of vector length used for cone head
        cone_radius: radius of the cone base
    """
    v = np.array(v, dtype=float)
    if start_from is None:
        start_from = np.zeros(3)
    else:
        start_from = np.array(start_from, dtype=float)

    end = start_from + v
    length = np.linalg.norm(v)
    if length < 1e-10:
        return

    cone_len = length * cone_ratio
    cone_base = end - (v / length) * cone_len

    # shaft line
    fig.add_trace(go.Scatter3d(
        x=[start_from[0], cone_base[0]],
        y=[start_from[1], cone_base[1]],
        z=[start_from[2], cone_base[2]],
        mode='lines',
        line=dict(color=color, width=4),
        opacity=alpha,
        showlegend=False,
    ))

    # cone head
    cx, cy, cz, ci, cj, ck = _cone_mesh(end, cone_base, radius=cone_radius)
    if cx is not None:
        fig.add_trace(go.Mesh3d(
            x=cx, y=cy, z=cz, i=ci, j=cj, k=ck,
            color=color, opacity=alpha,
            showlegend=False,
        ))

    # label
    if label is not None:
        mid = start_from + v / 2
        fig.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode='text', text=[label],
            textfont=dict(size=12, color=color),
            showlegend=False,
        ))


#####################################################
# Points
#####################################################

def draw_points(fig, points_list, labels=None, color='red', size=5):
    """
    Draw a list of 3D points.
    """
    pts = np.array([np.array(p) for p in points_list])
    text = labels if labels else [None] * len(pts)
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='markers+text',
        marker=dict(size=size, color=color),
        text=text,
        textposition='top center',
        textfont=dict(size=10),
        showlegend=False,
    ))


def draw_points_in_matrix(fig, M, color='red', size=5):
    """
    Draw points stored as columns of matrix M (3xN).
    """
    fig.add_trace(go.Scatter3d(
        x=M[0,:], y=M[1,:], z=M[2,:],
        mode='markers',
        marker=dict(size=size, color=color),
        showlegend=False,
    ))


#####################################################
# Matrix visualization
#####################################################

def draw_mat33(fig, M, label=None):
    """
    Visualize a 3x3 matrix as a parallelepiped (3 column vectors + 9 ghost edges).
    """
    u, v, w = M[:, 0], M[:, 1], M[:, 2]

    # main column vectors
    draw_vec3d(fig, u, color='red')
    draw_vec3d(fig, v, color='green')
    draw_vec3d(fig, w, color='blue')

    # ghost edges (parallelepiped wireframe)
    ghost = [
        (u, v), (u, w), (u, v+w),
        (v, u), (v, w), (v, u+w),
        (w, u), (w, v), (w, u+v),
    ]
    for vec, origin in ghost:
        draw_vec3d(fig, vec, color='gray', start_from=origin, alpha=0.25,
                   cone_ratio=0, cone_radius=0)

    if label is not None:
        s = u + v + w
        fig.add_trace(go.Scatter3d(
            x=[s[0]], y=[s[1]], z=[s[2]],
            mode='text', text=[label],
            textfont=dict(size=12),
            showlegend=False,
        ))


#####################################################
# Shapes / Polygons
#####################################################

def draw_polygons(fig, polygon_list, facecolors, alpha=0.8):
    """
    Draw a list of polygons (each polygon is an Nx3 array of vertices).
    Triangulates each polygon as a fan from vertex 0.
    """
    for poly, fc in zip(polygon_list, facecolors):
        pts = np.array(poly)
        n = len(pts)
        if n < 3:
            continue
        i_list, j_list, k_list = [], [], []
        for idx in range(1, n - 1):
            i_list.append(0)
            j_list.append(idx)
            k_list.append(idx + 1)
        fig.add_trace(go.Mesh3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            i=i_list, j=j_list, k=k_list,
            color=fc, opacity=alpha,
            showlegend=False,
        ))


def draw_circle_3d(fig, center, normal, radius, color='blue', alpha=0.3, n_segments=64):
    """
    Draw a filled circle in 3D space.
    Args:
        fig      : plotly Figure
        center   : (x,y,z) center point
        normal   : (x,y,z) normal vector of the circle plane
        radius   : radius
        color    : fill color
        alpha    : opacity
    """
    center = np.array(center, dtype=float)
    normal = np.array(normal, dtype=float)
    n_unit = normal / np.linalg.norm(normal)

    if abs(np.dot(n_unit, np.array([1,0,0]))) < 0.9:
        arb = np.array([1,0,0])
    else:
        arb = np.array([0,1,0])
    t1 = np.cross(n_unit, arb)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n_unit, t1)

    theta = np.linspace(0, 2*np.pi, n_segments+1)[:-1]
    ring = np.array([center + radius*(np.cos(t)*t1 + np.sin(t)*t2) for t in theta])

    # fan triangulation from center
    verts = np.vstack([center.reshape(1,3), ring])
    n = len(ring)
    i_list = [0]*n
    j_list = list(range(1, n+1))
    k_list = [((idx % n) + 1) for idx in range(1, n+1)]

    fig.add_trace(go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=i_list, j=j_list, k=k_list,
        color=color, opacity=alpha,
        showlegend=False,
    ))

    # circle edge outline
    ring_closed = np.vstack([ring, ring[0:1]])
    fig.add_trace(go.Scatter3d(
        x=ring_closed[:,0], y=ring_closed[:,1], z=ring_closed[:,2],
        mode='lines',
        line=dict(color=color, width=3),
        opacity=min(alpha + 0.3, 1.0),
        showlegend=False,
    ))


#####################################################
# Plane (3 points)
#####################################################

def draw_plane(fig, p1, p2, p3, r=1, triangle_color='cyan', circle_color='yellow',
               triangle_alpha=0.7, circle_alpha=0.3):
    """
    Draw a triangle (p1, p2, p3) and a circle of radius r on the same plane,
    centered at the centroid.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    centroid = (p1 + p2 + p3) / 3.0
    normal = np.cross(p2 - p1, p3 - p1)

    # circle on the plane
    draw_circle_3d(fig, centroid, normal, r,
                   color=circle_color, alpha=circle_alpha)

    # triangle
    fig.add_trace(go.Mesh3d(
        x=[p1[0], p2[0], p3[0]],
        y=[p1[1], p2[1], p3[1]],
        z=[p1[2], p2[2], p3[2]],
        i=[0], j=[1], k=[2],
        color=triangle_color, opacity=triangle_alpha,
        showlegend=False,
    ))

    # triangle edges
    for a, b in [(p1,p2), (p2,p3), (p3,p1)]:
        fig.add_trace(go.Scatter3d(
            x=[a[0],b[0]], y=[a[1],b[1]], z=[a[2],b[2]],
            mode='lines',
            line=dict(color='black', width=3),
            showlegend=False,
        ))

    # vertex labels
    for pt, name in zip([p1, p2, p3], ['P1', 'P2', 'P3']):
        fig.add_trace(go.Scatter3d(
            x=[pt[0]], y=[pt[1]], z=[pt[2]],
            mode='markers+text',
            marker=dict(size=4, color='red'),
            text=[name], textposition='top center',
            textfont=dict(size=10),
            showlegend=False,
        ))


#####################################################
# Plane (point + normal)
#####################################################

def draw_plane_from_normal(fig, p, n, r=3, plane_color='yellow', plane_alpha=0.2):
    """
    Draw a circular plane through point p with normal n, plus the normal arrow.
    """
    p = np.array(p, dtype=float)
    n = np.array(n, dtype=float)

    draw_circle_3d(fig, p, n, r, color=plane_color, alpha=plane_alpha)
    draw_vec3d(fig, n, color='red', start_from=p, label='n')

    fig.add_trace(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode='markers+text',
        marker=dict(size=4, color='red'),
        text=['P'], textposition='top center',
        textfont=dict(size=10),
        showlegend=False,
    ))


#####################################################
# Bounding Objects
#####################################################

def draw_bounding_sphere(fig, center, radius, color='blue', alpha=0.3):
    """
    Draw a bounding sphere.
    """
    center = np.array(center, dtype=float)
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        opacity=alpha,
        showscale=False,
    ))

    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers+text',
        marker=dict(size=3, color='red'),
        text=[f'C={tuple(center)}<br>r={radius}'],
        textposition='top center',
        textfont=dict(size=9),
        showlegend=False,
    ))


def _box_mesh(vertices):
    """Create Mesh3d trace for a box defined by 8 vertices."""
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]
    i_list, j_list, k_list = [], [], []
    for f in faces:
        # split quad into two triangles
        i_list.append(f[0]); j_list.append(f[1]); k_list.append(f[2])
        i_list.append(f[0]); j_list.append(f[2]); k_list.append(f[3])
    return i_list, j_list, k_list


def draw_aabb(fig, min_pt, max_pt, color='green', alpha=0.2):
    """
    Draw an Axis-Aligned Bounding Box (AABB).
    """
    min_pt = np.array(min_pt, dtype=float)
    max_pt = np.array(max_pt, dtype=float)

    vertices = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
    ])

    i, j, k = _box_mesh(vertices)
    fig.add_trace(go.Mesh3d(
        x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
        i=i, j=j, k=k,
        color=color, opacity=alpha,
        showlegend=False,
    ))

    center = (min_pt + max_pt) / 2
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers+text',
        marker=dict(size=3, color='red'),
        text=[f'min={tuple(min_pt)}<br>max={tuple(max_pt)}'],
        textposition='top center',
        textfont=dict(size=9),
        showlegend=False,
    ))


def draw_obb(fig, center, half_extents, rotation, color='orange', alpha=0.2):
    """
    Draw an Oriented Bounding Box (OBB).
    """
    center = np.array(center, dtype=float)
    half_extents = np.array(half_extents, dtype=float)
    R = np.array(rotation, dtype=float)

    signs = np.array([
        [-1,-1,-1], [+1,-1,-1], [+1,+1,-1], [-1,+1,-1],
        [-1,-1,+1], [+1,-1,+1], [+1,+1,+1], [-1,+1,+1],
    ])
    vertices = np.array([center + R @ (s * half_extents) for s in signs])

    i, j, k = _box_mesh(vertices)
    fig.add_trace(go.Mesh3d(
        x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
        i=i, j=j, k=k,
        color=color, opacity=alpha,
        showlegend=False,
    ))

    # local axes
    axis_names = ['u', 'v', 'w']
    axis_colors = ['red', 'green', 'blue']
    for idx in range(3):
        tip = center + R[:, idx] * half_extents[idx]
        fig.add_trace(go.Scatter3d(
            x=[center[0], tip[0]],
            y=[center[1], tip[1]],
            z=[center[2], tip[2]],
            mode='lines',
            line=dict(color=axis_colors[idx], width=4),
            name=f'{axis_names[idx]}-axis',
            showlegend=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers',
        marker=dict(size=3, color='red'),
        showlegend=False,
    ))
