import numpy as np
import matplotlib.pyplot as plt


def plot_vectors(ax, positions, vectors, start_index=0, end_index=-1, scale=1, plot_interval=100, linestyle='-'):
    """
    Plot a vector at each position, spaced by plot_interval.
    Args:
        ax: matplotlib axis
        positions: numpy array of shape (N, 3)
        vectors: numpy array of shape (N, 3)
        start_index: int
        end_index: int
        scale: float - multiplier to make the vectors larger or smaller
        plot_interval: int - Only plot the vectors every plot_interval positions
        linestyle: str - matplotlib linestyle
    """
    if end_index == -1:
        end_index = len(positions)

    for i in range(start_index, end_index, plot_interval):
        x = positions[i,0]
        y = positions[i,1]
        z = positions[i,2]
        vector = vectors[i,:] * scale
        ax.plot3D([x, x + vector[0]], [y, y + vector[1]], [z, z + vector[2]], color='k', linestyle=linestyle)


def plot_orientations(ax, position, rotation, start_index=0, end_index=-1, scale=0.2, plot_interval=100, linestyle='-'):
    """
    Plot a coordinate axis representing the orientation at each position, spaced by plot_interval. (X,Y,Z)<->(R,G,B)
    Args:
        ax: matplotlib axis
        position: numpy array of shape (N, 3)
        rotation: scipy.spatial.transform Rotation object of length N
        start_index: int
        end_index: int
        scale: float - multiplier to make the coordinate axes larger or smaller
        plot_interval: int - Only plot the coordinate axes every plot_interval positions
        linestyle: str - matplotlib linestyle
    """
    if end_index == -1:
        end_index = len(position)

    all_x_axes = rotation.apply(np.array([1,0,0]))
    all_y_axes = rotation.apply(np.array([0,1,0]))
    all_z_axes = rotation.apply(np.array([0,0,1]))
    for i in range(start_index, end_index, plot_interval):
        x = position[i,0]
        y = position[i,1]
        z = position[i,2]
        x_axis = all_x_axes[i,:] * scale
        y_axis = all_y_axes[i,:] * scale
        z_axis = all_z_axes[i,:] * scale
        ax.plot3D([x, x + x_axis[0]], [y, y + x_axis[1]], [z, z + x_axis[2]], color='r', linestyle=linestyle)
        ax.plot3D([x, x + y_axis[0]], [y, y + y_axis[1]], [z, z + y_axis[2]], color='g', linestyle=linestyle)
        ax.plot3D([x, x + z_axis[0]], [y, y + z_axis[1]], [z, z + z_axis[2]], color='b', linestyle=linestyle)


def plot_trajectory(ax, position, start_index=0, end_index=-1):
    """
    Plots a 3D trajectory.
    Args:
        ax: matplotlib axis
        position: numpy array of shape (N, 3)
        start_index: int
        end_index: int
    """
    if end_index == -1:
        end_index = len(position)
    ax.plot3D(position[start_index:end_index,0], position[start_index:end_index,1], position[start_index:end_index,2])


def generate_vertical_cylinder_points(center_x,center_y,radius,height_z):
    """
    Generate the (x,y,z) points for a vertical cylinder.
    Args:
        center_x: float Cylinder center x coordinate
        center_y: float Cylinder center y coordinate
        radius: float
        height_z: float
    """
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid
