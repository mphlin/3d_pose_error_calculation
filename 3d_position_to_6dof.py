import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from fit_cylinder import get_initial_circle_params, fit_circle_to_points
from plot_utils import generate_vertical_cylinder_points, plot_vectors, plot_orientations, plot_trajectory
pd.set_option("display.precision", 5)

# This script will take a trajectory of 3D positions (x,y,z) 
# and convert them to 6DOF (x,y,z,roll,pitch,yaw)
# Use case: getting pose ground-truth despite only collectiing position data with the API Radian or RTS

# Frame definitions: 
#   Robot frame (moving): x right, y forward, z is perpendicular to the asset (completes the right-handed coordinate system)
#   Camera frame wrt robot: x right, y towards bottom of robot, z forward

# input_file = r'data/vo_api_2023_07_11-19_34_40.csv'
# output_file = r'processed_data/processed_vo_api_2023_07_11-19_34_40.csv'
input_file = r'data/vo_api_08_01-2023_07_11-19_34_40.csv'
output_file = r'processed_data/processed_vo_api_08_01-2023_07_11-19_34_40.csv'

df = pd.read_csv(input_file)

########### Clean the data ###############

# Remove duplicate rows (where timestamps from sensors are the same)
df = df.drop_duplicates(subset=['/api_radian/pose/header/stamp', '/tf/vo_base/vo_camera/header/stamp'], ignore_index=True)

# Interpolate the API positions to fill in the holes where the API data was taken,
# and also the larger gaps where API lost line-of-sight
df['/api_radian/pose/pose/position/x'] = df['/api_radian/pose/pose/position/x'].interpolate(method='cubic')
df['/api_radian/pose/pose/position/y'] = df['/api_radian/pose/pose/position/y'].interpolate(method='cubic')
df['/api_radian/pose/pose/position/z'] = df['/api_radian/pose/pose/position/z'].interpolate(method='cubic')

# Make a column that has
#   1 if the robot is moving forward, 
#   -1 if it is moving backwards, 
#   0 if it is not moving (change in z is beneath a threshold)
# Assumption: in this dataset, the robot is moving forward iff it is moving upwards in the z direction faster than a threshold
# (In general you could get this from drive commands or encoders)
forward_threshold = 0.11 
df['forward'] = np.where(df['/api_radian/pose/pose/position/z'].diff() > 0, 1, -1)
df['forward'] = np.where(np.abs(df['/api_radian/pose/pose/position/z'].diff()) < forward_threshold, 0, df['forward'])

# Fill in all missing values with the previous value, and the first value with the next value
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

############### Fit a cylinder to the points ###################
# Find the circle that best fits the points
points = df[['/api_radian/pose/pose/position/x', '/api_radian/pose/pose/position/y', '/api_radian/pose/pose/position/z']].to_numpy()
param_radius = 13717 #mm
initial_parameters = get_initial_circle_params(points, param_radius=param_radius)
estimated_parameters_no_r = fit_circle_to_points(points, initial_parameters[:2],param_radius=initial_parameters[2] )
estimated_parameters = (estimated_parameters_no_r[0], estimated_parameters_no_r[1], initial_parameters[2], initial_parameters[3], initial_parameters[4])


############## Obtain the robot orientation from ground truth ######################
#   y = the tangent vector to the ground-truth trajectory
#   z = the vector perpendicular to the asset (a cylinder or a flat wall)
#   x = the cross product of y and z

# First, smooth the data using a Savitzky-Golay filter
positions = df[['/api_radian/pose/pose/position/x', 
                '/api_radian/pose/pose/position/y', 
                '/api_radian/pose/pose/position/z']].to_numpy()
smooth_positions = savgol_filter(positions, window_length=51, polyorder=3, axis=0)

tangent = smooth_positions[1:,:] - smooth_positions[0:-1,:]
tangent= np.concatenate((np.zeros((1,3)), tangent), axis=0) # keep same length as positions
norm = np.linalg.norm(tangent, axis=1)
norm[norm==0] = 1
robot_y = tangent / np.expand_dims(norm, axis=1)

# At the points where the ground-truth trajectory is not moving, set the tangent vector to the previous value
# (and set the sign according to the forward column)
forward_array = np.expand_dims(df['forward'].to_numpy(), axis=1)
robot_y = robot_y * forward_array

# Make any nan or 0 vectors into the previous vector
first_defined_index = np.where(np.any(robot_y, axis=1))[0][0]
robot_y[0,:] = robot_y[first_defined_index,:]
undefined_vector_indices = np.where(~np.any(robot_y, axis=1))[0]
for i in range(0, len(undefined_vector_indices)):
    robot_y[undefined_vector_indices[i],:] = robot_y[undefined_vector_indices[i]-1,:]

# # Smooth the tangent vectors again using the Savitzky-Golay filter
# robot_y = savgol_filter(robot_y, window_length=51, polyorder=3, axis=0)

# Next, get the vector perpendicular to the asset
# Assumption: the asset is a VERTICAL cylinder or wall, so the vector perpendicular to the asset is horizontal

# get the vector from the center of the cylinder to the point (at the same height)
cylinder_center = np.array([estimated_parameters[0], estimated_parameters[1],0])
center_to_point = positions - cylinder_center
center_to_point[:,2] = 0 # horizontal
norm = np.linalg.norm(center_to_point, axis=1)
norm[norm==0] = 1
robot_z = center_to_point / np.expand_dims(norm, axis=1)

# Finally, fill in the coordinate axes of the robot frame
robot_x = np.cross(robot_y, robot_z)
robot_from_api_matrix = np.stack((robot_x, robot_y, robot_z), axis=2)

# remove outliers as any rotation whose angle is significantly different from the previous rotation
def angle_between_rotation_matrices(R1, R2):
    # For two matrices R1, R2, the angle is related to the trace by tr(R1 R2^T) = 1 + 2cos(angle)
    return np.arccos(np.clip(0.5*(np.trace(R1 @ R2.T) - 1), -1.0, 1.0))

for i in range(1, len(robot_from_api_matrix)):
    angle = angle_between_rotation_matrices(robot_from_api_matrix[i-1,:,:], robot_from_api_matrix[i,:,:])
    if angle > 0.1:
        robot_from_api_matrix[i,:,:] = robot_from_api_matrix[i-1,:,:]


robot_from_api_rotation = R.from_matrix(robot_from_api_matrix)

##############################
"""
Make a new dataframe with the desired (processed) columns:
TODO: Change the VO node and API node so they automatically give you the robot frame wrt starting robot frame
    api.seconds
        The api's time in seconds since the start of the rosbag (not since epoch)
    api.position.x, api.position.y, api.position.z
        The x,y,z position in m (not mm) of the robot wrt the starting robot frame (not wrt the API)
    api.rotation.x, api.rotation.y, api.rotation.z, api.rotation.w
        The x,y,z,w quaternion of the robot wrt the starting robot frame (not wrt the API)
    vo.seconds
        The vo's time in seconds since the start of the rosbag (not since epoch)
    vo.position.x, vo.position.y, vo.position.z
        The x,y,z position in m of the robot wrt the starting robot frame (not camera frame wrt camera base)
    vo.rotation.x, vo.rotation.y, vo.rotation.z, vo.rotation.w
        The x,y,z,w quaternion of the robot wrt the starting robot frame (not camera frame wrt camera base)
    ros.seconds
        The ros time in seconds since the start of the rosbag (not since epoch)
"""
df_out = pd.DataFrame()
df_out['api.seconds'] = (df['/api_radian/pose/header/stamp'] - df['/api_radian/pose/header/stamp'][0])
df_out['api.position.x'] = (df['/api_radian/pose/pose/position/x']-df['/api_radian/pose/pose/position/x'][0]) / 1000.0
df_out['api.position.y'] = (df['/api_radian/pose/pose/position/y']-df['/api_radian/pose/pose/position/y'][0]) / 1000.0
df_out['api.position.z'] = (df['/api_radian/pose/pose/position/z']-df['/api_radian/pose/pose/position/z'][0]) / 1000.0

# Get the orientation of the robot wrt the starting frame (as opposed to the API)
robot_start_from_api_rotation = robot_from_api_rotation[0]
robot_from_robot_start_rotation = robot_start_from_api_rotation.inv() * robot_from_api_rotation
robot_from_robot_start_quat = robot_from_robot_start_rotation.as_quat()
df_out['api.rotation.x'] = robot_from_robot_start_quat[:,0]
df_out['api.rotation.y'] = robot_from_robot_start_quat[:,1]
df_out['api.rotation.z'] = robot_from_robot_start_quat[:,2]
df_out['api.rotation.w'] = robot_from_robot_start_quat[:,3]

###############################
# Get the VO orientation of robot vs robot start (The current VO is in the camera frame)
# TODO: Make the VO node output robot frame directly

camera_from_camera_base = R.from_quat(df[['/tf/vo_base/vo_camera/rotation/x',
                                            '/tf/vo_base/vo_camera/rotation/y',
                                            '/tf/vo_base/vo_camera/rotation/z',
                                            '/tf/vo_base/vo_camera/rotation/w']].to_numpy())

camera_from_robot = R.from_matrix(np.array([[1.0, 0.0, 0.0],
                                            [ 0.0, 0.0, -1.0],
                                            [ 0.0, 1.0, 0.0]]))

camera_base_from_api = robot_from_api_rotation[0] * camera_from_robot 

vo_robot_from_robot_start_rotation = camera_from_robot * camera_from_camera_base * camera_from_robot.inv()
vo_robot_from_robot_start_quat = vo_robot_from_robot_start_rotation.as_quat()
df_out['vo.rotation.x'] = vo_robot_from_robot_start_quat[:,0]
df_out['vo.rotation.y'] = vo_robot_from_robot_start_quat[:,1]
df_out['vo.rotation.z'] = vo_robot_from_robot_start_quat[:,2]
df_out['vo.rotation.w'] = vo_robot_from_robot_start_quat[:,3]

df_out['vo.seconds'] = (df['/tf/vo_base/vo_camera/header/stamp'] - df['/tf/vo_base/vo_camera/header/stamp'][0])

df_out['ros.seconds'] = (df['__time'] - df['__time'][0])

# Save the processed data
df_out.to_csv(output_file, index=False)

# ############################
# plot the trajectory with just the tangent vectors and vectors perpendicular to the asset
fig = plt.figure()
ax5 = plt.axes(projection='3d')
ax5.set_box_aspect((1,1,5))
ax5.grid(False)
ax5.set_xlabel('x (m)')
ax5.set_ylabel('y (m)')
ax5.set_zlabel('z (m)')
ax5.set_title('Trajectory and Tangent Vectors')
traj_start = len(df['/api_radian/pose/pose/position/x'])//2 # - 1000
traj_end = -1
plot_interval = 350
positions = df_out[['api.position.x', 'api.position.y', 'api.position.z']].to_numpy()
plot_trajectory(ax5, positions, start_index=traj_start, end_index=traj_end)
plot_vectors(ax5, positions, robot_y, start_index=traj_start, end_index=traj_end, scale=0.2, plot_interval=plot_interval)
plot_vectors(ax5, positions, robot_z, start_index=traj_start, end_index=traj_end, scale=0.2, plot_interval=plot_interval, linestyle='--')


# # plot the trajectory and orientation
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_box_aspect((1,1,5))
# ax.grid(False)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')
# ax.set_title('Trajectory and Orientation')

# traj_start = len(df['/api_radian/pose/pose/position/x'])//2 # - 1000
# traj_end = -1
# plot_interval = 350

# # robot_from_api_rotation = robot_start_from_api_rotation * robot_from_robot_start_rotation 
# vo_robot_from_api_rotation = robot_start_from_api_rotation * vo_robot_from_robot_start_rotation
# positions = df_out[['api.position.x', 'api.position.y', 'api.position.z']].to_numpy()
# plot_trajectory(ax, positions, start_index=traj_start, end_index=traj_end)
# plot_orientations(ax, positions, robot_from_api_rotation, start_index=traj_start, end_index=traj_end, scale=0.2, plot_interval=plot_interval)
# plot_orientations(ax, positions, vo_robot_from_api_rotation, start_index=traj_start, end_index=traj_end, scale=0.2, plot_interval=plot_interval, linestyle='--')


# Make a plot with the smoothed trajectory as a scatterplot, colored by the 'forward' column (whether moving up or down)
fig6 = plt.figure()
ax6 = plt.axes(projection='3d')
ax6.scatter3D(df_out['api.position.x'], df_out['api.position.y'], df_out['api.position.z'], c=df['forward'], cmap='coolwarm')
ax6.set_title('Smoothed Trajectory by Forward/Backward') 
ax6.set_box_aspect((1,1,5))
ax6.grid(False)
ax6.set_xlabel('x (m)')
ax6.set_ylabel('y (m)')
ax6.set_zlabel('z (m)')
# add a colorbar
cbar = fig6.colorbar(ax6.collections[0])
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(['Backward', 'Stationary', 'Forward'])
cbar.set_label('Movement Direction')

# Make a third plot with the trajectory and the cylinder
fig3 = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.scatter3D(df['/api_radian/pose/pose/position/x'], df['/api_radian/pose/pose/position/y'], df['/api_radian/pose/pose/position/z'], c=df['__time'], cmap='Greens')
ax3.set_title('Points and Cylinder')
ax3.set_box_aspect((1,1,1))
ax3.grid(False)

Xc,Yc,Zc = generate_vertical_cylinder_points(estimated_parameters[0], estimated_parameters[1], estimated_parameters[2], 4000)
ax3.plot_surface(Xc, Yc, Zc, alpha=0.5)

# # set x limits and y limits
ax3.set_xlim([-1800,-1300])
ax3.set_ylim([600,1100])

ax3.set_xlabel('x (mm)')
ax3.set_ylabel('y (mm)')
ax3.set_zlabel('z (mm)')


# Make a plot that shows the ground-truth yaw and the VO yaw for the trajectory

robot_from_robot_start_euler = robot_from_robot_start_rotation.as_euler('xyz', degrees=False)
vo_robot_from_robot_start_euler = vo_robot_from_robot_start_rotation.as_euler('xyz', degrees=False)

time = df_out['ros.seconds'].to_numpy() - df_out['ros.seconds'].to_numpy()[0]

fig4 = plt.figure()
ax4 = plt.axes()
ax4.plot(time, robot_from_robot_start_euler[:,2], label='API Orientation Estimate')
ax4.plot(time, vo_robot_from_robot_start_euler[:,2], label='Monocular Optical Flow VO')
ax4.set_xlabel('time (s)')
ax4.set_ylabel('yaw (rad)')
ax4.legend()
rosbag_title = input_file[-23:-4]
ax4.set_title(f'Visual Odometry Yaw: rosbag2_{rosbag_title}')


plt.show()













