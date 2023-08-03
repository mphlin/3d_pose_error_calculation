"""
Write a script that read datas from csv file in pandas and a calculate error from them.
"""
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt

from error_analysis import interpolate_line, get_1d_rms_error, get_3d_rms_error, get_3d_euclidean_norm, get_RT_matrix


#Get path    
absolute_path = os.path.dirname(__file__)
relative_path = "filter_rts_csv/filte_vs_rts2.csv"
full_path = os.path.join(absolute_path, relative_path)

data = pd.read_csv(full_path)
rts_y= data[['/triangulation_3D_pose/pose/pose/position/y']].fillna(0).to_numpy().reshape(-1, 1)
filter_y= data[['/odometry/filtered/pose/pose/position/y']].fillna(0).to_numpy().reshape(-1, 1)
odometry_y= data[['/sys/integrator/wheel_odometry/pose/pose/position/y']].fillna(0).to_numpy().reshape(-1, 1)

rts_x= data[['/triangulation_3D_pose/pose/pose/position/x']].fillna(0).to_numpy().reshape(-1, 1)
filter_x= data[['/odometry/filtered/pose/pose/position/x']].fillna(0).to_numpy().reshape(-1, 1)
odometry_x= data[['/sys/integrator/wheel_odometry/pose/pose/position/x']].fillna(0).to_numpy().reshape(-1, 1)

rts_z= data[['/triangulation_3D_pose/pose/pose/position/z']].fillna(0).to_numpy().reshape(-1, 1)
filter_z= data[['/odometry/filtered/pose/pose/position/z']].fillna(0).to_numpy().reshape(-1, 1)
odometry_z= data[['/sys/integrator/wheel_odometry/pose/pose/position/z']].fillna(0).to_numpy().reshape(-1, 1)


rts = np.hstack((interpolate_line(rts_x), interpolate_line(rts_y), interpolate_line(rts_z)))
filter = np.hstack((interpolate_line(filter_x), interpolate_line(filter_y), interpolate_line(filter_z)))
odometry = np.hstack((interpolate_line(odometry_x), interpolate_line(odometry_y), interpolate_line(odometry_z)))



print("RMSE of y dir", get_1d_rms_error(interpolate_line(rts_y), interpolate_line(filter_y)))
print("RMSE of x dir", get_1d_rms_error(interpolate_line(rts_x), interpolate_line(filter_x)))
print("RMSE of z dir", get_1d_rms_error(interpolate_line(rts_z), interpolate_line(filter_z)))

pose_error = get_3d_euclidean_norm(rts, filter)
print("test result = ", pose_error)
print("RMSE 3d pose", get_3d_rms_error(pose_error))

get_RT_matrix(ref_matrix=rts, target_matrix= odometry)
    
x = [i for i in range(odometry_y.shape[0])]
figure, axis = plt.subplots(4)
axis[0].plot(x, interpolate_line(odometry_y), color = 'r',  label = 'odometry_y')
axis[0].plot(x, interpolate_line(filter_y), color = 'g', label = 'filter_y')
axis[0].plot(x, interpolate_line(rts_y), color = 'b', label = 'rts_y')
axis[0].set_title("y direction vs frame id")
axis[0].set_xlabel("Frame_id")
axis[0].set_ylabel("y (meters)")

axis[1].plot(x, interpolate_line(odometry_x), color = 'r',  label = 'odometry_x')
axis[1].plot(x, interpolate_line(filter_x), color = 'g', label = 'filter_x')
axis[1].plot(x, interpolate_line(rts_x), color = 'b', label = 'rts_x')
axis[1].set_title("x direction vs frame id")
axis[1].set_xlabel("Frame_id")
axis[1].set_ylabel("x (meters)")

axis[2].plot(x, interpolate_line(odometry_z), color = 'r',  label = 'odometry_z')
axis[2].plot(x, interpolate_line(filter_z), color = 'g', label = 'filter_z')
axis[2].plot(x, interpolate_line(rts_z), color = 'b', label = 'rts_z')
axis[2].set_title("z direction vs frame id")
axis[2].set_xlabel("Frame_id")
axis[2].set_ylabel("z (meters)")

axis[3].plot(x, pose_error, color = 'b',  label = '3d_pose_error')
axis[3].set_title("3d pose error vs frame id")
axis[3].set_xlabel("Frame_id")
axis[3].set_ylabel("Error (meters^3)")
plt.legend()
plt.show()
