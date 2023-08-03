"""
Write a script that read datas from csv file and a transformation matrix from two data.
"""
from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
import numpy.ma as ma
data = pd.read_csv("imu_rts3.csv")
ori_imu= data[['/mti630/imu/orientation/x',
               '/mti630/imu/orientation/y',
               '/mti630/imu/orientation/z',
               '/mti630/imu/orientation/w'
            ]].fillna(0).to_numpy()
ori_rts= data[['/triangulation_2D_pose/pose/pose/orientation/x',
               '/triangulation_2D_pose/pose/pose/orientation/y',
               '/triangulation_2D_pose/pose/pose/orientation/z',
               '/triangulation_2D_pose/pose/pose/orientation/w'
            ]].fillna(0).to_numpy()
# pos_rts= data[['/triangulation_3D_pose/pose/pose/position/x',
#                '/triangulation_3D_pose/pose/pose/position/y',
#                '/triangulation_3D_pose/pose/pose/position/z'
#             ]].fillna(0).to_numpy()


ori_rts_stack = np.eye(3, dtype= float)
ori_imu_stack = np.eye(3, dtype= float)
mask_rts = np.any(ori_rts>0, axis=1).reshape(-1, 1)
mask_imu = np.any(ori_imu>0, axis=1).reshape(-1, 1)
print(np.count_nonzero(mask_imu))
print(np.count_nonzero(mask_rts))
mask = np.logical_and(mask_imu, mask_rts)
print(np.count_nonzero(mask))
rts_euler = np.zeros((1,3))
# for idx,  item in enumerate(ori_rts):
#     if mask_rts[idx]:
#         rts_euler = np.vstack((rts_euler, r.as_euler('zyx', degrees=True)))
# print("mean of angle = ", rts_euler[1:].mean(axis=0))
for idx,  item in enumerate(ori_imu):
    if mask_imu[idx]:
        r = R.from_quat(item)
        ori_imu_stack = np.vstack((ori_imu_stack, r.as_matrix()[0]))
        if mask_rts[idx]:
            r = R.from_quat(item)
            ori_rts_stack = np.vstack((ori_rts_stack, r.as_matrix()[0]))
        else:
            r = R.from_euler('zyx', [[-90, 0, 0]], degrees=True)
            ori_rts_stack = np.vstack((ori_rts_stack, r.as_matrix()[0][0]))
ori_rts_stack= ori_rts_stack[3:][:]
ori_imu_stack= ori_imu_stack[3:][:]
print ("ori_rts_stack size = ", ori_rts_stack.shape)
print ("ori_imu_stack size = ", ori_imu_stack.shape)
# print("test_rts = ", ori_rts_stack[0:3][0:3])
# print("test_imu= ", ori_imu_stack[0:3][0:3])
# print("ori_rts shape", ori_rts.shape)
# print("pos_rts shape", pos_rts.shape)
# print("ori_imu shape", ori_imu.shape)
R_pinv = np.dot(np.linalg.pinv(ori_rts_stack), ori_imu_stack)
print("Pinv R =", R_pinv)
print("p_norm = ", np.linalg.norm(R_pinv))
rotation_matrix, residuals, rank, min_singular_value = np.linalg.lstsq(ori_rts_stack, ori_imu_stack, rcond=None)
reverser_matrix =np.linalg.pinv(rotation_matrix) 
print("Final Rotation matirx = \n", rotation_matrix)
print("norm from rts to imu = ", np.linalg.norm(rotation_matrix))
print("norm from imu to rts = ", np.linalg.norm(np.linalg.pinv(rotation_matrix)))
print("Residuals = ", residuals)
print("Rank = ", rank)
print("min_singular_value = ", min_singular_value)


r = R.from_matrix(rotation_matrix)
r_v = R.from_matrix(reverser_matrix)
print("Fincal rotation degrees = ", r.as_euler('zyx', degrees=True))
print("Fincal rotation degrees = ", r_v.as_euler('zyx', degrees=True))
print("Fincal rotation vector = ",r.as_rotvec())
print("Fincal rotation vector = ",r_v.as_rotvec())