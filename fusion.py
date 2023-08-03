"""
Write a script that reads the IMU data from the rosbag and calculates 6 dof pose using the IMU data.
The IMU data is in the form of orientation, angular velocity, and linear acceleration.
"""
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np  
import matplotlib.pyplot as plt

# create a no bound 2d array to store the data
ori = np.empty((0, 4), float)
ori_cov = np.empty((0, 9), float)
ang_vel = np.empty((0, 3), float)
ang_vel_cov = np.empty((0, 9), float)
lin_acc = np.empty((0, 3), float)
lin_acc_cov = np.empty((0, 9), float)
TimeStamps = []

# RST pose initilization
rts_pose = np.empty((0, 3), float)
rts_pose_cov = np.empty((0, 9), float)
rts_ang_vel = np.empty((0, 3), float)
rts_ang_vel_cov = np.empty((0, 9), float)
rts_lin_acc = np.empty((0, 3), float)
rts_lin_acc_cov = np.empty((0, 9), float)
rts_TimeStamps = []

# create reader instance and open for reading
with Reader('/home/michael.lin/Desktop/imu_fusion/gp_monticello_18_44_07') as reader:
# with Reader('/home/michael.lin/Desktop/imu_fusion/gp_cedar_springs_run56') as reader:
    # topic and msgtype information is available on .connections list
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/mti630/imu':
            TimeStamps.append(timestamp)
            # deserialize the message
            msg = deserialize_cdr(rawdata, connection.msgtype)
            # print("msg info = ", msg.header.frame_id)
            
            ori = np.append(
                        ori, 
                        np.array(
                            [msg.orientation.x, 
                                msg.orientation.y, 
                                msg.orientation.z, 
                                msg.orientation.w]).reshape(1,ori.shape[1]),
                        axis=0
                        )
            ori_cov = np.append(
                            ori_cov, 
                            msg.orientation_covariance.reshape(1,ori_cov.shape[1]),
                            axis=0
                            )
            ang_vel = np.append(
                        ang_vel, 
                        np.array([
                                msg.angular_velocity.x, 
                                msg.angular_velocity.y, 
                                msg.angular_velocity.z, 
                                ]).reshape(1,ang_vel.shape[1]),
                        axis=0
                        )
            ang_vel_cov = np.append(
                            ang_vel_cov, 
                            msg.angular_velocity_covariance.reshape(1,ang_vel_cov.shape[1]),
                            axis=0
                            )
            lin_acc = np.append(
                        lin_acc, 
                        np.array([
                                msg.linear_acceleration.x, 
                                msg.linear_acceleration.y, 
                                msg.linear_acceleration.z, 
                                ]).reshape(1,lin_acc.shape[1]),
                        axis=0
                        )
            lin_acc_cov = np.append(
                            lin_acc_cov, 
                            msg.linear_acceleration_covariance.reshape(1,lin_acc_cov.shape[1]),
                            axis=0
                            )
        if connection.topic == '/trimble_prism':
            # TimeStamps.append(timestamp)
            # deserialize the message
            msg = deserialize_cdr(rawdata, connection.msgtype)
            # # print("msg    info = ", msg[0].header.frame_id)
            # print("msg    info = ", msg[0].header.stamp)
            # print("msg    info = ", msg[0].header.seq)
            rts_pose = np.append(
                        ori, 
                        np.array(
                            [msg.pose.pose.orientation.x, 
                                msg.pose.pose.orientation.y, 
                                msg.pose.pose.orientation.z]).reshape(1,ori.shape[1]),
                        axis=0
                        )
            rts_pose_cov = np.append(
                            ori_cov, 
                            msg.orientation_covariance.reshape(1,ori_cov.shape[1]),
                            axis=0
                            )
            print("msg    info = ", msg[0].pose.pose.position.x)
            print("msg    info = ", msg[0].pose.pose.position.y)
            print("msg    info = ", msg[0].pose.pose.position.z)
            print("msg    info = ", msg[0].pose.pose.orientation.x)
            print("msg    info = ", msg[0].pose.pose.orientation.y)
            print("msg    info = ", msg[0].pose.pose.orientation.z)
            print("msg    info = ", msg[0].pose.pose.orientation.w)
            print("msg    info = ", msg[0].pose.covariance)

            
            



            

            
#create three plot for each axis
print ("imu shape = ", ori.shape)
print ("imu shape = ", ori_cov.shape)
print ("imu shape = ", ang_vel.shape)
print ("imu shape = ", ang_vel_cov.shape)
print ("imu shape = ", lin_acc.shape)
print ("imu shape = ", lin_acc_cov.shape)
print ("total sensing freq = ", (TimeStamps[-1] - TimeStamps[0])/len(TimeStamps))

data_length = np.arange(0, ori.shape[0])
fig, orientation_axs = plt.subplots(4, 1)
orientation_axs[0].plot(data_length, ori[:, 0], label='x')
orientation_axs[1].plot(data_length, ori[:, 1], label='y')
orientation_axs[2].plot(data_length, ori[:, 2], label='z')
orientation_axs[3].plot(data_length, ori[:, 3], label='w')
orientation_axs[0].set_title('Orientation')
orientation_axs[0].set_ylabel('x')
orientation_axs[1].set_ylabel('y')
orientation_axs[2].set_ylabel('z')
orientation_axs[3].set_ylabel('w')
orientation_axs[3].set_xlabel('time')
orientation_axs[0].legend()
orientation_axs[1].legend()
orientation_axs[2].legend()
orientation_axs[3].legend()
plt.show()

# fig, orientation_axs = plt.subplots(3, 1)
# orientation_axs[0].plot(data_length, orientation[:, 0], label='x')
# orientation_axs[1].plot(data_length, orientation[:, 1], label='y')
# orientation_axs[2].plot(data_length, orientation[:, 2], label='z')
# orientation_axs[0].set_title('Orientation')
# orientation_axs[0].set_ylabel('x')
# orientation_axs[1].set_ylabel('y')
# orientation_axs[2].set_ylabel('z')
# orientation_axs[2].set_xlabel('time')
# orientation_axs[0].legend()
# orientation_axs[1].legend()
# orientation_axs[2].legend()
# plt.show()
            

# print("msg info = ", msg[0].header.frame_id)
# print("msg info = ", msg[0].header.stamp) 
# print("msg info = ", msg[0].header.seq)   
# print("msg info = ", msg[0].orientation.x)
# print("msg info = ", msg[0].orientation.y)
# print("msg info = ", msg[0].orientation.z)
# print("msg info = ", msg[0].orientation.w)
# print("msg info = ", msg[0].angular_velocity.x)
# print("msg info = ", msg[0].angular_velocity.y)
# print("msg info = ", msg[0].angular_velocity.z)
# print("msg info = ", msg[0].linear_acceleration.x)
# print("msg info = ", msg[0].linear_acceleration.y)
# print("msg info = ", msg[0].linear_acceleration.z)
# print("msg info = ", msg[0].orientation_covariance)
# print("msg info = ", msg[0].angular_velocity_covariance)
# print("msg info = ", msg[0].linear_acceleration_covariance)
# print("msg info = ", msg[0].orientation_covariance[0])
# print("msg info = ", msg[0].orientation_covariance[1])
# print("msg info = ", msg[0].orientation_covariance[2])
# print("msg info = ", msg[0].orientation_covariance[3])
# print("msg info = ", msg[0].orientation_covariance[4])
# print("msg info = ", msg[0].orientation_covariance[5])
# print("msg info = ", msg[0].orientation_covariance[6])
# print("msg info = ", msg[0].orientation_covariance[7])
# print("msg info = ", msg[0].orientation_covariance[8])
# print("msg info = ", msg[0].angular_velocity_covariance[0])
# print("msg info = ", msg[0].angular_velocity_covariance[1])
# print("msg info = ", msg[0].angular_velocity_covariance[2])
# print("msg info = ", msg[0].angular_velocity_covariance[3])
# print("msg info = ", msg[0].angular_velocity_covariance[4])
# print("msg info = ", msg[0].angular_velocity_covariance[5])
# print("msg info = ", msg[0].angular_velocity_covariance[6])
# print("msg info = ", msg[0].angular_velocity_covariance[7])
# print("msg info = ", msg[0].angular_velocity_covariance[8])



