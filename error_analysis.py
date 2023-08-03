import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Contains functions to calculate the relative and absolute orientation error
# between the estimated orientation and the true orientation

def calc_relative_orientation_difference(estimated_rotation, true_rotation, window_size=0):
    '''
    Calculate the rotation matrix between 
        the estimated rotation between the current orientation and the orientation [window_size] indices ago, and
        the true      rotation between the current orientation and the orientation [window_size] indices ago
    
    Args:
        estimated_rotation - Rotation object from scipy.spatial.transform.Rotation
        true_rotation - Rotation object from scipy.spatial.transform.Rotation
    Returns:
        diff - Rotation object from scipy.spatial.transform.Rotation
    '''
    if window_size < 1:
        estimated_start = estimated_rotation[0]
        true_start = true_rotation[0]
        estimated_R12 = estimated_rotation.inv() * estimated_start
        true_R12 = true_rotation.inv() * true_start
        diff = estimated_R12.inv() * true_R12
    else:
        estimated_R12 = estimated_rotation[window_size:].inv() * estimated_rotation[:-window_size]
        true_R12 = true_rotation[window_size:].inv() * true_rotation[:-window_size]
        diff = estimated_R12.inv() * true_R12
    
    return diff

def calc_relative_orientation_error(estimated_rotation, true_rotation, window_size=2):
    '''
    Calculate the angular difference between 
        the estimated rotation between the current orientation and the orientation [window_size] indices ago, and
        the true      rotation between the current orientation and the orientation [window_size] indices ago
    Both the estimated and true rotations are given as Rotation objects from scipy.spatial.transform.Rotation
    '''
    diff = calc_relative_orientation_difference(estimated_rotation, true_rotation, window_size=window_size)
    angle = np.arccos((np.trace(diff.as_matrix(), axis1=1, axis2=2) - 1.0)/2.0)
    return angle

def calulate_absolute_orientation_error(estimated_rotation, true_rotation):
    '''
    Calculate the angular difference between 
        the estimated rotation between the current orientation and the starting orientation, and
        the true      rotation between the current orientation and the starting orientation
    Both the estimated and true rotations are given as Rotation objects from scipy.spatial.transform.Rotation
    '''
    diff = calc_relative_orientation_difference(estimated_rotation, true_rotation, window_size=0)
    angle = np.arccos((np.trace(diff.as_matrix(), axis1=1, axis2=2) - 1.0)/2.0)
    return angle

def calculate_relative_yaw_error(estimated_rotation, true_rotation, window_size=2):
    '''
    Calculate the angular difference between 
        the estimated yaw between the current orientation and the orientation [window_size] indices ago, and
        the true      yaw between the current orientation and the orientation [window_size] indices ago
    Both the estimated and true rotations are given as Rotation objects from scipy.spatial.transform.Rotation
    TODO: This does not "unwrap" the robot's frame onto a plane, but it is ok for large radius assets
    '''
    diff = calc_relative_orientation_difference(estimated_rotation, true_rotation, window_size=window_size)

    diff_euler = diff.as_euler('xyz', degrees=False)
    yaw_error = diff_euler[:,2]
    return yaw_error

def calculate_absolute_yaw_error(estimated_rotation, true_rotation):
    diff = calc_relative_orientation_difference(estimated_rotation, true_rotation, window_size=0)
    diff_euler = diff.as_euler('xyz', degrees=False)
    yaw_error = diff_euler[:,2]
    return yaw_error


def main():
    bagfile_name = r'rosbag2_2023_07_11-19_34_40'
    # bagfile_timestamp = bagfile_name[8:]
    # input_file = r'processed_data/processed_vo_api_' + bagfile_timestamp + r'.csv'
    
    input_file = r'processed_data/processed_vo_api_08_01-2023_07_11-19_34_40.csv'
    df = pd.read_csv(input_file)

    # Fill in all missing values with the previous value, and the first value with the next value
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    true_rotation = R.from_quat(df[['api.rotation.x', 'api.rotation.y', 'api.rotation.z', 'api.rotation.w']].to_numpy())
    estimated_rotation = R.from_quat(df[['vo.rotation.x', 'vo.rotation.y', 'vo.rotation.z', 'vo.rotation.w']].to_numpy())

    window_size = 20
    angle = calc_relative_orientation_error(estimated_rotation, true_rotation, window_size)
    avg_angle = np.mean(angle)
    time = df['ros.seconds'].to_numpy()[window_size:]

    plt.figure()
    plt.plot(time, angle)
    plt.plot(time, np.ones(len(time)) * avg_angle, 'r--')
    plt.title('Relative orientation error: ' + bagfile_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend([f'Angle, window_size={window_size}', f'Average Angle difference: {avg_angle:.3f} rad'])

    # plot the absolute orientation error
    angle = calulate_absolute_orientation_error(estimated_rotation, true_rotation)
    time = df['ros.seconds'].to_numpy()
    plt.figure()
    plt.plot(time, angle)
    plt.title('Absolute orientation error: ' + bagfile_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend(['Angle'])

    # plot the relative yaw error
    yaw_error = calculate_relative_yaw_error(estimated_rotation, true_rotation, window_size)
    avg_yaw_error = np.mean(yaw_error)
    time = df['ros.seconds'].to_numpy()[window_size:]
    plt.figure()
    plt.plot(time, yaw_error)
    plt.plot(time, np.ones(len(time)) * avg_yaw_error, 'r--')
    plt.title('Relative yaw error: ' + bagfile_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend([f'Angle, window_size={window_size}', f'Average Angle difference: {avg_yaw_error:.3f} rad'])

    # plot the absolute yaw error
    yaw_error = calculate_absolute_yaw_error(estimated_rotation, true_rotation)
    time = df['ros.seconds'].to_numpy()
    plt.figure()
    plt.plot(time, yaw_error)
    plt.title('Absolute yaw error: ' + bagfile_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend(['Angle'])


    plt.show()


if __name__ == '__main__':
    main()
