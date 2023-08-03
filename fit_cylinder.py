import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import pandas as pd

###################
# Fit a cylinder to the (x,y,z) points
# The triangle_positioning_node.py and triangulation_trimble_positioning_params.yaml define the curvature to be
#    # x_curvature: 0.0729 # for the ops tank - 0.285

# TODO: Fit a 3D cylinder with variable center, radius, and orientation
def point_to_line_distance(params, points):
    """
    Calculate the perpendicular distance from a point (vectorized) to the line defined by the parameters (axis of the cylinder)
    """
    x0, y0, theta_x, theta_y, _ = params
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    # print(params)
    axis_vector = np.array([[np.cos(theta_y)*np.cos(theta_x), np.cos(theta_y)*np.sin(theta_x), np.sin(theta_y)]])
    center_to_pt_vector = np.array([x-x0, y-y0, z]).T
    projection_on_axis = axis_vector * (axis_vector @ center_to_pt_vector.T).T + np.array([[x0, y0, 0]])
    perpendicular_distance = np.linalg.norm(center_to_pt_vector - projection_on_axis, axis=1)
    # print(f"perpendicular_distance: {perpendicular_distance.shape}")
    return perpendicular_distance

def fit_cylinder_to_points(points, initial_parameters, max_iterations=1000):
    """
    points: np.array of dimension (N,3), storing N>5 rows of x,y,z coordinates
    initial_parameters: (5,) tuple of initial parameters for the least squares algorithm
        initial_parameters[0] = x coordinate of the cylinder center
        initial_parameters[1] = y coordinate of the cylinder center
        initial_parameters[2] = radius of the cylinder
        initial_parameters[3] = rotation angle (radian) about the x-axis
        initial_parameters[4] = rotation angle (radian) about the y-axis
    max_iterations: maximum number of iterations for the least squares algorithm

    Idea: 
        Given parameters[0:3] and the points, calculate the average distance from the points to the axis of the cylinder
        Error = (average radial distance)**2 - (parameters[4] (ie. expected radius))**2
        scipy.optimize.leastsq will guess a new set of parpameters to minimize the error function
        stop when the average radial distance is within a threshold of the expected radius
    """ 
    
    # want to minimize the difference between the estimated radius**2 and the actual radius**2
    error_function = lambda params, points: (
        np.abs(point_to_line_distance(params, points) - params[2])
        # points_to_circle_distance_squared(params, points) - params[2]**2
    )

    estimated_parameters , success = leastsq(error_function, initial_parameters, args=(points), maxfev=max_iterations)

    if not success:
        print("Failed to fit a cylinder to the points")
    return estimated_parameters

def get_initial_cylinder_params(points):
    pass


# Fit a circle to the position points
def points_to_circle_distance_squared(params, points):
    """
    Calculate the distance on the x,y plane between a point (vectorized) and the center of the circle defined by the parameters (axis of the cylinder)

    """
    x = points[:,0]
    y = points[:,1]
    xc = params[0]
    yc = params[1]
    return (x-xc)**2 + (y-yc)**2

def fit_circle_to_points(points, initial_parameters, param_radius, max_iterations=1000, tolerance=1.0e-37):
    """
    points: np.array of dimension (N,3), storing N>5 rows of x,y,z coordinates
    initial_parameters: (5,) tuple of initial parameters for the least squares algorithm
        initial_parameters[0] = x coordinate of the cylinder center
        initial_parameters[1] = y coordinate of the cylinder center
        initial_parameters[2] = radius of the cylinder
    max_iterations: maximum number of iterations for the least squares algorithm

    Idea: 
        Given parameters[0:3] and the points, calculate the average distance from the points to the axis of the cylinder
        Error = (average radial distance)**2 - (parameters[4] (ie. expected radius))**2
        scipy.optimize.leastsq will guess a new set of parpameters to minimize the error function
        stop when the average radial distance is within a threshold of the expected radius
    """ 
    error_function = lambda params, points: (
        np.abs(points_to_circle_distance_squared(params[:2], points) - param_radius**2)
    )

    estimated_parameters , success = leastsq(error_function, initial_parameters, args=(points), maxfev=max_iterations, ftol=tolerance)

    if not success:
        print("Failed to fit a circle to the points")
    else:
        print(f"Estimated Circle Parameters: x0: {estimated_parameters[0]}, y0: {estimated_parameters[1]}")#, radius: {estimated_parameters[2]}")
    
    return estimated_parameters

def get_initial_circle_params(points, param_radius=13717):
    # get the initial guess for the center and radius of the best-fit circle on the x-y plane
    # First get the "tangent" line by fitting a line to the (x,y) points
    # Then get the center of the circle by taking the average of the points and adding the radius in the direction perpendicular to the tangent line

    x_data = points[:,0]
    y_data = points[:,1]
    # remove nan values
    x_data = x_data[~np.isnan(x_data)]
    y_data = y_data[~np.isnan(y_data)]

    tangent_line_coeffs = np.polyfit(x_data, y_data, 1)
    # # get the perpendicular normal vector by taking the reciprocal of the slope and flipping the sign
    perp_normal_vector = np.array([1, 1/tangent_line_coeffs[0]])
    perp_normal_vector = perp_normal_vector / np.linalg.norm(perp_normal_vector)
    # # get the center of the circle
    point0 = np.array([x_data.mean(), y_data.mean()])
    center = point0 + param_radius * perp_normal_vector
    # print(f"tangent_line_coeffs: {tangent_line_coeffs}")
    # print(f"perp_normal_vector: {perp_normal_vector}")
    # print(f"point0: {point0}")
    # print(f"center: {center}")

    param_x0 = center[0] #df['api.position.x'].mean() + param_radius/np.sqrt(2) * np.sign(df['api.position.x'].mean())
    param_y0 = center[1] #df['api.position.y'].mean() + param_radius/np.sqrt(2) * np.sign(df['api.position.y'].mean()) 
    param_theta_x = 0
    param_theta_y = 0
    print(f"Initial Cylinder Parameters: x0: {param_x0}, y0: {param_y0}")
    print(f"Hardcoded Cylinder parameters: radius: {param_radius}, theta_x: {param_theta_x}, theta_y: {param_theta_y}")
    params = (param_x0, param_y0, param_radius, param_theta_x, param_theta_y)

    return params

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid
    
def main():
    input_file = r'data/vo_api_2023_07_11-19_34_40.csv'
    output_file = r'processed_data/processed_vo_api_2023_07_11-19_34_40.csv'

    df = pd.read_csv(input_file)

    # rename the header to something more readable
    df = df.rename(columns={'/api_radian/pose/pose/position/x': 'api.position.x', 
                            '/api_radian/pose/pose/position/y': 'api.position.y',
                            '/api_radian/pose/pose/position/z': 'api.position.z',
                            '/tf/vo_base/vo_camera/rotation/w': 'vo.rotation.w',
                            '/tf/vo_base/vo_camera/rotation/x': 'vo.rotation.x',
                            '/tf/vo_base/vo_camera/rotation/y': 'vo.rotation.y',
                            '/tf/vo_base/vo_camera/rotation/z': 'vo.rotation.z'
                            })

    # Remove duplicate rows (where timestamps from sensors are the same)
    df = df.drop_duplicates(subset=['/api_radian/pose/header/stamp', '/tf/vo_base/vo_camera/header/stamp'], ignore_index=True)

    # Interpolate the API positions to fill in the holes where the API data was taken,
    # and also the larger gaps where API lost line-of-sight
    df['api.position.x'] = df['api.position.x'].interpolate(method='cubic')
    df['api.position.y'] = df['api.position.y'].interpolate(method='cubic')
    df['api.position.z'] = df['api.position.z'].interpolate(method='cubic')

    # Fill in all missing values with the previous value, and the first value with the next value
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    # Find the circle that best fits the points
    points = df[['api.position.x', 'api.position.y', 'api.position.z']].to_numpy()
    param_radius = 13717 #mm
    initial_parameters = get_initial_circle_params(points, param_radius=param_radius)
    estimated_parameters_no_r = fit_circle_to_points(points, initial_parameters[:2],param_radius=initial_parameters[2] )
    estimated_parameters = (estimated_parameters_no_r[0], estimated_parameters_no_r[1], initial_parameters[2], initial_parameters[3], initial_parameters[4])



    # Plot the points and the cylinder
    fig3 = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.scatter3D(df['api.position.x'], df['api.position.y'], df['api.position.z'], c=df['__time'], cmap='Greens')
    ax3.set_title('Points and Cylinder')
    ax3.set_box_aspect((1,1,1))
    ax3.grid(False)

    cylinder_z_height = 4000
    Xc,Yc,Zc = data_for_cylinder_along_z(estimated_parameters[0], estimated_parameters[1], estimated_parameters[2], cylinder_z_height)
    ax3.plot_surface(Xc, Yc, Zc, alpha=0.5)

    # # set x limits and y limits
    ax3.set_xlim([-1800,-1300])
    ax3.set_ylim([600,1100])

    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_zlabel('z (mm)')

    plt.show()


if __name__ == '__main__':
    main()
