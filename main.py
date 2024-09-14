from live_simulation import *
from general_functions import *

# Initial state: robot at origin, facing along x-axis
x = np.array([[1.0], [1.0], [0.0]])

Vb = np.array([[0.0],
               [1.0],
               [0.3]])

u = np.array([[0.2], [0.2], [0], [0]])

# Previous odometry state (initial state)
odom_previous = np.array([[0.0],  # Theta
                          [0.3],  # x-position
                          [1.0]])  # y-position

theta_test = np.pi / 6  # 30 degrees in radians

mu = np.array([[0.0], [2.0], [1.0], [5.2], [3.1]])

if __name__ == '__main__':
    run_simulation()
    # print(np.round(motion_model(odom_previous, u), 3))

    # simulation_motion()
    # u = np.array([[0.0], [0.0]])
    # x = np.array([[0.7], [0.1], [0.0]])
    # diff_kinematics(x, u)
    # H = compute_H(theta_test)
    # print(integrate(p, Vb))
    # print(odometer_calculation(u_speed, odom_previous))
    # test_this()
    # print(H_t_low(mu, 1))
    # print(Fxj(1, 2))
