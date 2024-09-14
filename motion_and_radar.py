import numpy as np

wheel_radius = 0.1
ROBOT_WIDTH = 0.2
ROBOT_LENGTH = 0.4
dt = 0.1


def diff_kinematics(x: np.ndarray, u: np.ndarray):
    u = np.clip(u, 0.1, 0.5)

    left_motion = u[0, 0]
    right_motion = u[1, 0]

    if u[2, 0] != 0:
        left_motion = (u[0, 0] + u[2, 0]) / 2
    if u[3, 0] != 0:
        right_motion = (u[1, 0] + u[3, 0]) / 2

    stand = wheel_radius / (2 * ROBOT_WIDTH)
    omega = (right_motion - left_motion) * stand
    vel = (left_motion + right_motion) * stand * ROBOT_WIDTH

    x_dot = np.array([[omega],
                      [vel * np.cos(x[0, 0])],
                      [vel * np.sin(x[0, 0])]])

    return x_dot


def f_rk4_one_step(x0, u0):
    f1 = diff_kinematics(x0, u0)
    f2 = diff_kinematics(x0 + 0.5 * dt * f1, u0)
    f3 = diff_kinematics(x0 + 0.5 * dt * f2, u0)
    f4 = diff_kinematics(x0 + dt * f3, u0)
    x_pos = x0 + dt * (f1 + 2. * f2 + 2. * f3 + f4) / 6.

    return x_pos


def sign(point1, point2, point3):
    return (point1[0] - point3[0]) * (point2[1] - point3[1]) - (point2[0] - point3[0]) * (point1[1] - point3[1])


def is_point_in_triangle(coordination, triangle_v1, triangle_v2, triangle_v3):
    d1 = sign(coordination, triangle_v1, triangle_v2)
    d2 = sign(coordination, triangle_v2, triangle_v3)
    d3 = sign(coordination, triangle_v3, triangle_v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def radar_detection(x, landmark_list, f_point, l_point, r_point, scale):
    detects = []

    pos_x = x[1, 0]
    pos_y = x[2, 0]

    robot_angle = x[0, 0]
    for i, landmark_name in enumerate(landmark_list):

        noise = np.random.uniform(low=-0.005, high=0.005, size=1)
        lx = float(np.add(landmark_name[0], noise))
        ly = float(np.add(landmark_name[1], noise))

        xx = lx - pos_x
        yy = ly - pos_y

        landmark_angle = np.arctan2(yy, xx)
        angle_diff = landmark_angle - robot_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        if is_point_in_triangle((400 + lx * scale, 300 - ly * scale), f_point, l_point, r_point):
            distance = np.sqrt(xx ** 2 + yy ** 2)
            detects.append((distance, angle_diff, i + 1))
    return detects
