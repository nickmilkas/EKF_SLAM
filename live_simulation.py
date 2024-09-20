import pygame
from ekf_slam import *


# landmarks = np.array([[3.4, 3.6], [-3.6, 3.7], [3.3, -3.7], [-3.5, -3], [-3, 3]])

# Random landmark coordinates
coord1 = float(np.random.uniform(low=3, high=4.5, size=1))
coord2 = float(np.random.uniform(low=3, high=4.5, size=1))
coord3 = float(np.random.uniform(low=3, high=4.5, size=1))
coord4 = float(np.random.uniform(low=3, high=4.5, size=1))
landmarks = np.array([[3.4, coord1], [-3.6, -coord2], [3.3, -coord3], [-3.5, coord4], [-3, 1]])

color = 'aqua'
colors = [color] * len(landmarks)
dt = 0.1
X_BOUND = 800
Y_BOUND = 600
LANDMARK_RADIUS = 9
ROBOT_WIDTH = 0.2
ROBOT_LENGTH = 0.4
UNSEEN = 10e6
scale = 50
laser_length = 250
laser_width = 60
max_trace_length = 50
trace_interval = 120

association_threshold = 6
R = np.eye(3) * (0.1 / (scale ** 2))
Q = np.diag([0.02 / (scale ** 2), 0.05 / (scale ** 2)])


def run_simulation():
    # This is the main function that runs the simulation. It calls all the functions and create their path

    # -------------INITIAL SETUP-------------
    x = np.array([[0.0], [0.0], [0.0]])
    u_initial = np.array([[0.3], [0.3], [0.0], [0.0]])
    u = u_initial
    u_bad = u_initial
    odom_p = x

    # -------------EKF SLAM INITIALIZATION----------
    st = np.zeros((3, 1))  # Initial state
    st[:3, :] = np.copy(x)  # Robot's initial position
    sig = np.eye(3) * 0.001  # Initial uncertainty
    estimation = [(np.copy(st), sig)]

    # -------------PYGAME INITIALIZATION-------------
    pygame.init()
    screen = pygame.display.set_mode((X_BOUND, Y_BOUND))
    pygame.display.set_caption("Differential Drive Robot")
    clock = pygame.time.Clock()

    # -------------INITIAL TRACES-------------
    trace_points = [(400 + x[1, 0] * scale, 300 - x[2, 0] * scale, x[0, 0])]
    all_trace_points = [(400 + x[1, 0] * scale, 300 - x[2, 0] * scale, x[0, 0])]

    odom_trace_points = [(400 + odom_p[1, 0] * scale, 300 - odom_p[2, 0] * scale, odom_p[0, 0])]
    odom_all_points = [(400 + odom_p[1, 0] * scale, 300 - odom_p[2, 0] * scale, odom_p[0, 0])]

    ekf_trace_points = [(400 + st[1, 0] * scale, 300 - st[2, 0] * scale, st[0, 0])]
    all_ekf_trace_points = [(400 + st[1, 0] * scale, 300 - st[2, 0] * scale, st[0, 0])]

    step_counter = 0
    running = True

    while running:
        # -------------EVENT HANDLING-------------

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                noise = np.random.uniform(low=-0.02, high=0.02, size=(4, 1))
                if event.key == pygame.K_LEFT:
                    u = np.array([[0.12], [0.25], [0.0], [0.0]])
                    u_bad = np.add(u, noise)
                elif event.key == pygame.K_RIGHT:
                    u = np.array([[0.25], [0.12], [0.0], [0.0]])
                    u_bad = np.add(u, noise)
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    noise_in_straight_motion = np.random.uniform(low=-0.005, high=0.005, size=(4, 1))
                    u = u_initial
                    u_bad = np.add(u_initial, noise_in_straight_motion)

        # -------------STATE UPDATES-------------
        x_next = f_rk4_one_step(x, u_bad)
        x = x_next
        odom_p_next = update_odometry_rk4(odom_p, u)
        odom_p = odom_p_next

        # print("Real position: ", np.round(x, 4))

        screen.fill((255, 255, 255))

        # -------------ROBOT RENDERING-------------
        pos_x = int(400 + x[1, 0] * scale)
        pos_y = int(300 - x[2, 0] * scale)

        pos_x = max(0, min(X_BOUND, pos_x))
        pos_y = max(0, min(Y_BOUND, pos_y))

        odom_pos_x = int(400 + odom_p[1, 0] * scale)
        odom_pos_y = int(300 - odom_p[2, 0] * scale)

        robot_width_scaled = int(ROBOT_WIDTH * scale)
        robot_length_scaled = int(ROBOT_LENGTH * scale)

        rect_corners = find_corners(pos_x, pos_y, robot_length_scaled, robot_width_scaled, x[0][0])
        pygame.draw.polygon(screen, 'black', rect_corners, 0)

        # -------------TRACING ROBOT'S PATH-------------
        step_counter += 1
        if step_counter % trace_interval == 0:
            # trace_points.append((pos_x, pos_y, x[0, 0]))
            all_trace_points.append((pos_x, pos_y, x[0, 0]))
            odom_trace_points.append((odom_pos_x, odom_pos_y, odom_p[0, 0]))
            odom_all_points.append((odom_pos_x, odom_pos_y, odom_p[0, 0]))

            if len(trace_points) > max_trace_length:
                trace_points.pop(0)
            if len(odom_trace_points) > max_trace_length:
                odom_trace_points.pop(0)
            if len(ekf_trace_points) > max_trace_length:
                ekf_trace_points.pop(0)

        for trace_x, trace_y, trace_theta in all_trace_points:
            trace_width_scaled = int(ROBOT_WIDTH * scale * 0.5)
            trace_length_scaled = int(ROBOT_LENGTH * scale * 0.5)
            trace_corners = find_corners(trace_x, trace_y, trace_length_scaled, trace_width_scaled, trace_theta)
            pygame.draw.polygon(screen, 'red', trace_corners, 0)

        for odom_x, odom_y, odom_theta in odom_all_points:
            odom_width_scaled = int(ROBOT_WIDTH * scale * 0.5)
            odom_length_scaled = int(ROBOT_LENGTH * scale * 0.5)
            odom_corners = find_corners(odom_x, odom_y, odom_length_scaled, odom_width_scaled, odom_theta)
            pygame.draw.polygon(screen, 'blue', odom_corners, 0)

        # -------------RADAR TRIANGLE-------------
        end_x = pos_x + int(laser_length * np.cos(x[0, 0]))
        end_y = pos_y - int(laser_length * np.sin(x[0, 0]))

        front_point = (pos_x, pos_y)
        left_point = (end_x + int(laser_width * np.sin(x[0, 0])), end_y + int(laser_width * np.cos(x[0, 0])))
        right_point = (end_x - int(laser_width * np.sin(x[0, 0])), end_y - int(laser_width * np.cos(x[0, 0])))

        pygame.draw.polygon(screen, (0, 255, 0), [front_point, left_point, right_point], 2)

        # -------------EKF SLAM PROCESSING-------------
        if len(odom_all_points) > 1:

            z = radar_detection(x, landmarks, front_point, left_point, right_point, scale)
            mu, sig = estimation[-1]

            mu_new, sig_new = ekf_slam_step(mu, sig, u_bad, z, R, Q, association_threshold)

            ekf_trace_points.append((400 + mu_new[1, 0] * scale, 300 - mu_new[2, 0] * scale, mu_new[0, 0]))
            all_ekf_trace_points.append((400 + mu_new[1, 0] * scale, 300 - mu_new[2, 0] * scale, mu_new[0, 0]))
            estimation.append((mu_new, sig_new))

            num_landmarks = (len(mu_new) - 3) // 2
            for i in range(num_landmarks):
                l_idx = 3 + 2 * i
                lx = 400 + mu_new[l_idx, 0] * scale
                ly = 300 - mu_new[l_idx + 1, 0] * scale
                pygame.draw.circle(screen, 'slateblue', (float(lx), float(ly)), LANDMARK_RADIUS)

        for ekf_x, ekf_y, ekf_theta in all_ekf_trace_points:
            ekf_width_scaled = int(ROBOT_WIDTH * scale * 0.5)
            ekf_length_scaled = int(ROBOT_LENGTH * scale * 0.5)
            ekf_corners = find_corners(ekf_x, ekf_y, ekf_length_scaled, ekf_width_scaled, ekf_theta)
            # pygame.draw.polygon(screen, 'purple', ekf_corners, 0)

        # -------------LANDMARK HANDLING-------------
        for i, landmark in enumerate(landmarks):
            lx = int(400 + landmark[0] * scale)
            ly = int(300 - landmark[1] * scale)
            pygame.draw.circle(screen, pygame.Color(colors[i % len(colors)]), (lx, ly), LANDMARK_RADIUS)

            if is_point_in_triangle((lx, ly), front_point, left_point, right_point):
                pygame.draw.circle(screen, 'orangered', (lx, ly), LANDMARK_RADIUS + 1, 2)

        # -------------DISPLAY UPDATE-------------
        if step_counter % 10 == 0:
            pygame.display.flip()

        clock.tick(300)

    pygame.quit()
    mu_final, _ = estimation[-1]
    create_observation_map(all_trace_points, odom_all_points, all_ekf_trace_points, mu_final, scale)


def find_corners(x_value, y_value, length_scaled, width_scaled, theta):
    corners = [
        (x_value + length_scaled * np.cos(theta) - width_scaled * np.sin(theta) / 2,
         y_value - length_scaled * np.sin(theta) - width_scaled * np.cos(theta) / 2),
        (x_value + length_scaled * np.cos(theta) + width_scaled * np.sin(theta) / 2,
         y_value - length_scaled * np.sin(theta) + width_scaled * np.cos(theta) / 2),
        (x_value - length_scaled * np.cos(theta) + width_scaled * np.sin(theta) / 2,
         y_value + length_scaled * np.sin(theta) + width_scaled * np.cos(theta) / 2),
        (x_value - length_scaled * np.cos(theta) - width_scaled * np.sin(theta) / 2,
         y_value + length_scaled * np.sin(theta) - width_scaled * np.cos(theta) / 2)
    ]
    return corners
