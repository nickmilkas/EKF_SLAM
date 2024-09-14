from motion_and_radar import *
import matplotlib.pyplot as plt

X_BOUND = 800
Y_BOUND = 600

wheel_radius = 0.1
ROBOT_WIDTH = 0.2
ROBOT_LENGTH = 0.4
dt = 0.1

positions = [(ROBOT_LENGTH, ROBOT_WIDTH), (ROBOT_LENGTH, -ROBOT_WIDTH),
             (-ROBOT_LENGTH, -ROBOT_WIDTH), (-ROBOT_LENGTH, ROBOT_WIDTH)]
gammas = [-np.pi / 4., np.pi / 4., -np.pi / 4., np.pi / 4.]
betas = [0., 0., 0., 0.]


def integrate(p, Vb):
    # Rotate Vb to Vw
    R = np.array([[1., 0., 0.], [0., np.cos(p[0, 0]), -np.sin(p[0, 0])], [0., np.sin(p[0, 0]), np.cos(p[0, 0])]])
    if abs(Vb[0, 0]) < 1e-6:
        return p + R @ Vb * dt
    else:
        omega = Vb[0, 0]
        vx = Vb[1, 0]
        vy = Vb[2, 0]
        dp = np.array([[omega], [(vx * np.sin(omega) + vy * (np.cos(omega) - 1.)) / omega],
                       [(vy * np.sin(omega) + vx * (1. - np.cos(omega))) / omega]])
        return p + R @ dp * dt


def compute_H(theta):
    m = len(positions)  # number of wheels
    h_all = np.zeros((m, 3))
    for i in range(m):
        prefix = 1. / (wheel_radius * np.cos(gammas[i]))
        x_i, y_i = positions[i]
        s1 = np.sin(betas[i] + gammas[i])
        c1 = np.cos(betas[i] + gammas[i])
        s2 = np.sin(betas[i] + gammas[i] + theta)
        c2 = np.cos(betas[i] + gammas[i] + theta)
        h_all[i, :] = prefix * np.array([[x_i * s1 - y_i * c1, c2, s2]])
    return h_all


def odometer_calculation(u_speed, odom_previous):
    H0 = compute_H(0.)
    H0inv = np.linalg.pinv(H0)
    Vb = H0inv @ u_speed

    odom_p_new = integrate(odom_previous, Vb)

    return odom_p_new


def update_odometry_rk4(current_pos, wheel_speeds):
    current_pos = np.reshape(current_pos, (3, 1))

    new_pos = f_rk4_one_step(current_pos, wheel_speeds)

    return new_pos


def angle_dist(b, a):
    theta = b - a
    while theta < -np.pi:
        theta += 2. * np.pi
    while theta > np.pi:
        theta -= 2. * np.pi
    return theta


def motion_model(x, u):
    pose = x[:3, :]

    u = np.clip(u, 0.1, 0.5)

    left_motion = u[0, 0]
    right_motion = u[1, 0]

    theta = pose[0, 0]

    if u[2, 0] != 0:
        left_motion = (u[0, 0] + u[2, 0]) / 2
    if u[3, 0] != 0:
        right_motion = (u[1, 0] + u[3, 0]) / 2

    stand = wheel_radius / (2 * ROBOT_WIDTH)
    omega = (right_motion - left_motion) * stand
    vel = (left_motion + right_motion) * stand * ROBOT_WIDTH

    if omega != 0:
        x_new = pose[1, 0] + (vel / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        y_new = pose[2, 0] - (vel / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
    else:
        x_new = pose[1, 0] + vel * dt * np.cos(theta)
        y_new = pose[2, 0] + vel * dt * np.sin(theta)

    theta_new = theta + omega * dt

    x_updated = np.copy(x)
    x_updated[0, 0] = theta_new
    x_updated[1, 0] = x_new
    x_updated[2, 0] = y_new

    return x_updated


def motion_model_der(x, u, N):
    theta = x[0, 0]
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

    Gt = np.eye(3)
    if np.abs(omega) > 1e-4:
        Gt[0, 2] = - (vel / omega) * np.sin(theta) + (vel / omega) * np.sin(theta + omega * dt)
        Gt[1, 2] = (vel / omega) * np.cos(theta) - (vel / omega) * np.cos(theta + omega * dt)
    else:
        Gt[0, 2] = - vel * dt * np.sin(theta)
        Gt[1, 2] = vel * dt * np.cos(theta)

    G_expanded = np.eye(3 + 2 * N)
    G_expanded[:3, :3] = Gt

    return G_expanded


def observation_model(x, landmark_pose):
    noise = np.random.uniform(low=-0.005, high=0.005, size=1)
    dx = float(np.add(landmark_pose[0, 0] - x[1, 0], noise))
    dy = float(np.add(landmark_pose[1, 0] - x[2, 0], noise))
    rel = np.arctan2(dy, dx)
    rel = angle_dist(rel, x[0, 0])
    return np.array([[np.sqrt(dx ** 2 + dy ** 2), rel]]).T


def observation_model_deriv(x, landmark_pose):
    noise = np.random.uniform(low=-0.1, high=0.1, size=1)
    dx = float(np.add(landmark_pose[0, 0] - x[1, 0], noise))
    dy = float(np.add(landmark_pose[1, 0] - x[2, 0], noise))
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)

    jac = np.zeros((2, 5))
    jac[1, 0] = -1.

    jac[0, 1] = -dx / dist
    jac[0, 2] = -dy / dist
    jac[0, 3] = dx / dist
    jac[0, 4] = dy / dist

    jac[1, 1] = dy / dist_sq
    jac[1, 2] = -dx / dist_sq
    jac[1, 3] = -dy / dist_sq
    jac[1, 4] = dx / dist_sq

    return jac


def Fxj(l_idx, number_of_landmarks):
    F = np.zeros((5, 3 + 2 * number_of_landmarks))
    F[:3, :3] = np.eye(3)
    F[3:, l_idx * 2 + 1: l_idx * 2 + 3] = np.eye(2)

    return F


def compute_H_t(mu_bar, l_idx):
    x, y, theta = mu_bar[1, 0], mu_bar[2, 0], mu_bar[0, 0]
    lx, ly = mu_bar[l_idx, 0], mu_bar[l_idx + 1, 0]

    delta_x = lx - x
    delta_y = ly - y
    q = delta_x ** 2 + delta_y ** 2

    sqrt_q = np.sqrt(q)

    if sqrt_q < 1e-6:
        sqrt_q = 1e-6

    H_t = np.zeros((2, len(mu_bar)))

    # Partial derivatives of range (r) and bearing (phi) w.r.t. x, y, theta (robot state)
    H_t[0, 0] = -delta_x / sqrt_q  # d(r)/d(x)
    H_t[0, 1] = -delta_y / sqrt_q  # d(r)/d(y)
    H_t[0, 2] = 0  # d(r)/d(theta)

    H_t[1, 0] = delta_y / q  # d(phi)/d(x)
    H_t[1, 1] = -delta_x / q  # d(phi)/d(y)
    H_t[1, 2] = -1  # d(phi)/d(theta)

    # Partial derivatives of range (r) and bearing (phi) w.r.t. landmark positions
    H_t[0, l_idx] = delta_x / sqrt_q  # d(r)/d(lx)
    H_t[0, l_idx + 1] = delta_y / sqrt_q  # d(r)/d(ly)

    H_t[1, l_idx] = -delta_y / q  # d(phi)/d(lx)
    H_t[1, l_idx + 1] = delta_x / q  # d(phi)/d(ly)

    return H_t


def create_observation_map(trace_points, odom_trace_points, ekf_trace_points, mu, scale):
    plt.figure(figsize=(10, 8))

    real_x = [(point[0] - 400) / scale for point in trace_points]
    real_y = [(300 - point[1]) / scale for point in trace_points]

    odom_x = [(point[0] - 400) / scale for point in odom_trace_points]
    odom_y = [(300 - point[1]) / scale for point in odom_trace_points]

    ekf_x = [(point[0] - 400) / scale for point in ekf_trace_points]
    ekf_y = [(300 - point[1]) / scale for point in ekf_trace_points]

    plt.plot(real_x, real_y, 'red', label="Real Path")
    plt.plot(odom_x, odom_y, 'green', label="Odometry Path")
    plt.plot(ekf_x, ekf_y, 'purple', label="EKF Path")

    num_landmarks = (len(mu) - 3) // 2
    landmarks_x = []
    landmarks_y = []
    for i in range(num_landmarks):
        l_idx = 3 + 2 * i
        landmarks_x.append(mu[l_idx, 0])
        landmarks_y.append(mu[l_idx + 1, 0])

    plt.scatter(landmarks_x, landmarks_y, c='slateblue', marker='o', label="Landmarks")

    # Calculate limits based on trace points
    all_x = real_x + odom_x + ekf_x
    all_y = real_y + odom_y + ekf_y

    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    # Add 2 units for correction
    x_range = (x_max - x_min) + 4
    y_range = (y_max - y_min) + 4

    plt.xlim(x_min - 2, x_max + 2)
    plt.ylim(y_min - 2, y_max + 2)

    # Define tick intervals
    x_ticks = np.arange(x_min - 4, x_max + 4, 0.7)
    y_ticks = np.arange(y_min - 4, y_max + 4, 0.7)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.title('Robot Paths and Observed Landmarks (Real Coordinates)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
