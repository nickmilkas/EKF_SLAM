import numpy as np
from general_functions import *


def jacobian_motion_model(mu, u):
    theta = mu[2]
    v, w = u

    if np.abs(w) > 1e-4:
        Fx = np.array([
            [1, 0, -v / w * (np.cos(theta + w * dt) - np.cos(theta))],
            [0, 1, -v / w * (np.sin(theta + w * dt) - np.sin(theta))],
            [0, 0, 1]
        ])
    else:
        Fx = np.array([
            [1, 0, -v * dt * np.sin(theta)],
            [0, 1, v * dt * np.cos(theta)],
            [0, 0, 1]
        ])

    return Fx


def landmark_position(mu, z):
    r = z[0, 0]
    phi = z[1, 0]
    theta = mu[2]
    lx = mu[0] + r * np.cos(theta + phi)
    ly = mu[1] + r * np.sin(theta + phi)
    return np.array([lx, ly])


def compute_H(mu, landmark_pos):
    """Compute the Jacobian H of the measurement model."""
    delta_x = landmark_pos[0] - mu[0]
    delta_y = landmark_pos[1] - mu[1]
    q = delta_x ** 2 + delta_y ** 2
    sqrt_q = np.sqrt(q)
    H = np.array([
        [-delta_x / sqrt_q, -delta_y / sqrt_q, 0, delta_x / sqrt_q, delta_y / sqrt_q],
        [delta_y / q, -delta_x / q, -1, -delta_y / q, delta_x / q]
    ])
    return H


def measurement_model(mu, landmark_pos):
    """Compute expected measurement for a given landmark."""
    dx = landmark_pos[0] - mu[0]
    dy = landmark_pos[1] - mu[1]
    q = dx ** 2 + dy ** 2
    z_hat = np.array([np.sqrt(q), np.arctan2(dy, dx) - mu[2]])
    return z_hat


# Main EKF SLAM function
def ekf_slam_unknown_correspondences(mu, sigma, u, z, R, Q, association_threshold=5.991):
    # Step 1: Prediction
    number_of_landmarks = int((len(mu) - 3) / 2)
    mu_bar = motion_model(mu, u)
    G = motion_model_der(mu, u, number_of_landmarks)

    # Expand G and R to the size of the full state
    G_t = np.eye(len(mu_bar))
    G_t[:3, :3] = G
    R_t = np.zeros_like(sigma)
    R_t[:3, :3] = R

    sigma = G_t @ sigma @ G_t.T + R_t

    # Step 2: Correction
    for i, z_i in enumerate(z):

        # Extract r and phi from z_i
        r_i, phi_i = z_i[0], z_i[1]

        best_nis = float('inf')
        best_landmark = None

        # Step 2.1: Try to associate each observation with an existing landmark
        for j in range((len(mu_bar) - 3) // 2):
            l_idx = 3 + 2 * j
            landmark_pos = mu_bar[l_idx:l_idx + 2]

            # Compute the expected measurement and the Jacobian H
            z_hat = measurement_model(mu_bar, landmark_pos)
            H = compute_H(mu_bar, landmark_pos)

            # Innovation
            v = np.array([r_i - z_hat[0], phi_i - z_hat[1]])
            v[1] = np.mod(v[1] + np.pi, 2 * np.pi) - np.pi  # Normalize angle

            # Compute NIS (Normalized Innovation Squared)
            S = H @ sigma @ H.T + Q
            nis = v.T @ np.linalg.inv(S) @ v

            # Check if this is the best association
            if nis < best_nis and nis < association_threshold:
                best_nis = nis
                best_landmark = l_idx

        # Step 2.2: If no association is found, initialize a new landmark
        if best_landmark is None:
            # New landmark position
            new_landmark_pos = landmark_position(mu_bar[:3], z_i)
            mu_bar = np.append(mu_bar, new_landmark_pos)

            # Expand covariance matrix for new landmark
            sigma_expanded = np.zeros((len(sigma) + 2, len(sigma) + 2))
            sigma_expanded[:len(sigma), :len(sigma)] = sigma
            sigma_expanded[-2:, -2:] = 1e3 * np.eye(2)  # Large initial uncertainty
            sigma = sigma_expanded

        else:
            # Update the state for the associated landmark
            l_idx = best_landmark
            landmark_pos = mu_bar[l_idx:l_idx + 2]
            H = compute_H(mu_bar, landmark_pos)
            z_hat = measurement_model(mu_bar, landmark_pos)

            # Innovation
            v = np.array([r_i - z_hat[0], phi_i - z_hat[1]])
            v[1] = np.mod(v[1] + np.pi, 2 * np.pi) - np.pi  # Normalize angle

            # Kalman Gain
            S = H @ sigma @ H.T + Q
            K = sigma @ H.T @ np.linalg.inv(S)

            # Update state mean and covariance
            mu_bar = mu_bar + K @ v
            sigma = (np.eye(len(sigma)) - K @ H) @ sigma

    return mu_bar, sigma
