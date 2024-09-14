from general_functions import *


def ekf_slam_step(mu, sig, u, z, R, Q, association_threshold):
    # Prediction step
    mu_bar = motion_model(mu, u)
    print("Mu bar is:")
    print(mu_bar)
    number_of_landmarks = int((len(mu_bar) - 3) / 2)
    G = motion_model_der(mu, u, number_of_landmarks)
    R_all = np.zeros_like(G)
    R_all[:3, :3] = R
    sig_bar = G @ sig @ G.T + R_all

    for k in range(len(z)):
        r_i = z[k][0]
        phi_i = z[k][1]
        zk = np.array([[r_i, phi_i]]).T

        best_l_idx = None
        best_landmark = None
        min_probability = float('inf')

        for i in range((len(mu_bar) - 3) // 2):
            l_idx = 3 + i * 2

            z_i_pred = observation_model(mu_bar, mu_bar[l_idx:l_idx + 2, :])

            H_t = compute_H_t(mu_bar, l_idx)

            psi = H_t @ sig_bar @ H_t.T + Q

            probability_pi = (zk - z_i_pred).T @ np.linalg.inv(psi) @ (zk - z_i_pred)
            # VERY SMALL PROBABILITY, ALWAYS UPDATES, NOT INITIALIZES

            if probability_pi < min_probability and probability_pi < association_threshold:
                min_probability = probability_pi
                best_l_idx = l_idx
                best_landmark = (i + 1)

        if best_l_idx is None:
            print("INITIALIZE NEW")
            # Initialize new landmark
            l_idx = len(mu_bar)
            mu_bar = np.vstack((mu_bar, np.zeros((2, 1))))
            sig_bar = np.vstack((sig_bar, np.zeros((2, sig_bar.shape[1]))))
            sig_bar = np.hstack((sig_bar, np.zeros((sig_bar.shape[0], 2))))

            theta, xx, yy = mu_bar[0, 0], mu_bar[1, 0], mu_bar[2, 0]

            mu_bar[l_idx, 0] = xx + r_i * np.cos(theta + phi_i)
            mu_bar[l_idx + 1, 0] = yy + r_i * np.sin(theta + phi_i)
            sig_bar[l_idx:l_idx + 2, l_idx:l_idx + 2] = Q

        else:
            print("UPDATE IT")
            z_i_pred = observation_model(mu_bar, mu_bar[best_l_idx:best_l_idx + 2, :])
            F = Fxj(best_landmark, number_of_landmarks)
            H_t = observation_model_deriv(mu_bar, mu_bar[best_l_idx:best_l_idx + 2, :]) @ F
            print("Size of H_t before update: ", H_t.shape)
            print("Size of sig_bar before update:", sig_bar.shape)

            # KALMAN GAIN
            psi = H_t @ sig_bar @ H_t.T + Q
            K = sig_bar @ H_t.T @ np.linalg.inv(psi)

            # UPDATE STATE MEAN AND COVARIANCE - CHECKED

            mu_bar = mu_bar + K @ (zk - z_i_pred)
            mu_bar[0, 0] = angle_dist(mu_bar[0, 0], 0.)
            sig_bar = (np.eye(len(sig_bar)) - K @ H_t) @ sig_bar

    return mu_bar, sig_bar
