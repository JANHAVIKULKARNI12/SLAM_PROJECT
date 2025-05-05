import numpy as np
import matplotlib.pyplot as plt

def run():
    print("[Lab 1] Running Kalman Filter...")

    # -----------------------------
    # Initialization
    # -----------------------------
    x = np.array([[0], [0]])  # Initial state: position=0, velocity=0
    P = np.eye(2)

    Q = np.array([[0.1, 0], 
                  [0, 0.1]])

    R = np.array([[1, 0],
                  [0, 1]])

    F = np.array([[1, 1],
                  [0, 1]])

    H = np.array([[0, 1],
                  [0, 1]])

    # -----------------------------
    # Simulate true velocity
    # -----------------------------
    time = np.arange(0, 60, 1)
    true_velocity = np.zeros_like(time, dtype=float)
    true_velocity[:30] = 1.0 * time[:30]
    true_velocity[30:] = true_velocity[29]

    # -----------------------------
    # Simulate noisy sensor readings
    # -----------------------------
    np.random.seed(42)
    sensor1 = true_velocity + np.random.normal(0, 1, size=time.shape)
    sensor2 = true_velocity + np.random.normal(0, 1, size=time.shape)

    # -----------------------------
    # Kalman Filter loop
    # -----------------------------
    estimated_positions = []
    estimated_velocities = []

    for t in range(len(time)):
        x = F @ x
        P = F @ P @ F.T + Q

        z = np.array([[sensor1[t]], [sensor2[t]]])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        estimated_positions.append(x[0, 0])
        estimated_velocities.append(x[1, 0])

    # -----------------------------
    # Plot the results
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(time, true_velocity, label='True Velocity', linewidth=2)
    plt.plot(time, sensor1, label='Sensor 1 (Noisy)', linestyle='--', alpha=0.7)
    plt.plot(time, sensor2, label='Sensor 2 (Noisy)', linestyle='--', alpha=0.7)
    plt.plot(time, estimated_velocities, label='Estimated Velocity (KF)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Kalman Filter Velocity Estimation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("OUTPUT/lab1_kalman_output.png")  # Save output instead of showing
    plt.close()

if __name__ == "__main__":
    run()
