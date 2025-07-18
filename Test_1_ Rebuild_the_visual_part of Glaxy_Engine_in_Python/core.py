import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GalaxyEngine:
    def __init__(self, count, mass_range=(1, 10), bounds=100):
        self.bounds = bounds
        self.count = count
        self.masses = np.random.uniform(*mass_range, size=(count, 1))
        self.positions = np.random.uniform(-bounds, bounds, size=(count, 3))
        self.speeds = np.zeros((count, 3))
        self.accelerations = np.zeros((count, 3))

    def update(self, dt=0.01, interaction_rate=1.0, black_hole_mass=0.0):
        n = self.count
        acc = np.zeros_like(self.positions)

        for i in range(n):
            force = np.zeros(3)
            for j in range(int(interaction_rate * n)):
                if i == j:
                    continue
                r_vec = self.positions[j] - self.positions[i]
                r_norm2 = np.sum(r_vec**2) + 0.01  # Avoid division by zero
                force += r_vec / r_norm2

            # Center of gravity
            if black_hole_mass > 0.0:
                center_vec = -self.positions[i]
                center_r2 = np.sum(center_vec**2) + 0.01
                force += black_hole_mass * center_vec / center_r2

            acc[i] = force / interaction_rate

        self.accelerations = acc
        self.speeds += dt * self.accelerations
        self.positions += dt * self.speeds

        # Ensure boundary constraints
        self.positions = np.clip(self.positions, -self.bounds, self.bounds)


def plot_starfield(positions):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c='white',
        s=2,
        alpha=0.8
    )

    ax.set_title('Simulated Starfield', color='white')
    ax.tick_params(colors='white')
    ax.xaxis._axinfo["grid"]['color'] = "white"
    ax.yaxis._axinfo["grid"]['color'] = "white"
    ax.zaxis._axinfo["grid"]['color'] = "white"

    plt.show()