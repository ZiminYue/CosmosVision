import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required to enable 3D plotting in matplotlib, even if not used directly
from matplotlib.animation import FuncAnimation
from core import GalaxyEngine
import numpy as np  # Recommended: used for checking position arrays, etc.

# Create GalaxyEngine instance and generate the initial galaxy
engine = GalaxyEngine(num_particles=500)
engine.generate_spiral_galaxy()

# Initialize the figure window
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize particle scatter plot (start with empty positions)
sc = ax.scatter([], [], [], s=1)

# Set view limits
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-30, 30)
ax.set_title("3D Galaxy Simulation")
ax.axis('off')

# Frame update function
def update(frame):
    engine.update(dt=0.1)
    positions = engine.get_positions()

    if frame == 0:
        print("Example positions (first 5):\n", positions[:5])
        print("Contains NaN:", np.isnan(positions).any())
        print("Max value range:", np.max(positions, axis=0))
        print("Min value range:", np.min(positions, axis=0))

    # Update the 3D coordinates of the scatter plot
    sc._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    return sc,

# Start animation
ani = FuncAnimation(fig, update, frames=500, interval=30, blit=False)

# Show the plot
plt.show()