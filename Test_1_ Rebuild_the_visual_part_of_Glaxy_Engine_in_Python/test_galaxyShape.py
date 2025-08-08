import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from core import GalaxyEngine
import numpy as np

# Create GalaxyEngine instance (galaxy generation is already called in constructor)
engine = GalaxyEngine(num_particles=1000)

# Remove this line: already generated in constructor
# engine.generate_spiral_galaxy()

# Initialize the Figure window
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Get initial positions for setting up scatter plot
initial_positions = engine.get_positions()
sc = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], initial_positions[:, 2], 
                s=1, c='white', alpha=0.8)

# Set view range (adjust according to actual data)
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_zlim(-15, 15)
ax.set_title("3D Galaxy Simulation")
ax.set_facecolor('black')  # Set background to black
fig.patch.set_facecolor('black')  # Set figure background to black

# Frame update
def update(frame):
    # Execute physics simulation step
    engine.update(dt=0.05)  # Use smaller time step for improved stability
    positions = engine.get_positions()

    # Debug information (output only in first few frames)
    if frame < 3:
        print(f"Frame {frame}:")
        print("Position samples (first 3):\n", positions[:3])
        print("Is there a NaN：", np.isnan(positions).any())
        print("Is there a Inf：", np.isinf(positions).any())
        print("Position range X:", np.min(positions[:, 0]), "to", np.max(positions[:, 0]))
        print("Position range Y:", np.min(positions[:, 1]), "to", np.max(positions[:, 1]))
        print("Position range Z:", np.min(positions[:, 2]), "to", np.max(positions[:, 2]))
        print("-" * 40)

    # Check data validity
    if np.isnan(positions).any() or np.isinf(positions).any():
        print("Warning: Invalid data detected, stopping animation")
        return sc,

    # Update 3D coordinates of scatter plot
    sc._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    
    # Update title to show current frame number
    ax.set_title(f"3D Galaxy Simulation - Frame {frame}")
    
    return sc,

# Start animation
ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=False, repeat=True)

# Display output
plt.tight_layout()
plt.show()

# Optional: Save animation as gif (requires pillow or ffmpeg)）
# ani.save('galaxy_simulation.gif', writer='pillow', fps=20)