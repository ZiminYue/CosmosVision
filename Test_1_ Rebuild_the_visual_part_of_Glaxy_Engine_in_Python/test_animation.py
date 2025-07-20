import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core import GalaxyEngine

# Initialize the GalaxyEngine simulator with 1000 particles inside a square boundary of size 100
engine = GalaxyEngine(count=300, bounds=100)

# Create a matplotlib figure and axis for plotting
fig, ax = plt.subplots(figsize=(8, 8))

# Set plot limits to match the simulation bounds on both axes
ax.set_xlim(-engine.bounds, engine.bounds)
ax.set_ylim(-engine.bounds, engine.bounds)

# Set background color to black to resemble space
ax.set_facecolor('black')

# Initialize a scatter plot with empty data, size 1 for each particle, white color
scat = ax.scatter([], [], s=1, c='white')

def init():
    """
    Initialization function for the animation.
    Sets the scatter plot offsets to an empty array initially.
    
    Returns:
        tuple: scatter plot artist, required by FuncAnimation
    """
    scat.set_offsets(np.empty((0, 2)))  # No points at start
    return scat,

def update(frame):
    """
    Update function called for each frame of the animation.
    Advances the simulation by one step and updates scatter plot positions.
    
    Args:
        frame (int): Current frame number (unused here, but required by FuncAnimation)
    
    Returns:
        tuple: updated scatter plot artist
    """
    # Advance simulation by one timestep
    engine.update(dt=0.5, theta=0.5, black_hole_mass=6000)

    # Extract updated 2D positions of particles
    positions = np.array([p.pos for p in engine.particles])
    
    # Update scatter plot offsets with new positions
    scat.set_offsets(positions[:, :2])
    return scat,

# Create an animation object that calls `update` for 200 frames,
# initializes with `init`, uses blitting for efficiency, and updates every 30ms
ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=30)

# Add a title with white text color
plt.title('Galaxy Particle Simulation (2D)', color='white')

# Display the interactive animation window
plt.show()
