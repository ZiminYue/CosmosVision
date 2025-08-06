import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core import GalaxyEngine  # Import the core GalaxyEngine class

# Initialize the simulation engine
engine = GalaxyEngine()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter([], [], s=1)
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.axis('off')
ax.set_title("Spinning Galaxy Simulation")

# Frame update function
def update(frame):
    engine.update()
    sc.set_offsets(engine.positions)
    return sc,

# Run the animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()
