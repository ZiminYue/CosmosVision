import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from core import GalaxyEngine
import numpy as np

# Create GalaxyEngine instance (galaxy generation is already called in constructor)
engine = GalaxyEngine(num_particles=500)

# Store initial parameters for reset functionality
initial_num_particles = 500
reset_requested = False

def reset_galaxy():
    """Reset the galaxy to initial state"""
    global engine, reset_requested
    engine = GalaxyEngine(num_particles=initial_num_particles)
    reset_requested = True
    print("Galaxy reset to initial state! Press 'r' to reset again anytime.")

# 移除这行：构造函数中已经生成了
# engine.generate_spiral_galaxy()

# Initialize graphics window
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
ax.set_title("3D Galaxy Simulation - Press 'R' to Reset")
ax.set_facecolor('black')  # Set background to black
fig.patch.set_facecolor('black')  # Set figure background to black

# Key press event handler
def on_key_press(event):
    """Handle keyboard input"""
    if event.key.lower() == 'r':
        reset_galaxy()
    elif event.key.lower() == 'q':
        plt.close('all')
        print("Exiting simulation...")
    elif event.key.lower() == 'p':
        # Pause/Resume functionality
        if hasattr(ani, '_paused'):
            ani._paused = not ani._paused
        else:
            ani._paused = False
        if ani._paused:
            print("Simulation paused. Press 'P' again to resume.")
        else:
            print("Simulation resumed.")

# Connect keyboard event
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Frame update function
def update(frame):
    global reset_requested, sc
    
    # Check if pause is requested
    if hasattr(ani, '_paused') and ani._paused:
        return sc,
    
    # Handle reset request
    if reset_requested:
        # Update scatter plot with new initial positions
        new_positions = engine.get_positions()
        sc._offsets3d = (new_positions[:, 0], new_positions[:, 1], new_positions[:, 2])
        reset_requested = False
        ax.set_title("3D Galaxy Simulation - RESET - Press 'R' to Reset Again")
        return sc,
    
    # Execute physics simulation step
    engine.update(dt=0.05)  # Use smaller time step for improved stability
    positions = engine.get_positions()

    # Debug information (output only in first few frames)
    if frame < 3:
        print(f"Frame {frame}:")
        print("Sample positions (first 3):\n", positions[:3])
        print("Has NaN:", np.isnan(positions).any())
        print("Has Inf:", np.isinf(positions).any())
        print("X range:", np.min(positions[:, 0]), "to", np.max(positions[:, 0]))
        print("Y range:", np.min(positions[:, 1]), "to", np.max(positions[:, 1]))
        print("Z range:", np.min(positions[:, 2]), "to", np.max(positions[:, 2]))
        print("-" * 50)
        print("Controls: 'R'=Reset, 'P'=Pause/Resume, 'Q'=Quit")
        print("-" * 50)

    # Check data validity
    if np.isnan(positions).any() or np.isinf(positions).any():
        print("Warning: Invalid data detected, stopping animation")
        return sc,

    # Update 3D coordinates of scatter plot
    sc._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
    
    # Update title to show current frame number
    ax.set_title(f"3D Galaxy Simulation - Frame {frame} - Press 'R' to Reset")
    
    return sc,

# Start animation
ani = FuncAnimation(fig, update, frames=2000, interval=50, blit=False, repeat=True)

# Display graphics
plt.tight_layout()

# Print initial instructions
print("🌌 Galaxy Simulation Controls:")
print("  'R' or 'r' = Reset to initial state")
print("  'P' or 'p' = Pause/Resume animation") 
print("  'Q' or 'q' = Quit simulation")
print("  Mouse = Rotate view (drag to rotate)")
print("-" * 50)
print("Ready! Focus on the window and press keys to control.")

plt.show()

# Optional: Save animation as gif (requires pillow or ffmpeg)
# ani.save('galaxy_simulation.gif', writer='pillow', fps=20)