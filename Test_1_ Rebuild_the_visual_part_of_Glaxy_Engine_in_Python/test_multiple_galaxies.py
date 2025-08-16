import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core import GalaxyEngine, Particle

# Create an empty engine
engine = GalaxyEngine(num_particles=0)  # Create empty engine first
engine.particles = []  # Clear particle list

# Function to manually add multiple galaxies
def create_mini_galaxy(center, velocity, num_particles=120, arms=3):
    """Create a small galaxy - maintain spiral arm structure"""
    particles = []
    for i in range(num_particles):
        # Distance distribution - more like original version
        radius = max(5, np.random.gamma(2, 15))
        
        # Spiral angle - use original version parameters
        arm = i % arms
        base_angle = arm * (2 * np.pi / arms)
        spiral_angle = radius * 0.08  # Same spiral winding as original
        noise = np.random.normal(0, 0.15)  # Same noise as original
        theta = base_angle + spiral_angle + noise
        
        # Relative position (relative to galaxy center)
        x_rel = radius * np.cos(theta)
        y_rel = radius * np.sin(theta)
        z_rel = np.random.normal(0, 1.5)  # Same disk thickness as original
        
        # Absolute position (add galaxy center offset)
        x = x_rel + center[0]
        y = y_rel + center[1]
        z = z_rel + (center[2] if len(center) > 2 else 0)
        
        # Local rotation velocity - use original velocity calculation
        distance_2d = np.sqrt(x_rel**2 + y_rel**2)
        
        if distance_2d < 10:
            # Interior: rigid body rotation
            local_speed = distance_2d * 0.15  
        else:
            # Exterior: flat rotation curve
            local_speed = np.sqrt(engine.G * 1000.0 / (distance_2d + 5)) * 1.2
        
        # Radial velocity component
        radial_speed = 0.02 * np.sin(2 * spiral_angle)
        
        # Local velocity - clockwise rotation
        vx_local = local_speed * np.sin(theta) + radial_speed * np.cos(theta)
        vy_local = -local_speed * np.cos(theta) + radial_speed * np.sin(theta)
        vz_local = np.random.normal(0, 0.05)
        
        # Total velocity = local rotation + galaxy bulk motion
        vx = vx_local + velocity[0]
        vy = vy_local + velocity[1]
        vz = vz_local + (velocity[2] if len(velocity) > 2 else 0)
        
        particles.append(Particle(
            mass=np.random.uniform(0.8, 1.5),  # Same mass range as original
            pos=[x, y, z],
            vel=[vx, vy, vz],
        ))
    
    return particles

# Add three galaxies
print("Creating galaxies...")

# Galaxy 1: Left side, moving right - 3-arm spiral
galaxy1 = create_mini_galaxy(
    center=np.array([-80, 0, 0]), 
    velocity=np.array([8, 0, 0]), 
    num_particles=140,
    arms=3
)

# Galaxy 2: Right side, moving left - 5-arm spiral
galaxy2 = create_mini_galaxy(
    center=np.array([80, 0, 0]), 
    velocity=np.array([-11, 0, 0]), 
    num_particles=140,
    arms=5
)

# Galaxy 3: Top side, moving down - 4-arm spiral
galaxy3 = create_mini_galaxy(
    center=np.array([0, 80, 0]), 
    velocity=np.array([0, -11, 0]), 
    num_particles=120,
    arms=4
)

# Merge all particles
engine.particles = galaxy1 + galaxy2 + galaxy3
engine.num_particles = len(engine.particles)

print(f"Total particles: {engine.num_particles}")

# Adjust engine parameters for multi-galaxy simulation
engine.G = 1.0  # Maintain reasonable gravity
engine.central_mass = 500.0  # Moderate central mass
engine.use_sph = True  # Enable SPH fluid effects!

# Plot setup
fig, ax = plt.subplots(figsize=(10, 10))
scat = ax.scatter([], [], s=0.8, c='white', alpha=0.7)
ax.set_facecolor('black')
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_aspect('equal')
ax.set_title('Multi-Galaxy Collision Simulation')

# Update function
def update(frame):
    engine.update(dt=0.05)  # Use same time step as original
    
    if frame % 100 == 0:  # Print status every 100 frames
        print(f"Frame {frame} - SPH enabled: {engine.use_sph}")
    
    positions = engine.get_positions()
    scat.set_offsets(positions[:, :2])
    
    # Update title to show frame number
    ax.set_title(f'Multi-Galaxy Collision (SPH Enabled) - Frame {frame}')
    
    return scat,

# Start animation
ani = FuncAnimation(fig, update, frames=2000, interval=50, blit=True)

plt.tight_layout()
plt.show()

# Optional: Save animation
# ani.save('galaxy_collision.gif', writer='pillow', fps=25)