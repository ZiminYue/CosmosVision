from core import GalaxyEngine, plot_starfield
import time

# Initialize GalaxyEngine instance
galaxy = GalaxyEngine(
    count=500,                # Number of particles (can be adjusted)
    mass_range=(1, 5),        # Mass range for each particle
    bounds=100                # Spatial boundary size
)

# Simulate multiple time steps
num_steps = 100
for step in range(num_steps):
    galaxy.update(dt=0.02, interaction_rate=1.0, black_hole_mass=50.0)
    if step % 10 == 0:
        print(f"Step {step} done")

# Render the final image
plot_starfield(galaxy.positions)
