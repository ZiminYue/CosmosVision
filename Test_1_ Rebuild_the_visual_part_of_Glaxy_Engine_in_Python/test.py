from core import GalaxyEngine, plot_starfield

# Initialize a simulator and generate 1000 particles
engine = GalaxyEngine(count=1000, bounds=100)

# Simulate several frames in a row
for step in range(100):
    engine.update(dt=0.05, theta=0.5, black_hole_mass=50)

# Get the updated position
positions = engine.positions

# Visualize the current particle distribution
plot_starfield(positions)