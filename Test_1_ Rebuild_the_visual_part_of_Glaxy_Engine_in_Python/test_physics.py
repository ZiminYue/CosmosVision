from core import GalaxyEngine
import numpy as np

print("Testing physics engine...")
galaxy = GalaxyEngine(num_particles=100)  # very few particles
print(f"Created {len(galaxy.particles)} particles")

old_pos = galaxy.get_positions()
print("Running one update...")
galaxy.update(dt=0.1)
new_pos = galaxy.get_positions()

movement = np.linalg.norm(new_pos - old_pos, axis=1)
print(f"Movement: avg={np.mean(movement):.6f}, max={np.max(movement):.6f}")