import numpy as np
from vispy import app, scene

class BackgroundStars:
    def __init__(self, n_stars=5000, bounds=50, size=6, color=(1,1,1,0.7), seed=42):
        np.random.seed(seed)
        self.n_stars = n_stars
        self.bounds = bounds
        self.positions = np.random.uniform(-bounds, bounds, (n_stars, 3))
        self.velocities = np.random.normal(0, 0.005, (n_stars, 3))  # Micro-drift velocity
        self.sizes = np.ones(n_stars) * size
        self.colors = np.zeros((n_stars, 4), dtype=np.float32)
        self.colors[:, :3] = color[:3]
        self.colors[:, 3] = np.random.uniform(0.3, color[3], n_stars)  # Alpha gradient effect

    def apply_external_gravity(self, galaxies):
        """
        Apply external gravitational influence from galaxies
        galaxies: List containing galaxy information, each element should have 'pos' and 'mass' keys
        """
        if not galaxies:
            return
        
        for galaxy in galaxies:
            galaxy_pos = galaxy['pos']
            galaxy_mass = galaxy.get('mass', 1000.0)
            
            # Calculate distance vector from each star to galaxy center
            diff = galaxy_pos - self.positions
            distances = np.linalg.norm(diff, axis=1)
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-6)
            
            # Calculate gravity strength (simplified gravity model)
            gravity_strength = galaxy_mass / (distances ** 2 + 100.0)  # Add softening factor
            
            # Limit gravity strength to avoid excessive influence
            gravity_strength = np.minimum(gravity_strength, 0.005)
            
            # Calculate gravity direction (unit vector)
            directions = diff / distances[:, np.newaxis]
            
            # Apply gravity to velocity (small influence to maintain background stars' drift characteristics)
            gravity_force = directions * gravity_strength[:, np.newaxis] * 0.0001
            self.velocities += gravity_force

    def update(self, dt=1.0):
        # Random perturbation to make stars drift slowly
        self.velocities += np.random.normal(0, 0.0002, self.velocities.shape)
        self.positions += self.velocities * dt

        # Speed limit
        speed = np.linalg.norm(self.velocities, axis=1)
        max_speed = 0.01
        mask = speed > max_speed
        self.velocities[mask] = self.velocities[mask] / speed[mask][:, np.newaxis] * max_speed

        # Keep within boundaries, simple bounce
        mask_x = np.abs(self.positions[:,0]) > self.bounds
        self.velocities[mask_x,0] *= -1
        mask_y = np.abs(self.positions[:,1]) > self.bounds
        self.velocities[mask_y,1] *= -1
        mask_z = np.abs(self.positions[:,2]) > self.bounds/5
        self.velocities[mask_z,2] *= -1

    def refresh_canvas(self, scatter, galaxies=None):
        # galaxies can be None or a list
        if galaxies is not None:
            self.apply_external_gravity(galaxies)
        self.update()
        scatter.set_data(self.positions, face_color=self.colors, size=self.sizes, edge_color=None)

# ---- VisPy Demo ----
if __name__ == "__main__":
    canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(up='+z', fov=45, distance=100)

    bg = BackgroundStars(n_stars=3000, size=6, color=(1,1,1,0.6))
    scatter = scene.visuals.Markers()
    scatter.set_data(bg.positions, face_color=bg.colors, size=bg.sizes, edge_color=None)
    view.add(scatter)

    def update(ev):
        bg.refresh_canvas(scatter)

    timer = app.Timer(interval=1/60.0, connect=update, start=True)
    app.run()