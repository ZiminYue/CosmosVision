from vispy import app, scene
from core import GalaxyEngine
import numpy as np
import time

# Initialize
galaxy = GalaxyEngine(num_particles=900)
galaxy.central_mass *= 2.0

positions = galaxy.get_positions()
colors = np.ones((len(positions), 4), dtype=np.float32)
colors[:, 0] = 1.0
colors[:, 1] = 0.7
colors[:, 2] = 1.0
colors[:, 3] = 0.8
sizes = np.ones(len(positions)) * 4

# VisPy setup
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(up='+z', fov=45, distance=100)
scatter = scene.visuals.Markers()
scatter.set_data(positions, face_color=colors, size=sizes, edge_color=None)
view.add(scatter)

frame_count = 0
sph_interval = galaxy.sph_interval

def update(ev):
    global positions, frame_count
    frame_count += 1
    galaxy.update(dt=0.01)  # Update physics one time
    positions = galaxy.get_positions()
    
    # Galaxy color
    colors = np.ones((len(positions), 4), dtype=np.float32)
    colors[:, 0] = 1.0  # R
    colors[:, 1] = 0.7  # G
    colors[:, 2] = 1.0  # B
    colors[:, 3] = 0.8  # alpha
    
    scatter.set_data(positions, face_color=colors, size=sizes, edge_color=None)

timer = app.Timer(interval=1/25.0, connect=update, start=True)
app.run()
