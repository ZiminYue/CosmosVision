
import numpy as np
import threading
import time
from vispy import app, scene
from core import GalaxyEngine

# ========================
# Initialize GalaxyEngine
# ========================
galaxy = GalaxyEngine(num_particles=8000)  # Can start with fewer for testing

# Improve rotation visibility: increase central mass and rotation coefficient
galaxy.central_mass *= 1.5

# Generate colors, full white is brighter
colors = np.ones((len(galaxy.particles), 4))
colors[:, 0:3] = 1.0  # RGB white
colors[:, 3] = 1.0    # alpha

positions = galaxy.get_positions()
lock = threading.Lock()

# ========================
# VisPy window and scatter
# ========================
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(up='+z', fov=45, distance=150)

scatter = scene.visuals.Markers()

sizes = np.array([p.mass*2 for p in galaxy.particles])  # Particle size can be adjusted based on mass
colors = np.ones((len(galaxy.particles), 4))
colors[:, 0:3] = 1.0  # White
colors[:, 3] = 1.0    # alpha

scatter.set_data(positions, face_color=colors, size=sizes)

view.add(scatter)

# ========================
# Background thread: Physics update
# ========================
def physics_loop():
    global positions
    while True:
        galaxy.update(dt=0.2)   # Increase dt to make motion more visible
        # Optional: amplify velocity by 2~3x in update
        with lock:
            positions = galaxy.get_positions()
        time.sleep(0.01)         # Control CPU usage

threading.Thread(target=physics_loop, daemon=True).start()

# ========================
# Foreground: VisPy rendering
# ========================
def update(ev):
    global positions
    with lock:
        scatter.set_data(positions, face_color=colors, size=sizes)

# 15 FPS rendering, ensure smooth window performance
timer = app.Timer(interval=1/15.0, connect=update, start=True)
app.run()