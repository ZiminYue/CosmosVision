from vispy import app, scene
from core import GalaxyEngine
import numpy as np
import time

# Initialize galaxy
galaxy = GalaxyEngine(num_particles=1500)
galaxy.use_gpu = True    # GPU
galaxy.use_sph = True   # SPH
galaxy.central_mass *= 2.0

positions = galaxy.get_positions()

# Fixed colors & point sizes (one-time setup)
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

def update(ev):
    global positions, frame_count, galaxy, scatter

    frame_count += 1

    # ---- Timing: Physics update ----
    t0 = time.time()
    for _ in range(3):
        galaxy.update(dt=0.0033)
    t1 = time.time()
    update_time = (t1 - t0) * 1000  # milliseconds

    positions = galaxy.get_positions()

    # ---- Timing: Rendering ----
    t2 = time.time()
    scatter.set_data(positions, face_color=colors, size=sizes, edge_color=None)
    t3 = time.time()
    render_time = (t3 - t2) * 1000

    total_time = (t3 - t0) * 1000
    fps = 1000.0 / total_time if total_time > 0 else 0

    #print(f"[DEBUG] Frame {frame_count}: Update={update_time:.2f} ms, Render={render_time:.2f} ms, Total={total_time:.2f} ms ({fps:.1f} FPS)")

timer = app.Timer(interval=1/60.0, connect=update, start=True)
app.run()