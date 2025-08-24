import numpy as np
from vispy import app, scene
from core import GalaxyEngine
import threading
import time

print("=== Galaxy Visualization ===")

# ========================
# Initialize GalaxyEngine
# ========================
print("Initializing galaxy engine...")
galaxy = GalaxyEngine(num_particles=6000)
galaxy.central_mass *= 2.0  # 强化中心引力

# 初始化旋转速度
for p in galaxy.particles:
    r_vec = p.pos[:2]
    r_mag = np.linalg.norm(r_vec) + 1e-5
    v_mag = np.sqrt(galaxy.G * galaxy.central_mass / r_mag)
    v_dir = np.array([-r_vec[1], r_vec[0]]) / r_mag
    p.vel[:2] = v_dir * v_mag

positions = galaxy.get_positions()
actual_particle_count = len(galaxy.particles)
print(f"✓ Created {actual_particle_count} particles")

lock = threading.Lock()

# ========================
# Particle visual properties
# ========================
distances = np.linalg.norm(positions, axis=1)
max_dist = np.max(distances) if len(distances) > 0 else 1.0
normalized_dist = distances / max_dist

colors = np.ones((len(galaxy.particles), 4), dtype=np.float32)
colors[:, 0] = 1.0
colors[:, 1] = 1.0 - normalized_dist * 0.4
colors[:, 2] = 0.7 + normalized_dist * 0.3
colors[:, 3] = 0.8 - normalized_dist * 0.3

base_sizes = np.array([p.mass for p in galaxy.particles], dtype=np.float32)
sizes = base_sizes * 4.0 * (1.2 - normalized_dist * 0.5)
sizes = np.clip(sizes, 3.0, 25.0)

# ========================
# VisPy setup
# ========================
canvas = scene.SceneCanvas(
    keys='interactive',
    show=True,
    #bgcolor=(0.02, 0.02, 0.08, 1.0),
    size=(1400, 1000),
    title="Galaxy Simulation"
)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(
    up='+z',
    fov=45,
    distance=100,
    elevation=15,
    azimuth=45
)

scatter = scene.visuals.Markers()
scatter.set_data(positions, face_color=colors, size=sizes, symbol='o', edge_color=None)
view.add(scatter)

# ========================
# Physics loop (background thread)
# ========================
def physics_loop():
    global positions
    while True:
        galaxy.update(dt=0.2)
        with lock:
            positions = galaxy.get_positions()
        time.sleep(0.005)

threading.Thread(target=physics_loop, daemon=True).start()

# ========================
# Rendering loop
# ========================
frame_count = 0
last_report = time.time()
last_positions = positions.copy() if len(positions) > 0 else np.array([])

def update(ev):
    global positions, frame_count, last_report, last_positions, colors
    frame_count += 1
    with lock:
        current_positions = positions.copy()

    # Update color gradient based on distance from center
    distances = np.linalg.norm(current_positions, axis=1)
    max_dist = np.max(distances) if len(distances) > 0 else 1.0
    normalized_dist = distances / max_dist
    colors[:, 1] = 1.0 - normalized_dist * 0.4
    colors[:, 2] = 0.7 + normalized_dist * 0.3
    colors[:, 3] = 0.8 - normalized_dist * 0.3

    scatter.set_data(current_positions, face_color=colors, size=sizes, symbol='o', edge_color=None)

    # Optional status reporting
    current_time = time.time()
    if current_time - last_report > 3.0:
        if len(last_positions) > 0:
            movement = np.linalg.norm(current_positions - last_positions, axis=1)
            print(f"Frame {frame_count} - Avg movement: {np.mean(movement):.4f}, Max: {np.max(movement):.4f}")
        last_report = current_time
    last_positions[:] = current_positions

timer = app.Timer(interval=1/30.0, connect=update, start=True)

print("\n=== Galaxy Simulation Started ===")
print("- Left mouse: Rotate view")
print("- Right mouse/scroll: Zoom")

try:
    app.run()
except KeyboardInterrupt:
    print("\n✓ Galaxy simulation stopped")
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
