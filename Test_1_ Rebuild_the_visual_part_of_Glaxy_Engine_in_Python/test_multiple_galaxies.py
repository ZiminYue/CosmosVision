from vispy import app, scene
from core import GalaxyEngine
import numpy as np
import time

# ----------------------------
# Initialize two galaxies
# ----------------------------
gal1 = GalaxyEngine(num_particles=1600)
gal1.use_gpu = True
gal1.use_sph = True
gal1.central_mass *= 5.0

gal2 = GalaxyEngine(num_particles=1500)
gal2.use_gpu = True
gal2.use_sph = True
gal2.central_mass *= 2.0

# Shift initial positions and give relative velocities
for p in gal1.particles:
    p.pos += np.array([0, -30, 0]) # initial position
    p.vel += np.array([0, 8, 0])   # moving direction
for p in gal2.particles:
    p.pos += np.array([20, 0, 0])
    p.vel += np.array([-5, 0, 0])  

    ## Collision from different dimensions
    # x, y, z = p.pos
    # vx, vy, vz = p.vel
    # p.pos = np.array([x, z, y])  # Shift Y <-> Z
    # p.vel = np.array([vx, vz, vy])  
    # p.pos += np.array([30, 0, 0])   
    # p.vel += np.array([-0.5, 0, 0]) 

# ----------------------------
# Combine particles and set colors
# ----------------------------
def get_positions_colors(gal1, gal2):
    pos1 = gal1.get_positions()
    pos2 = gal2.get_positions()
    positions = np.vstack([pos1, pos2])
    
    colors1 = np.ones((len(pos1), 4), dtype=np.float32)
    colors1[:, 0] = 0.3   # cyan-ish
    colors1[:, 1] = 1.0
    colors1[:, 2] = 1.0
    colors1[:, 3] = 0.8

    colors2 = np.ones((len(pos2), 4), dtype=np.float32)
    colors2[:, 0] = 1.0   # magenta-ish
    colors2[:, 1] = 0.3
    colors2[:, 2] = 1.0
    colors2[:, 3] = 0.8

    colors = np.vstack([colors1, colors2])
    return positions, colors

positions, colors = get_positions_colors(gal1, gal2)
sizes = np.ones(len(positions)) * 3

# ----------------------------
# VisPy setup
# ----------------------------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(up='+z', fov=45, distance=150)

scatter = scene.visuals.Markers()
scatter.set_data(positions, face_color=colors, size=sizes, edge_color=None)
view.add(scatter)

frame_count = 0

# ----------------------------
# Update function
# ----------------------------
def update(ev):
    global frame_count, scatter

    frame_count += 1

    t0 = time.time()
    # Update both galaxies
    for _ in range(3):
        gal1.update(dt=0.0033)
        gal2.update(dt=0.0033)
    t1 = time.time()

    positions, colors = get_positions_colors(gal1, gal2)
    scatter.set_data(positions, face_color=colors, size=sizes, edge_color=None)
    t2 = time.time()

    total_time = (t2 - t0) * 1000
    fps = 1000.0 / total_time if total_time > 0 else 0
    #print(f"[DEBUG] Frame {frame_count}: Total={total_time:.2f} ms ({fps:.1f} FPS)")

timer = app.Timer(interval=1/60.0, connect=update, start=True)
app.run()