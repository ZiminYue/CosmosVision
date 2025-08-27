import numpy as np
from vispy import app, scene
from core import GalaxyEngine
from background import BackgroundStars  # Your previously fixed version

# ==== Configuration ====
bg_cfg = {
    "n_stars": 3000,
    "bounds": 50,
    "size": 5,
    "color": (1, 1, 1, 0.6),
    "speed_scale": 1.0  # Micro-drift speed
}

galaxy_cfgs = [
    {"num_particles": 1500, "pos": np.array([-20, 0, 0]), "color": (0.5, 0.7, 1, 1), "velocity_factor": 0.1},  # Reduced from 0.5
    {"num_particles": 1500, "pos": np.array([20, 0, 0]), "color": (1, 0.5, 0.7, 1), "velocity_factor": 0.1},   # Reduced from 0.5
]

# ==== Initialize Canvas ====
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(up='+z', fov=45, distance=100)

# ==== Background Stars ====
bg = BackgroundStars(n_stars=bg_cfg["n_stars"], bounds=bg_cfg["bounds"], size=bg_cfg["size"], color=bg_cfg["color"])
scatter_bg = scene.visuals.Markers()
scatter_bg.set_data(bg.positions, face_color=bg.colors, size=bg.sizes, edge_color=None)
view.add(scatter_bg)

# ==== Initialize Galaxies ====
galaxies = []
for cfg in galaxy_cfgs:
    engine = GalaxyEngine(num_particles=cfg["num_particles"], bounds=100, velocity_factor=cfg["velocity_factor"])
    positions = engine.get_positions()  # Initialize particle positions
    # Translate to specified position
    positions += cfg["pos"]
    # Write positions back to engine's internal particles
    for p, pos in zip(engine.particles, positions):
        p.pos[:] = pos
    galaxies.append({"engine": engine, "color": cfg["color"]})

# ==== Initialize Galaxy Rendering Points ====
scatter_galaxies = []
for g in galaxies:
    positions = g["engine"].get_positions()
    c = np.tile(g["color"], (len(positions), 1))
    scatter = scene.visuals.Markers()
    scatter.set_data(positions, face_color=c, size=4)
    view.add(scatter)
    scatter_galaxies.append(scatter)

# ==== Update Function ====
def update(ev):
    # Background stars affected by galaxy gravity
    bg.refresh_canvas(scatter_bg, galaxies=[{"pos": g["engine"].get_positions().mean(axis=0), "mass": 2000.0} for g in galaxies])

    # Galaxy update with slower time step
    for g, scatter in zip(galaxies, scatter_galaxies):
        g["engine"].update(dt=0.02)  # Reduced from 0.1 to make movement slower
        positions = g["engine"].get_positions()
        c = np.tile(g["color"], (len(positions), 1))
        scatter.set_data(positions, face_color=c, size=4)

# ==== Start Timer ====
timer = app.Timer(interval=1/60.0, connect=update, start=True)
app.run()