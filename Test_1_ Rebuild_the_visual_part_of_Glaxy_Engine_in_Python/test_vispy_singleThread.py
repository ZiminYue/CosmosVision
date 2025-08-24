import numpy as np
from vispy import app, scene
from core import GalaxyEngine

print("=== Single Thread Galaxy Visualization ===")
print("Testing without multi-threading to isolate the issue...")

# ========================
# Creating a galaxy
# ========================
print("Creating galaxy...")
galaxy = GalaxyEngine(num_particles=1000)  # Fewer particles
print(f"✅ Created {len(galaxy.particles)} particles")

# ========================
# Testing multiple physics updates in succession
# ========================
print("\n🧪 Testing multiple physics updates...")

for step in range(5):
    print(f"\nTesting step {step + 1}:")
    old_pos = galaxy.get_positions()
    
    try:
        galaxy.update(dt=0.1)
        new_pos = galaxy.get_positions()
        
        movement = np.linalg.norm(new_pos - old_pos, axis=1)
        avg_movement = np.mean(movement)
        max_movement = np.max(movement)
        
        print(f"  ✅ Step {step + 1} - Movement: avg={avg_movement:.4f}, max={max_movement:.4f}")
        
        # Check speed
        sample_vels = [np.linalg.norm(p.vel) for p in galaxy.particles[:3]]
        print(f"  Sample velocities: {[f'{v:.2f}' for v in sample_vels]}")
        
    except Exception as e:
        print(f"  ❌ Step {step + 1} FAILED: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n🎨 Setting up visualization...")

# ========================
# Get initial positions for visualization
# ========================
positions = galaxy.get_positions()

# Simple colors
colors = np.ones((len(galaxy.particles), 4), dtype=np.float32)
colors[:, 0] = 1.0  # Red
colors[:, 1] = 0.7  # Green
colors[:, 2] = 0.4  # Blue
colors[:, 3] = 0.8  # Alpha

sizes = np.full(len(galaxy.particles), 6.0, dtype=np.float32)

# ========================
# VisPy settings
# ========================
canvas = scene.SceneCanvas(
    keys='interactive',
    show=True,
    #bgcolor=(0.05, 0.05, 0.15, 1.0),
    size=(1000, 800),
    title="Single Thread Galaxy Test"
)

view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(
    up='+z',
    fov=60,
    distance=150,
    elevation=25,
    azimuth=30
)

scatter = scene.visuals.Markers()
scatter.set_data(positions, face_color=colors, size=sizes, symbol='o')
view.add(scatter)

print("✅ Visualization setup complete")

# ========================
# Single-threaded rendering loop - run physics directly in the render callback
# ========================
physics_step_count = 0
last_positions = positions.copy()

def update_everything(ev):
    global positions, physics_step_count, last_positions
    physics_step_count += 1
    
    try:
        # Save old positions
        old_positions = positions.copy()
        
        # Run physics directly in the rendering thread
        galaxy.update(dt=0.05)  # Smaller time step
        
        # Get new positions
        positions = galaxy.get_positions()
        
        # Update visualization
        scatter.set_data(positions, face_color=colors, size=sizes, symbol='o')
        
        # Calculate movement
        movement = np.linalg.norm(positions - old_positions, axis=1)
        avg_movement = np.mean(movement)
        max_movement = np.max(movement)
        
        # Report every 30 frames (1 second)
        if physics_step_count % 30 == 0:
            print(f"🔄 Frame {physics_step_count}:")
            print(f"   Movement: avg={avg_movement:.4f}, max={max_movement:.4f}")
            
            # Compare to last position
            total_change = np.linalg.norm(positions - last_positions, axis=1)
            total_avg = np.mean(total_change)
            total_max = np.max(total_change)
            print(f"   Total change since last report: avg={total_avg:.4f}, max={total_max:.4f}")
            
            if total_max > 5.0:
                print("   🎉 EXCELLENT: Large movements - galaxy is animating!")
            elif total_max > 1.0:
                print("   ✅ GOOD: Clear movement detected")
            elif total_max > 0.1:
                print("   ⚠️  OK: Some movement")
            else:
                print("   ❌ PROBLEM: Very little movement")
            
            last_positions = positions.copy()
        
    except Exception as e:
        print(f"❌ Update error at frame {physics_step_count}: {e}")
        import traceback
        traceback.print_exc()

# 30 FPS update (physics + rendering)
timer = app.Timer(interval=1/30.0, connect=update_everything, start=True)

print("\n" + "="*60)
print("🌌 SINGLE THREAD GALAXY TEST")
print("="*60)
print("This version runs physics and rendering in the same thread.")
print("If this works, the problem was multi-threading.")
print("If this doesn't work, the problem is in the physics engine.")
print("")
print("Watch for movement reports every second!")
print("🎮 Controls: Left mouse = rotate, Right mouse = zoom")
print("="*60)

try:
    app.run()
except KeyboardInterrupt:
    print("\n✅ Test completed")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()