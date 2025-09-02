import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import random
import math

from vispy import app, scene
from core import GalaxyEngine
from background import BackgroundStars

# =======================
# Shared State (global dict for params)
# =======================
galaxy_params = {
    "star_speed": 0,
    "galaxy_size": 0,
    "galaxy_brightness": 0,
    "galaxy_color_temp": 0,
    "galaxy_count": 0,
    "input_started": False # Flag to indicate if input has started
}

# Camera trajectory parameters - ADJUST THESE TO CONTROL CAMERA BEHAVIOR
camera_params = {
    "auto_orbit": True,      # Enable automatic orbit
    "orbit_speed": 0.3,      # Orbit speed (0.1-2.0) - Higher = faster rotation
    "zoom_speed": 0.1,       # Zoom speed (not currently used)
    "base_distance": 60,     # Base viewing distance (20-120) - Higher = further away
    "distance_amplitude": 20, # Distance variation range (10-40) - Higher = more zoom variation
    "elevation_amplitude": 30, # Elevation angle variation (10-60) - Higher = more up/down movement
    "time_scale": 1.0        # Overall time scale multiplier (0.5-3.0) - Higher = faster everything
}

# =======================
# Multi-Galaxy System Class
# =======================
class MultiGalaxySystem:
    def __init__(self, bounds=40):
        self.bounds = bounds
        self.galaxies = []
        self.initialize_galaxies()
        
    def initialize_galaxies(self):
        """Initialize 2-5 random galaxies"""
        num_galaxies = random.randint(2, 5)
        
        for i in range(num_galaxies):
            # Random initial positions (scattered in space)
            angle = (2 * np.pi * i / num_galaxies) + random.uniform(-0.5, 0.5)
            distance = random.uniform(15, 25)
            init_pos = np.array([
                np.cos(angle) * distance + random.uniform(-5, 5),
                np.sin(angle) * distance + random.uniform(-5, 5),
                random.uniform(-10, 10)
            ])
            
            # Random galaxy parameters
            galaxy_data = {
                'engine': GalaxyEngine(
                    num_particles=random.randint(800, 1500), 
                    bounds=self.bounds,
                    velocity_factor=random.uniform(0.02, 0.06)
                ),
                'center_pos': init_pos,
                'target_pos': np.array([0, 0, 0]),  # The target of all galaxies is the center
                'velocity': np.array([0.0, 0.0, 0.0]),
                'mass': random.uniform(800, 2000),
                'base_size': random.uniform(0.7, 1.3),  # Diameter multiples
                'base_color': self.generate_galaxy_color(),
                'collision_radius': random.uniform(12, 20),  # Increase the collision radius to make galaxies not merge easily
                'merged': False,
                'merge_timer': 0,
                'spiral_phase': random.uniform(0, 2*np.pi) 
            }
            
            # Set the initial position
            positions = galaxy_data['engine'].get_positions()
            positions += galaxy_data['center_pos']
            galaxy_data['engine'].positions = positions
            
            self.galaxies.append(galaxy_data)
    
    def generate_galaxy_color(self):
        """Generates a series of blue galaxy color (with small fluctuations)"""
        
        base_blue = [0.3, 0.5, 0.7]  # Base blue color
        
        # Color fluctuation adjustment parameter (range: 0.05-0.3)
        COLOR_VARIATION = 0.15  # Larger = more obvious fluctuations, smaller = more similar
        
        # Add small random fluctuations, maintaining the blue color
        return [max(0.2, min(0.8, base_blue[i] + random.uniform(-COLOR_VARIATION, COLOR_VARIATION))) for i in range(3)]
    
    def update(self, dt, user_params):
        """Update movement and collisions of all galaxies"""
        collision_strength = user_params["star_speed"] * 0.3 + 0.1
        size_influence = user_params["galaxy_size"]
        
        # Update the movements of all galaxies
        for i, galaxy in enumerate(self.galaxies):
            if galaxy['merged']:
                galaxy['merge_timer'] += dt
                continue
                
            # Calculate the gravitational force toward the center
            to_center = galaxy['target_pos'] - galaxy['center_pos']
            center_distance = np.linalg.norm(to_center)
            
            if center_distance > 1.0:
                # Galaxy gathering speed adjustment parameters
                APPROACH_SPEED = 3  # Larger = faster gathering, smaller = slower gathering (recommended range: 1.5-4.0)
                VELOCITY_DAMPING = 0.95  # Velocity damping, smaller for smoother movement (recommended range: 0.9-0.98)
                SPIRAL_STRENGTH = 0.1  # Spiral movement strength (recommended range: 0.05-0.3)
                
                # Force for moving toward the center
                center_force = to_center / center_distance * collision_strength * APPROACH_SPEED
                
                # Add slight spiral motion
                galaxy['spiral_phase'] += dt * 0.5
                spiral_offset = np.array([
                    np.cos(galaxy['spiral_phase']) * 2,
                    np.sin(galaxy['spiral_phase']) * 2,
                    0
                ]) * (center_distance / 20)
                
                galaxy['velocity'] = galaxy['velocity'] * VELOCITY_DAMPING + (center_force + spiral_offset * SPIRAL_STRENGTH) * dt
                galaxy['center_pos'] += galaxy['velocity'] * dt
            
            # Check for collisions with other galaxies
            for j, other in enumerate(self.galaxies):
                if i >= j or other['merged']:
                    continue
                    
                distance = np.linalg.norm(galaxy['center_pos'] - other['center_pos'])
                
                # Collision detection adjustment parameter 
                COLLISION_THRESHOLD_MULTIPLIER = 1.5  # Collision distance multiplier (larger = harder to collide, recommended range: 1.0-3.0)
                collision_threshold = (galaxy['collision_radius'] + other['collision_radius']) * COLLISION_THRESHOLD_MULTIPLIER
                
                if distance < collision_threshold:
                    # Start merging
                    self.merge_galaxies(galaxy, other, dt)
            
            # Update galaxies' interior particle system
            galaxy['engine'].update(dt=dt * (0.5 + collision_strength))
            positions = galaxy['engine'].get_positions()
            
            # Apply the influence in scale
            scale_factor = galaxy['base_size'] * (0.7 + size_influence * 0.6)
            center = positions.mean(axis=0)
            positions = center + (positions - center) * scale_factor
            
            # Noise and flatten effect
            noise = np.random.normal(0, 0.45, positions.shape)
            positions += noise
            positions[:, 2] *= 0.2  # Flatten the Z axis to 20%
            
            # Apply the center offset
            positions += galaxy['center_pos']
            
            galaxy['current_positions'] = positions
    
    def merge_galaxies(self, galaxy1, galaxy2, dt):
        """Manage the galaxy mergers"""
        # Only when galaxies are very close do they merge
        distance = np.linalg.norm(galaxy1['center_pos'] - galaxy2['center_pos'])
        
        # Merge distance threshold - must be very close to merge
        MERGE_DISTANCE_THRESHOLD = 5.0  # Smaller = easier to merge, larger = harder to merge
        
        if distance > MERGE_DISTANCE_THRESHOLD:
            return  # If the distance is too far, do not merge
        
        # Mark smaller galaxies as merged
        if galaxy1['mass'] < galaxy2['mass']:
            galaxy1['merged'] = True
            survivor = galaxy2
            merged = galaxy1
        else:
            galaxy2['merged'] = True
            survivor = galaxy1
            merged = galaxy2
        
        # Merge Mass and Particles
        survivor['mass'] += merged['mass'] * 0.5  # Reduce mass growth
        survivor['collision_radius'] += merged['collision_radius'] * 0.2  # Reduce radius growth
        
        # Create new merged positions (quality weighted)
        total_mass = survivor['mass'] + merged['mass']
        new_center = (survivor['center_pos'] * survivor['mass'] + 
                     merged['center_pos'] * merged['mass']) / total_mass
        survivor['center_pos'] = new_center
        
        # Mix the color
        for i in range(3):
            survivor['base_color'][i] = (survivor['base_color'][i] + merged['base_color'][i]) / 2
    
    def get_active_galaxies(self):
        """Get unmerged galaxies"""
        return [g for g in self.galaxies if not g['merged']]
    
    def reset_system(self):
        """Reset the whole galaxy system"""
        self.galaxies.clear()
        self.initialize_galaxies()

# =======================
# Camera Trajectory Controller Class
# =======================
class CameraController:
    def __init__(self, camera):
        self.camera = camera
        self.time = 0.0
        self.paused = False
        self.manual_mode = False  # For mouse control mode
        
        # Orbit patterns
        self.orbit_patterns = {
            'circular': self.circular_orbit,
            'figure8': self.figure8_orbit,
            'spiral': self.spiral_orbit,
            'cinematic': self.cinematic_orbit,
            'pendulum': self.pendulum_orbit,
            'zoom_orbit': self.zoom_orbit,  # NEW: Rotate + zoom in/out
            'manual': self.manual_control   # NEW: Mouse control
        }
        self.current_pattern = 'cinematic'
        self.pattern_keys = list(self.orbit_patterns.keys())
        self.pattern_index = 0  # Default to circular
        
    def circular_orbit(self, t):
        """Circular orbit"""
        azimuth = t * camera_params["orbit_speed"] * 60
        elevation = 15 + 10 * math.sin(t * 0.3)
        distance = camera_params["base_distance"] + 10 * math.sin(t * 0.5)
        return azimuth, elevation, distance
    
    def figure8_orbit(self, t):
        """Figure-8 orbit"""
        azimuth = 60 * math.sin(t * camera_params["orbit_speed"])
        elevation = 30 * math.sin(t * camera_params["orbit_speed"] * 2)
        distance = camera_params["base_distance"] + 15 * math.sin(t * 0.3)
        return azimuth, elevation, distance
    
    def spiral_orbit(self, t):
        """Spiral orbit"""
        azimuth = t * camera_params["orbit_speed"] * 30
        elevation = 20 * math.sin(t * 0.2) + 10
        distance = camera_params["base_distance"] + 20 * math.sin(t * 0.1)
        return azimuth, elevation, distance
    
    def cinematic_orbit(self, t):
        """Cinematic smooth orbit - MOST POPULAR"""
        # Slow orbit + smooth zoom + gentle pitch
        azimuth = t * camera_params["orbit_speed"] * 20
        elevation = 10 + 15 * math.sin(t * 0.15) * math.cos(t * 0.08)
        zoom_cycle = 0.7 + 0.3 * math.sin(t * 0.1)  # Between 0.4 and 1.0
        distance = camera_params["base_distance"] * zoom_cycle
        return azimuth, elevation, distance
    
    def pendulum_orbit(self, t):
        """Pendulum-style orbit"""
        azimuth = 45 * math.sin(t * camera_params["orbit_speed"] * 0.5)
        elevation = 20 + 25 * math.cos(t * 0.3)
        distance = camera_params["base_distance"] + 10 * math.sin(t * 0.4)
        return azimuth, elevation, distance
    
    def zoom_orbit(self, t):
        """Rotate while zooming in and out dramatically"""
        # Continuous slow rotation
        azimuth = t * camera_params["orbit_speed"] * 50
        
        # Gentle elevation change
        elevation = 15 + 10 * math.sin(t * 0.1)
        
        # Dramatic zoom cycle: far -> close -> far (takes about 30 seconds per cycle)
        zoom_phase = t * 1  # Slow zoom cycle
        # Create a smooth zoom that goes: 11x -> 1x -> 11x base distance
        zoom_factor = 1 + 10 * (0.5 + 0.5 * math.sin(zoom_phase))
        distance = camera_params["base_distance"] * zoom_factor
        
        return azimuth, elevation, distance
    
    def manual_control(self, t):
        """Manual mouse control - returns current camera settings"""
        # In manual mode, don't change anything - user controls with mouse
        return self.camera.azimuth, self.camera.elevation, self.camera.distance
    
    def update(self, dt):
        """Update camera position"""
        if not camera_params["auto_orbit"] or self.paused:
            return
        
        # Special handling for manual mode
        if self.current_pattern == 'manual':
            self.manual_mode = True
            return
        else:
            self.manual_mode = False
            
        self.time += dt * camera_params["time_scale"]
        
        # Get current orbit pattern parameters
        orbit_func = self.orbit_patterns[self.current_pattern]
        azimuth, elevation, distance = orbit_func(self.time)
        
        # Apply user input influence
        # Galaxy brightness affects zoom speed
        speed_multiplier = 1.0 + galaxy_params["galaxy_brightness"] * 0.5
        
        # Galaxy size affects viewing distance
        size_influence = galaxy_params["galaxy_size"]
        distance *= (0.8 + 0.4 * size_influence)
        
        # Movement speed affects orbit speed
        if hasattr(self, 'time'):
            movement_speed = galaxy_params["star_speed"]
            self.time += dt * movement_speed * 0.3
        
        # Update camera
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = distance
    
    def next_pattern(self):
        """Switch to next orbit pattern"""
        self.pattern_index = (self.pattern_index + 1) % len(self.pattern_keys)
        self.current_pattern = self.pattern_keys[self.pattern_index]
        
        # Special message for manual mode
        if self.current_pattern == 'manual':
            print(f"Camera orbit mode: {self.current_pattern} (Use mouse to control)")
        else:
            print(f"Camera orbit mode: {self.current_pattern}")
    
    def toggle_pause(self):
        """Pause/resume camera movement"""
        self.paused = not self.paused
        print(f"Camera movement: {'paused' if self.paused else 'resumed'}")
    
    def is_manual_mode(self):
        """Check if currently in manual control mode"""
        return self.current_pattern == 'manual'

# =======================
# Input Thread (Mediapipe)
# =======================
def run_input():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

    prev_landmarks = None

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape
            debug_info = {}

            if results.pose_landmarks:
                    # Mark input as started
                    galaxy_params["input_started"] = True
                    
                    landmarks = results.pose_landmarks.landmark

                    # ---- 1. Moving speed -> Star speed ----
                    # Detection points: NOSE (0), LEFT_SHOULDER (11), RIGHT_SHOULDER (12), LEFT_WRIST (15), RIGHT_WRIST (16)
                    # Method: Calculate pixel displacement for each point between frames, then average
                    if prev_landmarks is not None:
                        diffs = []
                        key_points = [
                            mp_pose.PoseLandmark.NOSE.value,          # Point 0: Face center
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value,  # Point 11: Left shoulder joint
                            mp_pose.PoseLandmark.RIGHT_SHOULDER.value, # Point 12: Right shoulder joint
                            mp_pose.PoseLandmark.LEFT_WRIST.value,     # Point 15: Left wrist
                            mp_pose.PoseLandmark.RIGHT_WRIST.value     # Point 16: Right wrist
                        ]
                        
                        for i in key_points:
                            if i < len(landmarks) and i < len(prev_landmarks):
                                dx = (landmarks[i].x - prev_landmarks[i].x) * w
                                dy = (landmarks[i].y - prev_landmarks[i].y) * h
                                diffs.append(np.sqrt(dx**2 + dy**2))
                        
                        if diffs:
                            moving_speed = np.mean(diffs)
                            debug_info["raw_speed"] = moving_speed
                            galaxy_params["star_speed"] = min(moving_speed / 8.0, 1.0)

                    prev_landmarks = [lm for lm in landmarks]

                    # ---- 2. Range of motion -> Galaxy size ----
                    # Detection: Core body landmarks only for more reasonable range
                    # Method: Calculate range using torso and arm landmarks only
                    core_points = [
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value,   # 11
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,  # 12
                        mp_pose.PoseLandmark.LEFT_ELBOW.value,      # 13
                        mp_pose.PoseLandmark.RIGHT_ELBOW.value,     # 14
                        mp_pose.PoseLandmark.LEFT_WRIST.value,      # 15
                        mp_pose.PoseLandmark.RIGHT_WRIST.value,     # 16
                        mp_pose.PoseLandmark.LEFT_HIP.value,        # 23
                        mp_pose.PoseLandmark.RIGHT_HIP.value        # 24
                    ]
                    
                    valid_core_landmarks = []
                    for point_idx in core_points:
                        if point_idx < len(landmarks) and landmarks[point_idx].visibility > 0.5:
                            valid_core_landmarks.append((landmarks[point_idx].x, landmarks[point_idx].y))
                    
                    debug_info["valid_core_count"] = len(valid_core_landmarks)
                    
                    if len(valid_core_landmarks) > 2:
                        xs, ys = zip(*valid_core_landmarks)
                        motion_range_x = (max(xs) - min(xs)) * w  # Horizontal span in pixels
                        motion_range_y = (max(ys) - min(ys)) * h  # Vertical span in pixels
                        total_motion_range = motion_range_x + motion_range_y
                        
                        debug_info["motion_range_x"] = motion_range_x
                        debug_info["motion_range_y"] = motion_range_y
                        debug_info["total_motion_range"] = total_motion_range
                        
                        # Use reference code threshold of 1500
                        galaxy_params["galaxy_size"] = min(total_motion_range / 1500.0, 1.0)

                    # ---- 3. Size in camera -> Galaxy brightness ----
                    # Detection points: LEFT_SHOULDER (11) and RIGHT_SHOULDER (12)
                    # Method: Measure horizontal distance between shoulders in pixels
                    # Logic: Closer to camera = wider shoulder span = higher brightness
                    try:
                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]   # Point 11
                        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] # Point 12
                        
                        shoulder_dist = abs(left_shoulder.x - right_shoulder.x) * w
                        
                        debug_info["left_shoulder_x"] = left_shoulder.x * w
                        debug_info["right_shoulder_x"] = right_shoulder.x * w
                        debug_info["shoulder_dist"] = shoulder_dist
                        
                        # Use reference code threshold of 1000
                        galaxy_params["galaxy_brightness"] = min(shoulder_dist / 750.0, 1.0)
                        
                    except Exception as e:
                        debug_info["brightness_error"] = str(e)

                    # ---- 4. Distance to center -> Galaxy color temperature ----
                    # Detection point: NOSE (0) - most stable facial landmark
                    # Method: Euclidean distance from nose to frame center (0.5, 0.5) with expanded center zone
                    # Logic: Expanded center area = cool colors, far edges = warm colors
                    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]  # Point 0: Nose tip
                    dist_to_center = np.sqrt((nose.x-0.5)**2 + (nose.y-0.5)**2)
                    debug_info["dist_to_center"] = dist_to_center
                    adjusted_distance = max(0, dist_to_center - 0.2)
                    galaxy_params["galaxy_color_temp"] = min(adjusted_distance * 3, 1.0)

                    # ---- 5. Hand distance -> Galaxy count ----
                    # Detection points: LEFT_WRIST (15) and RIGHT_WRIST (16)
                    # Method: Euclidean distance between wrist positions
                    # Logic: Hands apart = more galaxies, hands together = fewer galaxies
                    try:
                        lh = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]  # Point 15: Left wrist joint
                        rh = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value] # Point 16: Right wrist joint
                        hand_dist = np.sqrt((lh.x-rh.x)**2 + (lh.y-rh.y)**2)
                        
                        debug_info["hand_dist"] = hand_dist
                        debug_info["hand_dist_pixels"] = hand_dist * w  # Show pixel distance for debugging
                        
                        # Use reference code multiplier of 2
                        galaxy_params["galaxy_count"] = min(hand_dist * 1.2, 1.0)
                        
                    except Exception as e:
                        debug_info["hand_error"] = str(e)


                    # Draw skeleton
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


            # Display galaxy parameters with descriptions
            display_names = {
                "star_speed": "Moving speed (Collision force)",
                "galaxy_size": "Range of motion (Galaxy size)", 
                "galaxy_brightness": "Size in camera (Brightness)",
                "galaxy_color_temp": "Dist. to center (Color variation)",
                "galaxy_count": "Hand distance (Interaction strength)",
            }

            y0 = 30
            for i, (k, v) in enumerate(galaxy_params.items()):
                if k == "input_started": # Skip internal state marker
                    continue
                color = (0, int(255 * v), int(255 * (1-v)))
                cv2.putText(image, f"{display_names[k]}: {v:.3f}", (10, y0 + 30*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            # Display debug info on the right side
            debug_y = 30
            for key, value in debug_info.items():
                display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                cv2.putText(image, f"{key}: {display_value}", 
                           (w-400, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
                debug_y += 20

            # Add camera control instructions
            controls_text = [
                "Press ESC to exit, R to reset galaxies",
                "C: Switch camera orbit pattern",
                "SPACE: Pause/Resume camera movement", 
                f"Current mode: {camera_controller.current_pattern if 'camera_controller' in locals() else 'Auto Orbit'}"
            ]
            
            for i, text in enumerate(controls_text):
                cv2.putText(image, text, (10, h - 120 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow('Multi-Galaxy Input', image)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

# =======================
# Output Thread (Multi-Galaxy Vispy)
# =======================
def run_output():
    # Background star range adjustment parameters
    BACKGROUND_RANGE = 60  # Background star distribution range (larger = wider distribution, recommended range: 40-100)
    BACKGROUND_STAR_COUNT = 3000  # Number of background stars
    
    bg = BackgroundStars(n_stars=BACKGROUND_STAR_COUNT, bounds=BACKGROUND_RANGE, size=5, color=(1,1,1,0.6))
    canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), show=True)
    view = canvas.central_widget.add_view()
    
    # Use TurntableCamera and create camera controller
    view.camera = scene.cameras.TurntableCamera(up='+z', fov=45, distance=60)
    camera_controller = CameraController(view.camera)
    
    # Enable mouse interaction for manual mode
    view.camera.interactive = True

    # Add the background stars first
    scatter_bg = scene.visuals.Markers()
    scatter_bg.set_data(bg.positions, face_color=bg.colors, size=bg.sizes, edge_color=None)
    view.add(scatter_bg)

    # Multiple galaxy system
    multi_galaxy = MultiGalaxySystem(bounds=40)
    galaxy_scatters = []

    def reset_galaxies():
        """Reset the galaxy system"""
        # Clear old visuals
        for scatter in galaxy_scatters:
            try:
                view.remove(scatter)
            except:
                pass
        galaxy_scatters.clear()
        
        # Reinitialization
        multi_galaxy.reset_system()
        
        # Create new visuals
        for _ in multi_galaxy.galaxies:
            scatter_g = scene.visuals.Markers()
            view.add(scatter_g)
            galaxy_scatters.append(scatter_g)

    # Initialize galaxy visual object
    reset_galaxies()

    def update(ev):
        if not galaxy_params["input_started"]:
            return
            
        # Update camera trajectory (only if not in manual mode)
        if not camera_controller.is_manual_mode():
            camera_controller.update(1/60.0)  # Assume 60FPS
            
        # Update multiple galaxy system
        dt = galaxy_params["star_speed"] * 0.02 + 0.005
        multi_galaxy.update(dt, galaxy_params)
        
        # Updated visuals for each galaxy
        active_galaxies = multi_galaxy.get_active_galaxies()
        
        for i, (galaxy, scatter) in enumerate(zip(active_galaxies, galaxy_scatters[:len(active_galaxies)])):
            if 'current_positions' not in galaxy:
                continue
                
            positions = galaxy['current_positions']
            
            # Calculate color - cool and warm based on color_temp parameter
            color_temp = galaxy_params["galaxy_color_temp"]
            brightness = galaxy_params["galaxy_brightness"]
            
            # Start with a base blue color
            base_color = galaxy['base_color'].copy()
            
            # Apply warm and cool tones: color_temp larger = warm (reddish), smaller = cool (bluish)
            if color_temp > 0.5:  # Warm
                warmth_factor = (color_temp - 0.5) * 2  # 0 to 1
                base_color[0] += warmth_factor * 0.4  # Add red
                base_color[2] -= warmth_factor * 0.3  # Reduce blue
            else:  # Cool
                coolness_factor = (0.5 - color_temp) * 2  # 0 to 1
                base_color[2] += coolness_factor * 0.3  # Add blue
                base_color[0] -= coolness_factor * 0.2  # Reduce red
            
            # Make sure the color is within the valid range
            base_color = [max(0.1, min(1.0, c)) for c in base_color]
            base_color.append(0.3 + 0.7 * brightness)  # alpha
            
            
            # Set Color - All particles have the same color
            colors = np.tile(base_color, (len(positions), 1))
            
            
            scatter.set_data(positions, face_color=colors, size=6, edge_color=None)

            
        
        # Hide extra scatter objects
        for j, scatter in enumerate(galaxy_scatters[len(active_galaxies):]):
            # Make sure hidden objects do not affect rendering
            scatter.set_data(np.array([[1000, 1000, 1000]]), face_color=(0,0,0,0), size=0)
        
        # Update background stars - ensure background stars are displayed
        galaxy_centers = []
        for galaxy in active_galaxies:
            if 'center_pos' in galaxy:
                galaxy_centers.append({
                    "pos": galaxy['center_pos'], 
                    "mass": galaxy['mass'] * (1 + galaxy_params["galaxy_count"] * 2)
                })
        
        # Refresh background stars
        if galaxy_centers:
            bg.refresh_canvas(scatter_bg, galaxies=galaxy_centers)
        
        # Apply background brightness
        bg_brightness = 0.07 + 0.9 * galaxy_params["galaxy_brightness"]
        bg_colors = bg.colors.copy()
        bg_colors[:, 3] = bg_colors[:, 3] * bg_brightness
        scatter_bg.set_data(bg.positions, face_color=bg_colors, size=bg.sizes, edge_color=None)

    # Keyboard controls
    def on_key_press(event):
        if event.text.lower() == 'r':
            reset_galaxies()
        elif event.text.lower() == 'c':
            camera_controller.next_pattern()
            # Enable/disable camera interactivity based on mode
            view.camera.interactive = camera_controller.is_manual_mode()
        elif event.text == ' ':  # Space key
            camera_controller.toggle_pause()
        elif event.text.lower() == 'o':
            camera_params["auto_orbit"] = not camera_params["auto_orbit"]
            print(f"Auto orbit: {'enabled' if camera_params['auto_orbit'] else 'disabled'}")
        elif event.text == '+' or event.text == '=':
            camera_params["orbit_speed"] = min(camera_params["orbit_speed"] * 1.2, 2.0)
            print(f"Orbit speed: {camera_params['orbit_speed']:.2f}")
        elif event.text == '-':
            camera_params["orbit_speed"] = max(camera_params["orbit_speed"] * 0.8, 0.05)
            print(f"Orbit speed: {camera_params['orbit_speed']:.2f}")
        elif event.text.lower() == 'z':
            camera_params["base_distance"] = max(camera_params["base_distance"] - 5, 20)
            print(f"Base distance: {camera_params['base_distance']}")
        elif event.text.lower() == 'x':
            camera_params["base_distance"] = min(camera_params["base_distance"] + 5, 120)
            print(f"Base distance: {camera_params['base_distance']}")

    canvas.events.key_press.connect(on_key_press)
    timer = app.Timer(interval=1/60.0, connect=update, start=True)
    app.run()

# =======================
# Main
# =======================
if __name__ == "__main__":
    print("=== Multi-Galaxy System Camera Control ===")
    print("Keyboard Controls:")
    print("  R - Reset galaxies")
    print("  C - Switch camera orbit pattern")
    print("  SPACE - Pause/Resume camera movement")
    print("  O - Enable/Disable auto orbit")
    print("  +/- - Increase/Decrease orbit speed")
    print("  Z/X - Zoom in/out (adjust viewing distance)")
    print("")
    print("Camera Orbit Patterns:")
    print("  1. circular - Circular orbit")
    print("  2. figure8 - Figure-8 trajectory")
    print("  3. spiral - Spiral trajectory")
    print("  4. cinematic - Cinematic smooth orbit (default)")
    print("  5. pendulum - Pendulum-style orbit")
    print("  6. zoom_orbit - Rotate with dramatic zoom in/out")
    print("  7. manual - Manual mouse control")
    print("")
    print("Mouse Controls (in manual mode):")
    print("  - Left click + drag: Rotate view")
    print("  - Right click + drag: Zoom in/out")
    print("  - Middle click + drag: Pan")
    print("")
    print("Camera Speed Adjustment Guide:")
    print("  Edit 'camera_params' dictionary at the top of the file:")
    print("  - orbit_speed: 0.1-2.0 (rotation speed)")
    print("  - base_distance: 20-120 (viewing distance)")
    print("  - time_scale: 0.5-3.0 (overall speed multiplier)")
    print("  - distance_amplitude: 10-40 (zoom variation)")
    print("  - elevation_amplitude: 10-60 (up/down movement)")
    print("")
    
    t1 = threading.Thread(target=run_input, daemon=True)
    t1.start()
    run_output()