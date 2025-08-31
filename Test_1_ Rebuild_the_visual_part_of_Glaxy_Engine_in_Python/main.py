import cv2
import mediapipe as mp
import numpy as np
import time
import threading

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
    "input_started": False  # Flag to indicate if input has started
}

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
    motion_history = []

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
                "star_speed": "Moving speed (Star speed)",
                "galaxy_size": "Range of motion (Galaxy size)", 
                "galaxy_brightness": "Size in camera (Galaxy brightness)",
                "galaxy_color_temp": "Dist. to center (Color temp)",
                "galaxy_count": "Hand distance (Galaxy count)",
            }

            y0 = 30
            for i, (k, v) in enumerate(galaxy_params.items()):
                if k == "input_started":  # Skip internal state marker
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

            cv2.putText(image, "Press ESC to exit", (10, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow('Input (Mediapipe)', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# =======================
# Output Thread (Vispy Galaxy)
# =======================
def run_output():
    # Background - use same bounds as galaxy to ensure size matching
    bg = BackgroundStars(n_stars=2000, bounds=40, size=5, color=(1,1,1,0.6))
    canvas = scene.SceneCanvas(keys='interactive', size=(1200,900), show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(up='+z', fov=45, distance=60)

    scatter_bg = scene.visuals.Markers()
    scatter_bg.set_data(bg.positions, face_color=bg.colors, size=bg.sizes, edge_color=None)
    view.add(scatter_bg)

    # Galaxy - reduce bounds to match background
    engine = GalaxyEngine(num_particles=2000, bounds=40, velocity_factor=0.04)  # Lower velocity for stability
    positions = engine.get_positions()
    scatter_g = scene.visuals.Markers()
    scatter_g.set_data(positions, face_color=(0.5,0.7,1,1), size=5, edge_color=None)  # Remove black edge
    view.add(scatter_g)

    # Fixed particle size
    FIXED_PARTICLE_SIZE = 6

    def update(ev):
        # Wait for input to start before updating animation
        if not galaxy_params["input_started"]:
            return
            
        # Use input parameters to drive galaxy behavior
        speed = galaxy_params["star_speed"] * 0.015 + 0.005  # Reduced speed for smoother motion
        color_temp = galaxy_params["galaxy_color_temp"]
        brightness = galaxy_params["galaxy_brightness"]
        size_factor = galaxy_params["galaxy_size"]
        count_factor = galaxy_params["galaxy_count"]
    

        # Color: Blue -> Red based on color temperature
        base_color = np.array([
            0.3 + 0.7 * color_temp,  # Red component
            0.5,                     # Green component
            0.7 - 0.4 * color_temp,  # Blue component
            0.3 + 0.7 * brightness   # Alpha (brightness)
        ])

        # Update galaxy with dynamic speed
        engine.update(dt=speed)
        positions = engine.get_positions()
        
        # Apply size factor to particle spread - keep within background bounds
        if size_factor > 0:
            center = positions.mean(axis=0)
            # Moderate scaling to maintain galaxy structure
            spread_factor = 0.6 + size_factor * 0.6  # Range from 0.6 to 1.2
            positions = center + (positions - center) * spread_factor
            
            # Add slight randomization to prevent clustering
            noise = np.random.normal(0, 0.45, positions.shape)
            positions += noise
            # Flatten the galaxy vertically to reduce towering effect
            positions[:, 2] *= 0.2  # Compress Z-axis to 30% of original height
        
        # Use fixed particle size and remove edge color
        c = np.tile(base_color, (len(positions), 1))
        scatter_g.set_data(positions, face_color=c, size=FIXED_PARTICLE_SIZE, edge_color=None)

        # Background stars affected by galaxy
        galaxy_mass = 1000 + count_factor * 3000  
        bg.refresh_canvas(scatter_bg, galaxies=[{"pos": positions.mean(axis=0), "mass": galaxy_mass}])
        
        # Apply brightness to background stars
        bg_brightness = 0.07 + 0.9 * brightness  
        bg_colors = bg.colors.copy()
        bg_colors[:, 3] = bg_colors[:, 3] * bg_brightness  # Modify alpha channel
        scatter_bg.set_data(bg.positions, face_color=bg_colors, size=bg.sizes, edge_color=None)

        

    timer = app.Timer(interval=1/60.0, connect=update, start=True)
    app.run()


# =======================
# Main: Run input & output in parallel
# =======================
if __name__ == "__main__":
    t1 = threading.Thread(target=run_input, daemon=True)
    t1.start()
    run_output()