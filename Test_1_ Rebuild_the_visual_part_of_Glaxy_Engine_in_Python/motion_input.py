import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_time = time.time()
    prev_landmarks = None
    motion_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # RGB back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        galaxy_params = {
            "star_speed": 0,
            "galaxy_size": 0,
            "galaxy_brightness": 0,
            "galaxy_color_temp": 0,
            "galaxy_count": 0,
            "galaxy_stability": 0
        }

        # Debug info
        debug_info = {}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # ---- 1. Moving speed -> Star speed ----
            if prev_landmarks is not None:
                diffs = []
                key_points = [
                    mp_pose.PoseLandmark.NOSE.value,
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                    mp_pose.PoseLandmark.LEFT_WRIST.value,
                    mp_pose.PoseLandmark.RIGHT_WRIST.value
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
            valid_landmarks = [(lm.x, lm.y) for lm in landmarks if lm.visibility > 0.5]
            debug_info["valid_landmark_count"] = len(valid_landmarks)
            
            if len(valid_landmarks) > 0:
                xs, ys = zip(*valid_landmarks)
                motion_range_x = (max(xs) - min(xs)) * w
                motion_range_y = (max(ys) - min(ys)) * h
                total_motion_range = motion_range_x + motion_range_y
                
                # Debug values
                debug_info["motion_range_x"] = motion_range_x
                debug_info["motion_range_y"] = motion_range_y
                debug_info["total_motion_range"] = total_motion_range
                
                galaxy_params["galaxy_size"] = min(total_motion_range / 1500.0, 1.0)

            # ---- 3. Size in camera -> Galaxy brightness ----
            #  Head-shoulder distance
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            shoulder_dist = abs(left_shoulder.x - right_shoulder.x) * w

            # Debug values
            debug_info["left_shoulder_x"] = left_shoulder.x * w
            debug_info["right_shoulder_x"] = right_shoulder.x * w
            debug_info["shoulder_dist"] = shoulder_dist

            galaxy_params["galaxy_brightness"] = min(shoulder_dist / 1000.0, 1.0)

            # ---- 4. Distance to center -> Galaxy color temperature ----
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            dist_to_center = np.sqrt((nose.x-0.5)**2 + (nose.y-0.5)**2)
            galaxy_params["galaxy_color_temp"] = min(dist_to_center * 3, 1.0)

            # ---- 5. Hand distance -> Galaxy count ----
            lh = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            rh = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            hand_dist = np.sqrt((lh.x-rh.x)**2 + (lh.y-rh.y)**2)
            galaxy_params["galaxy_count"] = min(hand_dist * 2, 1.0)

            # ---- 6. Stability -> Galaxy stability ----
            motion_history.append(galaxy_params["star_speed"])
            if len(motion_history) > 15:
                motion_history.pop(0)
            
            if len(motion_history) > 1:
                stability_variance = np.var(motion_history)
                galaxy_params["galaxy_stability"] = max(0, 1.0 - stability_variance * 100)

            # Draw skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display galaxy parameters with action names
        display_names = {
            "star_speed": "Moving speed (Star speed)",
            "galaxy_size": "Range of motion (Galaxy size)", 
            "galaxy_brightness": "Size in camera (Galaxy brightness)",
            "galaxy_color_temp": "Dist. to center (Color temp)",
            "galaxy_count": "Hand distance (Galaxy count)",
            "galaxy_stability": "Stability (Galaxy stability)"
        }

        y0 = 30
        for i, (k, v) in enumerate(galaxy_params.items()):
            # Color coding: red for low values, green for high values
            color = (0, int(255 * v), int(255 * (1-v)))
            cv2.putText(image, f"{display_names[k]}: {v:.3f}", (10, y0 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display debug info on the right side
        debug_y = 30
        for key, value in debug_info.items():
            cv2.putText(image, f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}", 
                       (w-300, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
            debug_y += 20

        # Add instructions
        cv2.putText(image, "Press ESC to exit", (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow('Cosmos Vision Input', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()