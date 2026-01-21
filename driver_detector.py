import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pygame

class DriverDrowsinessDetector:
    """
    A class to detect driver drowsiness, yawning, and distraction using computer vision.
    
    Attributes:
        ear_threshold (float): Eye Aspect Ratio threshold for drowsiness.
        ear_consec_frames (int): Number of frames EAR must be below threshold.
        mar_threshold (float): Mouth Aspect Ratio threshold for yawning.
        distraction_threshold (float): Horizontal nose offset threshold for distraction.
        alarm_file (str): Path to the sound file for the alarm.
    """
    
    # --- Landmark Indices (MediaPipe Face Mesh) ---
    # Left Eye (p1..p6)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 380, 373]
    # Right Eye (p1..p6)
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    # Mouth (p1..p8)
    MOUTH_INDICES = [78, 82, 13, 312, 308, 191, 80, 88]

    def __init__(self, alarm_file="mixkit-classic-alarm-995.wav", webcam_index=1):
        print("Initializing Driver Drowsiness Detector...")
        
        # --- Configuration ---
        self.ear_threshold = 0.30
        self.ear_consec_frames = 10
        self.mar_threshold = 0.60
        self.distraction_threshold = 0.05
        self.distraction_timeout = 2.0
        
        # --- State Variables ---
        self.counter = 0              # Frames drowsy
        self.yawn_counter = 0         # Frames yawning
        self.alarm_on = False         # Is alarm currently playing?
        self.distraction_start_time = None
        self.driver_status = "NORMAL"
        self.running = True
        
        # --- Initialization ---
        self._init_mediapipe()
        self._init_sound(alarm_file)
        self._init_camera(webcam_index)
        
        print("System Ready. Press 'q' to exit.")

    def _init_mediapipe(self):
        """Sets up MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _init_sound(self, alarm_file):
        """Sets up Pygame mixer and loads the alarm sound."""
        try:
            pygame.mixer.init()
            self.alarm_sound = pygame.mixer.Sound(alarm_file)
            print(f"Successfully loaded alarm sound: {alarm_file}")
        except Exception as e:
            print(f"Warning: Could not load sound file '{alarm_file}': {e}")
            print("Alarm will be silent.")
            self.alarm_sound = None

    def _init_camera(self, index):
        """Opens the webcam connection."""
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Please check your connection.")

    def _sound_alarm_thread(self):
        """
        Thread function to play the alarm sound. 
        Runs in a separate thread to avoid blocking the video feed.
        """
        if self.alarm_sound:
            channel = self.alarm_sound.play()
            start_time = time.time()
            
            # Keep thread alive while playing, but limit max duration to 2.5s per trigger
            while channel.get_busy() and (time.time() - start_time < 2.5):
                time.sleep(0.1)
                
            self.alarm_sound.stop()
        
        # Cooldown prevents immediate re-trigger
        time.sleep(1.0) 
        self.alarm_on = False

    def trigger_alarm(self):
        """Starts the alarm thread if not already running."""
        if not self.alarm_on:
            self.alarm_on = True
            t = threading.Thread(target=self._sound_alarm_thread)
            t.daemon = True
            t.start()

    @staticmethod
    def euclidean_distance(ptA, ptB):
        """Helper to calculate distance between two points."""
        return np.linalg.norm(np.array(ptA) - np.array(ptB))

    def calculate_ear(self, eye_points):
        """Calculates Eye Aspect Ratio (EAR)."""
        # Vertical distances
        A = self.euclidean_distance(eye_points[1], eye_points[5])
        B = self.euclidean_distance(eye_points[2], eye_points[4])
        # Horizontal distance
        C = self.euclidean_distance(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def calculate_mar(self, mouth_points):
        """Calculates Mouth Aspect Ratio (MAR)."""
        # Vertical distances
        A = self.euclidean_distance(mouth_points[1], mouth_points[7])
        B = self.euclidean_distance(mouth_points[2], mouth_points[6])
        C = self.euclidean_distance(mouth_points[3], mouth_points[5])
        # Horizontal distance
        D = self.euclidean_distance(mouth_points[0], mouth_points[4])
        return (A + B + C) / (3.0 * D)

    def process_frame(self, image):
        """
        Main logic to process a single frame:
        1. Find face landmarks.
        2. Calculate EAR/MAR.
        3. Update drowsiness/distraction state.
        4. Draw overlays.
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        
        # Draw on original BGR image
        h, w, c = image.shape
        self.driver_status = "NORMAL" # Reset status for this frame

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # --- Helper to get pixel coordinates ---
                def get_points(indices):
                    points = []
                    for i in indices:
                        lm = face_landmarks.landmark[i]
                        points.append((int(lm.x * w), int(lm.y * h)))
                    return points

                # --- 1. Extract Features ---
                left_eye_pts = get_points(self.LEFT_EYE_INDICES)
                right_eye_pts = get_points(self.RIGHT_EYE_INDICES)
                mouth_pts = get_points(self.MOUTH_INDICES)

                # --- 2. Calculate Metrics ---
                left_ear = self.calculate_ear(left_eye_pts)
                right_ear = self.calculate_ear(right_eye_pts)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_mar(mouth_pts)

                # --- 3. Drowsiness Detection (Eyes) ---
                if avg_ear < self.ear_threshold:
                    self.counter += 1
                    if self.counter >= self.ear_consec_frames:
                        self.driver_status = "DROWSY!"
                else:
                    self.counter = 0

                # --- 4. Yawn Detection (Mouth) ---
                if mar > self.mar_threshold:
                    self.yawn_counter += 1
                    if self.yawn_counter >= 10:
                        self.driver_status = "YAWNING!"
                else:
                    self.yawn_counter = 0

                # --- 5. Distraction Detection (Head Pose) ---
                nose_tip_x = face_landmarks.landmark[4].x
                left_eye_outer = face_landmarks.landmark[33].x
                right_eye_outer = face_landmarks.landmark[263].x
                face_center_x = (left_eye_outer + right_eye_outer) / 2
                
                distraction_offset = abs(nose_tip_x - face_center_x)

                if distraction_offset > self.distraction_threshold:
                    if self.distraction_start_time is None:
                        self.distraction_start_time = time.time()
                    elif time.time() - self.distraction_start_time > self.distraction_timeout:
                        if self.driver_status == "NORMAL":
                            self.driver_status = "DISTRACTED!"
                else:
                    self.distraction_start_time = None

                # --- 6. Visualization ---
                # Draw landmarks
                for pt in left_eye_pts + right_eye_pts + mouth_pts:
                    cv2.circle(image, pt, 1, (0, 255, 0), -1)
                
                # Display Metrics
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"MAR: {mar:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, f"Distract: {distraction_offset:.3f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- 7. Alarm Logic ---
        color = (0, 255, 0) # Green
        if self.driver_status in ["DROWSY!", "DISTRACTED!", "YAWNING!"]:
            color = (0, 0, 255) # Red
            self.trigger_alarm()
            
        cv2.putText(image, f"STATUS: {self.driver_status}", (w // 2 - 100, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        return image

    def run(self):
        """Main loop to capture frames and process them."""
        while self.cap.isOpened() and self.running:
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Process the frame
            image = self.process_frame(image)
            
            # Show the output
            cv2.imshow('Driver Monitoring', image)
            
            # Quit on 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        self.cleanup()

    def cleanup(self):
        """Releases resources."""
        print("Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        print("Terminated successfully.")

if __name__ == "__main__":
    # Create an instance and run it
    detector = DriverDrowsinessDetector()
    detector.run()