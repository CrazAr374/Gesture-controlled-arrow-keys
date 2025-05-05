import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class GestureKeyboardController:
    def __init__(self):
        # Performance settings
        self.process_every_n_frames = 2  # Only process every nth frame
        self.frame_counter = 0
        self.reduced_resolution = True  # Process frames at lower resolution
        self.resolution_scale = 0.5  # Scale factor for resolution reduction
        
        # Initialize MediaPipe Hands with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Reduced from 0.7
            min_tracking_confidence=0.5,
            model_complexity=0  # Use the lightest model (0 = lightest, 1 = medium)
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize webcam with optimized resolution
        self.cap = cv2.VideoCapture(0)
        
        # Screen dimensions
        self.screen_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Region boundaries (percentages of screen)
        self.left_boundary = 0.3
        self.right_boundary = 0.7
        self.top_boundary = 0.3
        self.bottom_boundary = 0.7
        
        # Key mapping
        self.key_mapping = {
            "left": "left",
            "right": "right",
            "up": "up",
            "down": "down"
        }
        
        # Debounce settings
        self.last_action_time = 0
        self.debounce_time = 0.5  # seconds
        
        # Current region and action
        self.current_region = "center"
        self.last_region = "center"
        
        # Gesture detection
        self.last_gesture = None
        self.gesture_debounce_time = 1.0  # seconds
        self.last_gesture_time = 0
        
        # Smoothing - reduced history length
        self.position_history = []
        self.history_length = 3  # Reduced from 5
        
        # UI elements - simplified drawing
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6  # Reduced from 0.7
        self.font_thickness = 1  # Reduced from 2
        self.text_color = (255, 255, 255)
        self.box_color = (0, 0, 255)
        
        # Calibration mode
        self.calibration_mode = False
        
        # FPS calculation
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0
        
    def get_smoothed_position(self, x, y):
        """Apply smoothing to hand position"""
        self.position_history.append((x, y))
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        
        if not self.position_history:
            return x, y
        
        avg_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
        avg_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)
        
        return avg_x, avg_y
    
    def detect_region(self, x, y):
        """Detect which region the finger is in"""
        if x < self.screen_width * self.left_boundary:
            return "left"
        elif x > self.screen_width * self.right_boundary:
            return "right"
        elif y < self.screen_height * self.top_boundary:
            return "up"
        elif y > self.screen_height * self.bottom_boundary:
            return "down"
        else:
            return "center"
    
    def detect_gesture(self, hand_landmarks):
        """Detect specific hand gestures"""
        # Get all finger tip positions
        finger_tips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        # Get finger base positions
        finger_bases = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        ]
        
        # Check if fingers are extended
        fingers_extended = []
        for i in range(5):
            if i == 0:  # Thumb is special case
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                if finger_tips[i].x < finger_bases[i].x:  # For right hand
                    fingers_extended.append(True)
                else:
                    fingers_extended.append(False)
            else:
                if finger_tips[i].y < finger_bases[i].y:  # Finger is extended if tip is above base
                    fingers_extended.append(True)
                else:
                    fingers_extended.append(False)
        
        # Detect gestures based on extended fingers
        if all(fingers_extended):
            return "open_palm"
        elif not any(fingers_extended):
            return "fist"
        elif fingers_extended[0] and not any(fingers_extended[1:]):
            return "thumbs_up"
        elif fingers_extended[1] and not any([fingers_extended[0]] + fingers_extended[2:]):
            return "index_pointing"
        elif fingers_extended[1] and fingers_extended[2] and not any([fingers_extended[0]] + fingers_extended[3:]):
            return "peace"
        else:
            return "other"
    
    def process_gesture(self, gesture):
        """Process detected gestures and trigger keyboard actions"""
        current_time = time.time()
        
        if gesture != self.last_gesture and current_time - self.last_gesture_time > self.gesture_debounce_time:
            self.last_gesture = gesture
            self.last_gesture_time = current_time
            
            if gesture == "open_palm":
                pyautogui.press('space')
                return "Space"
            elif gesture == "fist":
                return "Stop Input"
            elif gesture == "thumbs_up":
                pyautogui.press('enter')
                return "Enter"
            elif gesture == "peace":
                pyautogui.press('escape')
                return "Escape"
        
        return None
    
    def draw_regions(self, frame):
        """Draw region boundaries on the frame - optimized version"""
        h, w, _ = frame.shape
        
        # Only draw outlines instead of full lines
        cv2.line(frame, (int(w * self.left_boundary), int(h * 0.45)), (int(w * self.left_boundary), int(h * 0.55)), (0, 255, 0), 2)
        cv2.line(frame, (int(w * self.right_boundary), int(h * 0.45)), (int(w * self.right_boundary), int(h * 0.55)), (0, 255, 0), 2)
        cv2.line(frame, (int(w * 0.45), int(h * self.top_boundary)), (int(w * 0.55), int(h * self.top_boundary)), (0, 255, 0), 2)
        cv2.line(frame, (int(w * 0.45), int(h * self.bottom_boundary)), (int(w * 0.55), int(h * self.bottom_boundary)), (0, 255, 0), 2)
        
        # Simplified labels - only draw when in calibration mode
        if self.calibration_mode:
            cv2.putText(frame, "L", (int(w * 0.15), int(h * 0.5)), self.font, self.font_scale, self.text_color, self.font_thickness)
            cv2.putText(frame, "R", (int(w * 0.85), int(h * 0.5)), self.font, self.font_scale, self.text_color, self.font_thickness)
            cv2.putText(frame, "U", (int(w * 0.5), int(h * 0.15)), self.font, self.font_scale, self.text_color, self.font_thickness)
            cv2.putText(frame, "D", (int(w * 0.5), int(h * 0.85)), self.font, self.font_scale, self.text_color, self.font_thickness)
            cv2.putText(frame, "C", (int(w * 0.5), int(h * 0.5)), self.font, self.font_scale, self.text_color, self.font_thickness)
    
    def draw_ui(self, frame, action=None, gesture=None):
        """Draw minimal UI elements on the frame"""
        h, w, _ = frame.shape
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, h - 60), self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Draw minimal action box
        if action:
            cv2.putText(frame, f"Action: {action}", (10, 25), self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Draw minimal gesture box
        if gesture:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), self.font, self.font_scale, self.text_color, self.font_thickness)
        
        # Minimal mode indicator
        mode_text = "Calibration" if self.calibration_mode else "Normal"
        cv2.putText(frame, mode_text, (w - 100, 25), self.font, self.font_scale, self.text_color, self.font_thickness)
    
    def toggle_calibration_mode(self):
        """Toggle calibration mode"""
        self.calibration_mode = not self.calibration_mode
    
    def adjust_boundaries(self, x, y):
        """Adjust region boundaries in calibration mode"""
        if not self.calibration_mode:
            return
        
        # Normalize coordinates
        norm_x = x / self.screen_width
        norm_y = y / self.screen_height
        
        # Find closest boundary and adjust it
        distances = [
            (abs(norm_x - self.left_boundary), "left"),
            (abs(norm_x - self.right_boundary), "right"),
            (abs(norm_y - self.top_boundary), "top"),
            (abs(norm_y - self.bottom_boundary), "bottom")
        ]
        
        closest = min(distances, key=lambda x: x[0])
        
        if closest[1] == "left":
            self.left_boundary = norm_x
        elif closest[1] == "right":
            self.right_boundary = norm_x
        elif closest[1] == "top":
            self.top_boundary = norm_y
        elif closest[1] == "bottom":
            self.bottom_boundary = norm_y
    
    def run(self):
        """Main loop to run the application - optimized for performance"""
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    print("Failed to capture frame from webcam")
                    break
                
                # Calculate FPS
                self.curr_frame_time = time.time()
                self.fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.curr_frame_time != self.prev_frame_time else 0
                self.prev_frame_time = self.curr_frame_time
                
                # Flip the frame horizontally for a more intuitive mirror view
                frame = cv2.flip(frame, 1)
                
                # Increment frame counter
                self.frame_counter += 1
                
                # Reduce resolution for processing
                if self.reduced_resolution:
                    process_frame = cv2.resize(frame, (0, 0), fx=self.resolution_scale, fy=self.resolution_scale)
                else:
                    process_frame = frame
                
                action = None
                detected_gesture = None
                
                # Only process every n frames
                if self.frame_counter % self.process_every_n_frames == 0:
                    # Convert the BGR image to RGB
                    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                    
                    # Process the frame with MediaPipe
                    results = self.hands.process(rgb_frame)
                    
                    # If hands are detected
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Get index finger tip coordinates and scale back to original size
                            index_finger = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            x, y = int(index_finger.x * self.screen_width), int(index_finger.y * self.screen_height)
                            
                            # Apply smoothing
                            smooth_x, smooth_y = self.get_smoothed_position(x, y)
                            
                            # Draw simple index finger position
                            cv2.circle(frame, (int(smooth_x), int(smooth_y)), 5, (0, 0, 255), -1)
                            
                            # Detect region
                            self.current_region = self.detect_region(smooth_x, smooth_y)
                            
                            # Detect gestures
                            gesture = self.detect_gesture(hand_landmarks)
                            gesture_action = self.process_gesture(gesture)
                            if gesture_action:
                                detected_gesture = gesture_action
                            
                            # In calibration mode, adjust boundaries
                            if self.calibration_mode:
                                self.adjust_boundaries(smooth_x, smooth_y)
                            # In normal mode, trigger keyboard actions
                            elif self.current_region != "center" and self.current_region != self.last_region:
                                current_time = time.time()
                                if current_time - self.last_action_time > self.debounce_time:
                                    self.last_action_time = current_time
                                    key = self.key_mapping.get(self.current_region)
                                    if key:
                                        pyautogui.press(key)
                                        action = f"Pressed {key}"
                            
                            self.last_region = self.current_region
                
                # Draw simplified UI elements
                self.draw_regions(frame)
                self.draw_ui(frame, action, detected_gesture)
                
                # Display the frame
                cv2.imshow('Hand Gesture Keyboard Controller', frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.toggle_calibration_mode()
                elif key == ord('p'):  # New key to toggle performance settings
                    self.process_every_n_frames = 1 if self.process_every_n_frames > 1 else 2
                    self.reduced_resolution = not self.reduced_resolution
                    print(f"Performance settings: Skip frames={self.process_every_n_frames}, Reduced resolution={self.reduced_resolution}")
        
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    print("Starting Hand Gesture Keyboard Controller...")
    print("Instructions:")
    print("- Move your index finger to different regions to trigger arrow keys")
    print("- Open palm: Press Spacebar")
    print("- Fist: Stop input")
    print("- Thumbs up: Press Enter")
    print("- Peace sign: Press Escape")
    print("- Press 'c' to toggle calibration mode")
    print("- Press 'p' to toggle performance settings")
    print("- Press 'q' to quit")
    
    controller = GestureKeyboardController()
    controller.run()