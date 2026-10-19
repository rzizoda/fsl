import cv2
import mediapipe as mp
import pyttsx3
import joblib
import os
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

MODEL_FILE = "fsl_model.joblib"

if not os.path.exists(MODEL_FILE):
    messagebox.showerror("Error", f"'{MODEL_FILE}' not found. Please collect data and run train_model.py first!")
    exit(1)

model = joblib.load(MODEL_FILE)
engine = pyttsx3.init()
engine.setProperty("rate", 160)

# Variables to control automatic speech so it doesn't repeat infinitely
last_spoken_sign = ""
last_speech_time = 0

def speak(text):
    engine.say(text)
    engine.runAndWait()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

class FSLAutoDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("FSL Auto-Detector (A-Z)")
        self.window.geometry("800x650")

        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        self.current_prediction = ""

        self.video_label = tk.Label(window)
        self.video_label.pack(pady=10)

        self.prediction_label = tk.Label(window, text="Auto-Detecting Sign: ---", font=("Arial", 18, "bold"), fg="purple")
        self.prediction_label.pack(pady=5)

        # Controls Panel
        btn_frame = tk.Frame(window)
        btn_frame.pack(fill="x", pady=10)

        self.btn_switch = tk.Button(btn_frame, text="Switch Camera", command=self.switch_camera, bg="#2196F3", fg="white", font=("Arial", 11))
        self.btn_switch.pack(side="left", expand=True, fill="x", padx=5)

        self.btn_exit = tk.Button(btn_frame, text="Exit", command=self.on_exit, bg="#f44336", fg="white", font=("Arial", 11, "bold"))
        self.btn_exit.pack(side="left", expand=True, fill="x", padx=5)

        self.update_video()

    def update_video(self):
        global last_spoken_sign, last_speech_time
        ret, frame = self.cap.read()
        
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = hand_landmarks.landmark
                    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
                    features = []
                    for lm in landmarks:
                        features.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                    # Auto-predict the sign
                    prediction = model.predict([features])[0]
                    self.current_prediction = prediction
                    self.prediction_label.config(text=f"Detected Sign: {prediction}")

                    # Auto-speak if the sign changes or 2 seconds have passed since last speech
                    current_time = time.time()
                    if prediction != last_spoken_sign or (current_time - last_speech_time > 2.0):
                        last_spoken_sign = prediction
                        last_speech_time = current_time
                        # Run speech in background/non-blocking manner if desired, or directly:
                        # (Note: engine.say blocks slightly, but works great for letters)
                        engine.say(prediction)
                        engine.runAndWait()
            else:
                self.current_prediction = ""
                self.prediction_label.config(text="Auto-Detecting Sign: ---")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_video)

    def switch_camera(self):
        self.cap.release()
        self.camera_index = 1 if self.camera_index == 0 else 0
        self.cap = cv2.VideoCapture(self.camera_index)

    def on_exit(self):
        self.cap.release()
        hands.close()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FSLAutoDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()
