import cv2
import mediapipe as mp
import csv
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Setup Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

CSV_FILE = "fsl_dataset.csv"

# Write headers if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        headers = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(headers)


class DataCollectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("FSL Data Collector")
        self.window.geometry("800x650")

        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        self.current_features = None
        self.saved_count = 0

        # UI Layout
        self.label_prompt = tk.Label(window, text="Enter Sign Label (e.g., A, B, HELLO):", font=("Arial", 12))
        self.label_prompt.pack(pady=5)

        self.entry_label = tk.Entry(window, font=("Arial", 12), width=15)
        self.entry_label.insert(0, "A")
        self.entry_label.pack(pady=5)

        self.video_label = tk.Label(window)
        self.video_label.pack(pady=10)

        # Buttons Panel
        btn_frame = tk.Frame(window)
        btn_frame.pack(fill="x", pady=10)

        self.btn_save = tk.Button(btn_frame, text="Save Sample (Space)", command=self.save_sample, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
        self.btn_save.pack(side="left", expand=True, fill="x", padx=5)

        self.btn_switch = tk.Button(btn_frame, text="Switch Camera", command=self.switch_camera, bg="#2196F3", fg="white", font=("Arial", 11))
        self.btn_switch.pack(side="left", expand=True, fill="x", padx=5)

        self.btn_exit = tk.Button(btn_frame, text="Exit", command=self.on_exit, bg="#f44336", fg="white", font=("Arial", 11, "bold"))
        self.btn_exit.pack(side="left", expand=True, fill="x", padx=5)

        self.status_label = tk.Label(window, text="Saved 0 samples", font=("Arial", 10, "italic"))
        self.status_label.pack(pady=5)

        self.window.bind("<space>", lambda event: self.save_sample())
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            self.current_features = None
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
                    self.current_features = features

            # Convert to Tkinter Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_video)

    def save_sample(self):
        label_text = self.entry_label.get().strip().upper()
        if not label_text:
            messagebox.showwarning("Warning", "Please enter a sign label first!")
            return

        if self.current_features is not None:
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label_text] + self.current_features)
            self.saved_count += 1
            self.status_label.config(text=f"Saved {self.saved_count} samples for '{label_text}'")
        else:
            messagebox.showinfo("Info", "No hand detected in the frame.")

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
    app = DataCollectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()
