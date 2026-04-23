# Driver Drowsiness & Distraction Detection System

A real-time computer vision system designed to enhance road safety by monitoring driver behavior. This application detects signs of fatigue (drowsiness, yawning) and distraction (looking away from the road) using facial landmarks and triggers an audible alarm to alert the driver.

---

## 🌟 Features

- **Drowsiness Detection**: Monitors the Eye Aspect Ratio (EAR) to detect when the driver's eyes stay closed for an extended period.
- **Yawning Detection**: Monitors the Mouth Aspect Ratio (MAR) to identify repetitive or prolonged yawning.
- **Distraction Detection**: Tracks the horizontal orientation of the head to detect if the driver is looking away from the road for too long.
- **Real-time Alert System**: Triggers a loud audio alarm (`.wav` file) when any anomaly is detected.
- **Visual Feedback**: Provides a live video feed with status overlays (NORMAL, DROWSY, YAWNING, DISTRACTED) and calculated metrics (EAR, MAR, Distraction Offset).

---

## 🛠️ Technology Stack

- **Language**: Python
- **Libraries**:
  - **OpenCV**: Image processing and video capture.
  - **MediaPipe**: High-fidelity facial landmark detection (Face Mesh).
  - **Pygame**: Audio management for alarms.
  - **NumPy**: Mathematical computations for aspect ratios.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher.
- A functional webcam.

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/driver-drowsiness-detection.git
   cd driver-drowsiness-detection
   ```

2. **Set up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Simply execute the main script:
```bash
python driver_detector.py
```

---

## 📊 How It Works

### 1. Eye Aspect Ratio (EAR)
Calculates the ratio between the vertical and horizontal distances of the eye. If the ratio drops below a certain threshold (default: 0.30) for a specific number of frames, it indicates the driver's eyes are closing.

### 2. Mouth Aspect Ratio (MAR)
Calculates the vertical opening of the mouth relative to its width. If this ratio exceeds the threshold (default: 0.60), a yawning event is detected.

### 3. Head Pose Offset
Calculates the horizontal distance between the nose tip and the center of the face (based on the distance between outer eye corners). If the offset exceeds the threshold for more than 2 seconds, the driver is marked as "DISTRACTED".

---

## ⚙️ Configuration

You can customize the sensitivity of the detector in the `__init__` method of the `DriverDrowsinessDetector` class in `driver_detector.py`:

```python
self.ear_threshold = 0.30       # Sensitivity for eye closure
self.ear_consec_frames = 10     # Consecutive frames for drowsiness
self.mar_threshold = 0.60       # Sensitivity for yawning
self.distraction_threshold = 0.05 # Head pose sensitivity
self.distraction_timeout = 2.0  # Seconds before distraction alert
```

---

## 🎮 Controls

- **Q**: Press 'q' on the keyboard to exit the application and release resources.

---

## 📂 Project Structure

```text
├── driver_detector.py        # Main application script
├── mixkit-classic-alarm-995.wav # Alarm audio file
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It should not be used as a primary safety system in a real-world driving environment. Always stay alert and focused while driving.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
