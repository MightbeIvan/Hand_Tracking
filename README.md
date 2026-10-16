
# Hand and Eye Tracking

A Python computer vision project that uses a webcam to track hand gestures and estimate eye gaze. The project uses MediaPipe for landmark detection and OpenCV to process and display live webcam footage.

I'm building this project to learn more about computer vision and how real-time tracking systems work. It currently includes hand gesture recognition and an experimental eye-tracking system with a basic calibration process.

## Features

### Hand Tracking
- Detects hand gestures using a pretrained MediaPipe model.
- Supports tracking up to two hands.
- Displays recognized gestures and their confidence scores.
- Uses a live webcam feed to process frames in real time.

### Eye Tracking
- Detects facial landmarks around the eyes and irises.
- Displays eye and iris tracking points.
- Estimates gaze position using iris measurements.
- Includes a five-point calibration process.
- Displays an estimated gaze point in a separate tracking window.
- Allows recalibration while the program is running.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## Project Structure

```text
Hand_Tracking/
├── main.py
├── hand_tracking.py
├── eye_tracking.py
├── requirements.txt
├── models/
│   ├── gesture_recognizer.task
│   └── face_landmarker.task
├── .gitignore
└── README.md
```

The `.venv` directory contains the local Python virtual environment. It should not be committed to the repository because virtual environments are specific to a machine and operating system.

## Requirements

Before running the project, make sure you have:

- Python installed.
- A working webcam.
- Git installed if you want to clone the repository.
- The required MediaPipe model files downloaded into the `models` directory.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MightbeIvan/Hand_Tracking.git
cd Hand_Tracking
```

### 2. Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows using PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
mediapipe
opencv-python
numpy
```

## Download the Models

The project uses pretrained MediaPipe models. Download both files and place them inside the `models` directory.

### Hand Gesture Recognition Model

Download:

https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

Save the file as:

```text
models/gesture_recognizer.task
```

### Face Landmark Model

Download:

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Save the file as:

```text
models/face_landmarker.task
```

Make sure the filenames match exactly. The programs use these paths to load the models.

## Running the Project

Start the application from the project directory:

```bash
python main.py
```

The main menu provides three options:

1. Hand Tracking
2. Eye Tracking
3. Exit

Select an option to launch the corresponding program.

### Using Hand Tracking

Choose Hand Tracking from the menu. The webcam window displays recognized hand gestures and their confidence scores.

Press `Q` to close the webcam window and return to the main menu.

### Using Eye Tracking

Choose Eye Tracking from the menu. The program opens a calibration process before starting live gaze estimation.

1. Look at the displayed calibration target.
2. Press `Space` when you are ready to capture the target.
3. Repeat for all five targets.
4. Observe the estimated gaze point in the tracking window.

During live tracking:

- Press `C` to recalibrate.
- Press `Q` to exit eye tracking.

For more consistent results, use good lighting, position the webcam near eye level, and try to keep your head relatively steady during calibration.

## How It Works

### Hand Tracking

The hand-tracking program captures webcam frames using OpenCV and passes them to MediaPipe's Gesture Recognizer. The model identifies supported gestures and returns predictions with confidence scores. OpenCV then displays the results on the video feed.

### Eye Tracking

The eye-tracking program uses MediaPipe Face Landmarker to detect facial landmarks, including points around the eyes and irises.

It calculates normalized iris positions relative to the eye regions. During calibration, it records these measurements while the user looks at five targets. NumPy is used to fit a simple linear mapping between the recorded measurements and the target screen coordinates.

The resulting mapping estimates a gaze position during live tracking.

## Current Limitations

This project is still under development.

- Gaze estimation is experimental and may not accurately identify where someone is looking.
- Calibration results can vary with lighting, camera position, head movement, and individual eye characteristics.
- The current calibration uses a simple linear model and only five targets.
- Eye tracking estimates gaze position but does not currently control the mouse cursor.
- Gesture recognition is limited to the gestures supported by the pretrained model.
- Performance depends on the computer, webcam, and installed package versions.

## Future Improvements

Some improvements I would like to explore include:

- Improving gaze estimation and calibration accuracy.
- Adding more calibration points.
- Reducing jitter in the estimated gaze position.
- Adding a visual interface for selecting tracking modes.
- Exploring eye-controlled cursor movement.
- Adding more information about detected gestures and tracking performance.

## Troubleshooting

**The program cannot find a model file**

Check that the model is inside the `models` directory and that its filename matches the path expected by the program.

**Python cannot import `cv2` or `mediapipe`**

Activate the project's virtual environment and install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

**The webcam does not open**

Check your operating system's camera permissions, close other applications using the webcam, and verify that the correct camera is selected.

**Eye tracking is inaccurate**

Improve the lighting, face the webcam, keep the camera near eye level, and repeat the calibration process.

## License

This project is intended for learning and experimentation. If you plan to distribute it, consider adding a license that explains how others may use, modify, and share the code.

---

Built with Python while learning about computer vision, gesture recognition, and eye tracking.
