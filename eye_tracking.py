
import time
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

WINDOW_NAME = "Calibrated Eye Tracking"
CAMERA_NAME = "Webcam Preview"

WIDTH = 1280
HEIGHT = 720

CAPTURE_SECONDS = 1.5
MIN_SAMPLES = 10

# Five calibration targets, expressed as screen proportions.
TARGETS = [
    (0.10, 0.10),
    (0.90, 0.10),
    (0.50, 0.50),
    (0.10, 0.90),
    (0.90, 0.90),
]

# Eye landmark indices.
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def average_point(landmarks, indices):
    return (
        sum(landmarks[i].x for i in indices) / len(indices),
        sum(landmarks[i].y for i in indices) / len(indices),
    )


def normalized_eye_position(landmarks, corners, upper, lower, iris):
    """Measure iris position relative to eye corners and eyelids."""
    left_x = landmarks[corners[0]].x
    right_x = landmarks[corners[1]].x

    top_y = landmarks[upper].y
    bottom_y = landmarks[lower].y

    iris_x, iris_y = average_point(landmarks, iris)

    eye_width = abs(right_x - left_x)
    eye_height = abs(bottom_y - top_y)

    if eye_width < 0.001 or eye_height < 0.001:
        return None

    horizontal = (iris_x - min(left_x, right_x)) / eye_width
    vertical = (iris_y - min(top_y, bottom_y)) / eye_height

    return horizontal, vertical


def extract_features(landmarks):
    """Combine both eyes into two normalized gaze features."""
    left = normalized_eye_position(
        landmarks, (33, 133), 159, 145, LEFT_IRIS
    )

    right = normalized_eye_position(
        landmarks, (362, 263), 386, 374, RIGHT_IRIS
    )

    if left is None or right is None:
        return None

    return np.array([
        (left[0] + right[0]) / 2,
        (left[1] + right[1]) / 2,
    ], dtype=float)


def create_canvas(target=None, message=""):
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    canvas[:] = (25, 27, 32)

    cv2.putText(
        canvas, "CALIBRATED EYE TRACKING",
        (35, 45), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (240, 240, 240), 2
    )

    if message:
        cv2.putText(
            canvas, message,
            (35, HEIGHT - 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (210, 210, 210), 1
        )

    if target is not None:
        x = int(target[0] * WIDTH)
        y = int(target[1] * HEIGHT)

        cv2.circle(canvas, (x, y), 23, (255, 255, 255), 2)
        cv2.circle(canvas, (x, y), 12, (0, 180, 255), -1)
        cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1)

    return canvas


def show_camera(frame, landmarks=None):
    preview = frame.copy()

    if landmarks is not None:
        height, width = preview.shape[:2]

        # Highlight the iris points.
        for index in LEFT_IRIS + RIGHT_IRIS:
            point = landmarks[index]
            x = int(point.x * width)
            y = int(point.y * height)
            cv2.circle(preview, (x, y), 3, (0, 165, 255), -1)

    cv2.imshow(CAMERA_NAME, preview)


def read_features(camera, landmarker, timestamp):
    success, frame = camera.read()

    if not success:
        return None, None, None, timestamp

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    timestamp = max(timestamp + 1, int(time.monotonic() * 1000))
    result = landmarker.detect_for_video(image, timestamp)

    if not result.face_landmarks:
        show_camera(frame)
        return None, frame, None, timestamp

    landmarks = result.face_landmarks[0]
    features = extract_features(landmarks)

    show_camera(frame, landmarks)
    return features, frame, landmarks, timestamp


def wait_for_space(camera, landmarker, canvas, timestamp):
    """Keep refreshing the webcam until the user presses Space."""
    while True:
        features, frame, landmarks, timestamp = read_features(
            camera, landmarker, timestamp
        )

        message = (
            "Face detected - press SPACE to capture"
            if features is not None
            else "Look toward the webcam, then press SPACE"
        )

        display = canvas.copy()
        cv2.putText(
            display, message,
            (35, 95), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (230, 230, 230), 2
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False, timestamp

        if key == ord(" ") and features is not None:
            return True, timestamp


def capture_target(camera, landmarker, target, index, timestamp):
    """Collect multiple measurements while the target stays visible."""
    canvas = create_canvas(
        target,
        f"Calibration target {index}/5 - look at the dot"
    )

    print(f"Target {index}/5: look at the dot and press SPACE.")

    ready, timestamp = wait_for_space(
        camera, landmarker, canvas, timestamp
    )

    if not ready:
        return None, timestamp

    samples = []
    start = time.monotonic()

    while time.monotonic() - start < CAPTURE_SECONDS:
        features, frame, landmarks, timestamp = read_features(
            camera, landmarker, timestamp
        )

        if features is not None:
            samples.append(features)

        display = canvas.copy()
        elapsed = time.monotonic() - start
        remaining = max(0, CAPTURE_SECONDS - elapsed)

        cv2.putText(
            display, f"Capturing... {remaining:.1f}s",
            (35, 95), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 220, 120), 2
        )

        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None, timestamp

    if len(samples) < MIN_SAMPLES:
        print("Not enough face samples. Please repeat this target.")
        return "retry", timestamp

    average_features = np.mean(samples, axis=0)
    print(f"Captured {len(samples)} samples.")
    return average_features, timestamp


def calibrate(camera, landmarker, timestamp):
    feature_samples = []
    screen_targets = []

    index = 0

    while index < len(TARGETS):
        target = TARGETS[index]

        features, timestamp = capture_target(
            camera, landmarker, target, index + 1, timestamp
        )

        if features is None:
            return None, timestamp

        if isinstance(features, str):
            continue

        feature_samples.append(features)
        screen_targets.append(target)
        index += 1

    # Fit a simple linear mapping:
    # [1, iris_horizontal, iris_vertical] -> [screen_x, screen_y]
    X = np.array(feature_samples, dtype=float)
    Y = np.array(screen_targets, dtype=float)

    design = np.column_stack([
        np.ones(len(X)),
        X[:, 0],
        X[:, 1],
    ])

    coefficients, _, _, _ = np.linalg.lstsq(
        design, Y, rcond=None
    )

    print("Calibration complete!")
    return coefficients, timestamp


def predict_gaze(features, coefficients):
    row = np.array([1.0, features[0], features[1]])
    prediction = row @ coefficients

    x = float(np.clip(prediction[0], 0.0, 1.0))
    y = float(np.clip(prediction[1], 0.0, 1.0))

    return x, y


def run_live_tracking(camera, landmarker, coefficients, timestamp):
    history = deque(maxlen=5)

    print("Live gaze tracking started.")
    print("Press C to recalibrate or Q to quit.")

    while True:
        features, frame, landmarks, timestamp = read_features(
            camera, landmarker, timestamp
        )

        if features is None:
            display = create_canvas(
                message="Face not detected - Q to quit"
            )
        else:
            gaze_x, gaze_y = predict_gaze(features, coefficients)
            history.append((gaze_x, gaze_y))

            smooth_x = sum(p[0] for p in history) / len(history)
            smooth_y = sum(p[1] for p in history) / len(history)

            display = create_canvas()

            px = int(smooth_x * WIDTH)
            py = int(smooth_y * HEIGHT)

            cv2.circle(display, (px, py), 18, (255, 255, 255), 2)
            cv2.circle(display, (px, py), 8, (0, 180, 255), -1)

            cv2.putText(
                display,
                f"Estimated gaze: X={smooth_x:.2f}, Y={smooth_y:.2f}",
                (35, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 220, 120), 2
            )

        cv2.putText(
            display, "C = Recalibrate | Q = Quit",
            (35, HEIGHT - 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (210, 210, 210), 1
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return "quit", timestamp

        if key == ord("c"):
            return "calibrate", timestamp


def main():
    if not MODEL_PATH.exists():
        print(f"Face model not found: {MODEL_PATH}")
        print("Check that models/face_landmarker.task exists.")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open webcam.")
        camera.release()
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    cv2.namedWindow(CAMERA_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_NAME, 480, 360)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    timestamp = 0

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                print("\n=== EYE TRACKING CALIBRATION ===")
                print("You will calibrate five screen targets.")
                print("Look at each dot and press SPACE.")
                print("Keep your head as still as comfortable.")
                print("Press Q during calibration to exit.")

                coefficients, timestamp = calibrate(
                    camera, landmarker, timestamp
                )

                if coefficients is None:
                    print("Calibration cancelled.")
                    break

                action, timestamp = run_live_tracking(
                    camera, landmarker, coefficients, timestamp
                )

                if action == "quit":
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Eye tracking closed.")


if __name__ == "__main__":
    main()
