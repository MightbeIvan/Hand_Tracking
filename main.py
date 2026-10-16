
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gesture_recognizer.task"


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Download the pretrained model into the models folder.")
        return

    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Could not open your webcam.")
        print("Check camera permissions or try camera index 1.")
        return

    print("Hand Tracking is running!")
    print("Show your hand to the camera.")
    print("Press Q to quit.")

    try:
        with vision.GestureRecognizer.create_from_options(
            options
        ) as recognizer:

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a webcam frame.")
                    break

                # Mirror the image like a selfie camera.
                frame = cv2.flip(frame, 1)

                # OpenCV uses BGR; MediaPipe expects RGB.
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000

                result = recognizer.recognize_for_video(
                    mp_image, timestamp_ms
                )

                if result.gestures:
                    for index, gestures in enumerate(result.gestures):
                        if not gestures:
                            continue

                        gesture = gestures[0]
                        label = gesture.category_name
                        confidence = gesture.score

                        cv2.putText(
                            frame,
                            f"{label} ({confidence:.0%})",
                            (20, 45 + index * 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (80, 255, 120),
                            2,
                        )
                else:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow("Hand Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
