import os
os.environ["GLOG_minloglevel"] = "2"
import cv2, time, joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from features import hands_to_features

clf = joblib.load("bisindo_mlp.joblib")
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    # NO flip before detection: dataset photos are unmirrored, so the webcam must match
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.monotonic() * 1000))

    h, w = frame.shape[:2]
    for hand in result.hand_landmarks:
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

    text = "No hands"
    if result.hand_landmarks:
        probs = clf.predict_proba([hands_to_features(result.hand_landmarks)])[0]
        best = probs.argmax()
        letter, conf = clf.classes_[best], probs[best]
        text = f"{letter}  {conf:.0%}" if conf > 0.7 else "?"

    display = cv2.flip(frame, 1)                 # mirror ONLY for display
    cv2.putText(display, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("BISINDO", display)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
    if cv2.getWindowProperty("BISINDO", cv2.WND_PROP_VISIBLE) < 1: break

cap.release(); cv2.destroyAllWindows(); landmarker.close()