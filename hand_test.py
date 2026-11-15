import os
os.environ["GLOG_minloglevel"] = "2"
import cv2, time #mediapipe uses timestamp
import mediapipe as mp
from mediapipe.tasks import python as mp_python #provides general setup tools, like BaseOptions.
from mediapipe.tasks.python import vision #contains vision model like HandLandmarker

#set up: video mode, 2 hands, BISINDO
options = vision.HandLandmarkerOptions(
    base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task"), #trained neural network
    running_mode = vision.RunningMode.VIDEO, #passing continuous frame, uses video
    num_hands = 2 #default = 1, BISINDO = 2 hands
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1) #flip
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb) #wraps to mediapipe format
    result = landmarker.detect_for_video(mp_image, int(time.monotonic() * 1000)) #clock in seconds, only moves forward
    #*1000 converts to miliseconds

    #drawing keypoints
    h, w = frame.shape[:2] #takes first 2 numbers of the shape (height, width)
    for hand, handed in zip(result.hand_landmarks, result.handedness): #zip -> walks through both lists together
        for lm in hand:
            cv2.circle(frame, (int(lm.x*w), int (lm.y*h)), 4, (0, 255, 0), -1)
        wrist = hand[0] #keypoint 0
        name = handed[0].category_name
        name = "Left" if name == "Right" else "Right" #swap names
        cv2.putText(frame, name,
                    (int(wrist.x*w), int(wrist.y*h)+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (20, 40), #how many hands r found
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("hands", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
    if cv2.getWindowProperty("hands", cv2.WND_PROP_VISIBLE) < 1: break
cap.release(); cv2.destroyAllWindows(); landmarker.close()