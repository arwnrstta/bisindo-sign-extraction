import os
os.environ["GLOG_minloglevel"] = "2"
import glob, cv2, numpy as np, joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import hands_to_features

DATASET = "bisindo_dataset"

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,      # single photos, not video
    num_hands=2)
landmarker = vision.HandLandmarker.create_from_options(options)

X, y, skipped = [], [], 0
for letter in sorted(os.listdir(DATASET)):
    folder = os.path.join(DATASET, letter)
    if not os.path.isdir(folder): continue
    for path in glob.glob(os.path.join(folder, "*")):
        img = cv2.imread(path)
        if img is None: continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            skipped += 1; continue
        X.append(hands_to_features(result.hand_landmarks)); y.append(letter)
    print(f"{letter} done")
print(f"Used {len(X)} images, skipped {skipped} (no hand found)")

X, y = np.array(X), np.array(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=0)
clf.fit(X_tr, y_tr)
print(f"Test accuracy: {accuracy_score(y_te, clf.predict(X_te)):.1%}")
joblib.dump(clf, "bisindo_mlp.joblib")
print("Saved model to bisindo_mlp.joblib")