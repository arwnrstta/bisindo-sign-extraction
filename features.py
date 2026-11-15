import numpy as np

def hands_to_features(hand_landmarks):
    """detected hands = 126 numbers (2 hands x 21 points x 3 coords)"""
    hands = [np.array([[lm.x, lm.y, lm.z] for lm in hand]) for hand in hand_landmarks]
    hands.sort(key=lambda p: p[0,0])
    feats = []
    for i in range(2):
        if i < len (hands):
            pts  = hands[i] - hands[i][0]
            scale = np.max(np.linalg.norm(pts[:, :2], axis = 1)) or 1.0
            feats.append((pts/scale).flatten())
        else:
            feats.append(np.zeros(63))
    return np.concatenate(feats)