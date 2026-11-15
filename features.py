import numpy as np

def hands_to_feature(hand_landmarks):
    """detected hands = 126 numbers (2 hands x 21 points x 3 coords)"""
    hands = [np.array([[lm.x, lm.y, lm.z] for lm in hand]) for hand in hand_landmarks]
    hands.sort(key)