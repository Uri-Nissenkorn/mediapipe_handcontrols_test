import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.8,min_tracking_confidence=0.8):
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image,1)

        img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_RGB)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
                



        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
cap.realease()
cv2.destroyAllWindows()