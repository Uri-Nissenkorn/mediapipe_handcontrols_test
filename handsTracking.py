import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def ind_tilt(pose, l , r):
    if pose<l:
        return 'L'
    elif pose>r:
        return 'R'
    else:
        return 'C'


with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)

        img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_RGB)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            idx_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            idx_base = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
            idx_pose = idx_tip-idx_base
            
            '''
            index finger limits:
            [LEFT HAND]
            *LEFT - x<0 
            *CENTER - 0<x<0.14 
            *RIGHT - 0.14>x
            '''

            print(f'Index finger tip tilt:',ind_tilt(idx_pose, 0, 0.14))
            
           

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.realease()
cv2.destroyAllWindows()
