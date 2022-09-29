import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller

def ind_tilt(pose, l, r):
    if pose < l:
        return 'L'
    elif pose > r:
        return 'R'
    else:
        return 'C'


class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.basePosition=True
        
        self.mouse = Controller()

    def start(self):
        """run a loop where camera detects hand motions
        """
        with self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()
                image = cv2.flip(image, 1)

                img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(img_RGB)

                self.hands_action(image, results)

                cv2.imshow("Hand Tracking", image)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

            self.cap.realease()
            cv2.destroyAllWindows()

    def hands_action(self, image, results):
        """process image and mediapipe hand landmarks

        Args:
            image (_type_): camera image
            results (_type_): media pipe hand landmarks
        """
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        # draw identified landmarks on camera
                        self.mp_draw.draw_landmarks(
                            image,
                            hand_landmark,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        # identify tilt and scroll mouse accordinly
                        # only identifies movments starting from base position ("C")
                        tilt = self.identify_tilt(hand_landmark)
                        print('tilt: ', tilt)
                        if tilt!='C' and self.basePosition:
                            if tilt=='R':
                                self.mouse.scroll(1,0)
                            elif tilt=='L':
                                self.mouse.scroll(-1,0)
                            self.basePosition=False
                        elif tilt=='C' and not self.basePosition:
                            self.basePosition=True
                    
    

    def identify_tilt(self, landmark):
        """identify specific tilt

        Args:
            landmark (_type_): hand landmarks

        Returns:
            _type_: "R"/"C"/"L" for speciific tilt
        """
        idx_tip = landmark.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        idx_base = landmark.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x
        idx_pose = idx_tip-idx_base

        return ind_tilt(idx_pose, 0, 0.16)



if __name__=="__main__":
    HandTracker().start()