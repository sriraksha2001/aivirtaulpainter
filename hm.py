import mediapipe as mp
import cv2
import time


class handDetector:

    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        landmarks = results.multi_hand_landmarks
        # print(results.multi_hand_landmarks)
        if landmarks:

            for handLms in landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

        # for id, lm in enumerate(handLms.landmark):
        # print(id,lm)
        #    height, width, c = img.shape
        #    cx, cy = int(lm.x*width), int(lm.y*height)
        #    print(id, ", x=",cx, ", y=",cy)
        #    if id%10 == 0:
        #        cv2.circle(img, (cx,cy), 8, (255,0,255), cv2.FILLED)


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.FindHands(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

import main

if __name__ == '__main__':
    main()
