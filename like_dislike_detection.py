import cv2 as cv
from cv2 import imshow
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self,img,draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handsLms,self.mpHand.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0,draw = True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                # print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,255),5,cv.FILLED)
        return lmlist
    def gesture(self,img):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:
                six, eight, ten, twelve, fourteen, sixteen, eighteen, twenty = [handsLms.landmark[i].x for i in [6, 8, 10, 12, 14, 16, 18, 20]]
                thumbx = handsLms.landmark[4].x
                thumby = handsLms.landmark[4].y
                palm = handsLms.landmark[0].y
                if thumby < palm and eight > six and twelve > ten and sixteen > fourteen and twenty > eighteen and thumbx > six:
                    cv.putText(img, 'like', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if thumby < palm and eight < six and twelve < ten and sixteen < fourteen and twenty < eighteen and thumbx < six:
                    cv.putText(img, 'like', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if thumby > palm and eight > six and twelve > ten and sixteen > fourteen and twenty > eighteen and thumbx > six:
                    cv.putText(img, 'dislike', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if thumby > palm and eight < six and twelve < ten and sixteen < fourteen and twenty < eighteen and thumbx < six:
                    cv.putText(img, 'dislike', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    cTime = 0

    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHand(img)

        lmlist = detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        detector.gesture(img)
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

        imshow('Frame', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()
