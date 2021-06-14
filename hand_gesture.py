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
        #Change to RGB
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handsLms in self.results.multi_hand_landmarks:

                #21 points of hand
                _0x,_1x,_2x,_3x,_4x,_5x,_6x,_7x,_8x,_9x,_10x,_11x,_12x,_13x,_14x,_15x,_16x,_17x,_18x,_19x,_20x = [handsLms.landmark[i].x for i in range(21)]
                _0y, _1y, _2y, _3y, _4y, _5y, _6y, _7y, _8y, _9y, _10y, _11y, _12y, _13y, _14y, _15y, _16y, _17y, _18y, _19y, _20y = [handsLms.landmark[i].y for i in range(21)]

                #To make easier to detect where (optional)
                palmx,palmy = handsLms.landmark[0].x, handsLms.landmark[0].y
                thumbx,thumby = handsLms.landmark[4].x, handsLms.landmark[4].y
                indexx,indexy = handsLms.landmark[8].x, handsLms.landmark[8].y
                middlex, middley = handsLms.landmark[12].x, handsLms.landmark[12].y
                ringx,ringy = handsLms.landmark[16].x, handsLms.landmark[16].y
                utx,uty = handsLms.landmark[20].x, handsLms.landmark[20].y

                #Like and dislike
                if _4y < palmy and _8x > _6x and _12x > _10x and _16x > _14x and _20x > _18x and _4x > _6x and _4y > _5y:
                    cv.putText(img, 'like', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if _4y < palmy and _8x < _6x and _12x < _10x and _16x < _14x and _20x < _18x and _4x < _6x and _4y > _5y:
                    cv.putText(img, 'like', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if _4y > palmy and _8x > _6x and _12x > _10x and _16x > _14x and _20x > _18x and _4x > _6x and _4y < _5y:
                    cv.putText(img, 'dislike', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if _4y > palmy and _8x < _6x and _12x < _10x and _16x < _14x and _20x < _18x and _4x < _6x and _4y < _5y:
                    cv.putText(img, 'dislike', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

                #Swear
                if _4y < palmy and _12x > _11x and _8x < _11x and _16x <_11x and _20x < _11x:
                    cv.putText(img, 'Fuck', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if _4y < palmy and _12x < _11x and _8x > _11x and _16x >_11x and _20x > _11x:
                    cv.putText(img, 'Fuck', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if palmy < middley and _12y < _11y and _8y > _11y and _16y >_11y and _20y > _11y:
                    cv.putText(img, 'Fuck', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)
                if palmy > middley and _12y > _11y and _8y < _11y and _16y < _11y and _20y < _11y:
                    cv.putText(img, 'Fuck', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

                #Okay
                if abs(thumby - indexy) < 0.02:
                    cv.putText(img, 'OKAY', (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)






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

        imshow('Hand Gesture', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()
