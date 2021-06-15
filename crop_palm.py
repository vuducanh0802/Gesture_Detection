import numpy as np


def crop_hand(img,palmx,palmy,thumbx,thumby,middlex, middley,utx,uty):
    """
    Idea: for 4 points palm, thumb, middle and pinky, we crop the image such that the output is the hand only and those
    4 points are on the edges of the image


    :param palmx: x coordinate of palm point, received from mediapipe in Hand_Gesture.py
    :param palmy:
    :param thumbx:
    :param thumby: y coordinate of thumb point, received from mediapipe in Hand_Gesture.py
    :param middlex:
    :param middley:
    :param utx: x coordinate of pinky point, received from mediapipe in Hand_Gesture.py
    :param uty:  (the reason I called "ut" is that in Vietnamese "pinky" finger called "ut" finger, making it faster to type)
    :return:
    """
    #checking angle
    angle = np.arctan(round(abs(middley - palmy)/abs(middlex - palmx +1e-10))) / np.pi

    if angle > 0.4:
        x = thumbx - thumbx / 100
        x_inc = utx + utx / 100
        y = palmy - palmy / 100
        y_inc = middley + middley / 100
    elif angle < 0.1:
        x = palmx - palmx / 100
        x_inc = middlex + middlex / 100
        y = uty - uty / 100
        y_inc = thumby + thumby / 100
    else:
        x = palmx - palmx / 100
        x_inc = middlex + middlex / 100
        y = palmy - palmy/100
        y_inc = middley + middley / 100

    return img[x:x_inc,y:y_inc]


