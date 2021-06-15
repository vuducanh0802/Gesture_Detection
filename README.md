Like_Dislike_Detection

A small project to detect simple gestures which are thumb up and thumb down (like and dislike)

Tool: Opencv, mediapipe

Content: like_dislike_detection.py and hand_gesture.py detect gestures (~95% similar to each other)
         train_isPalmar.py trains a CNN model to detect palmar or dorsal, dataset: https://www.kaggle.com/shyambhu/hands-and-palm-images-dataset
         isPalmar_detection.py is to check palmar (a medium between hand_gesture.py and train_isPalmar.py)

Edit: Added swear and okay gestures to another python file

Edit: add palmar detection to improve accuracy, but just for video, not realtime
