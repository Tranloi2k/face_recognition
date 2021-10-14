
import argparse
import cv2
from keras.models import load_model
import numpy as np
import imutils
from imutils import face_utils
from keras.preprocessing.image import img_to_array
import dlib
from resize_correctly import preprocess

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")

args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
# initialize the class labels

image = cv2.imread(args["image"])
image1 = image.copy()
image1 = imutils.resize(image1, width=600)
(a, b) = image.shape[:2]
if a<400 :
    image = imutils.resize(image, width= 400)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

(H, W) = image.shape[:2]
(H1, W1) = image1.shape[:2]
t = W1/W
q = H1/H



# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
# loop over the face detections
for (i, rect) in enumerate(rects):
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)

    roi = gray[y - 20:y + h + 20, x - 20:x + w + 20]
    roi = preprocess(roi, 64, 64)
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0) / 255.0
    (x, y, w, h) = (x*t, y*q, w*t, h*q)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    (dan, loi) = model.predict(roi)[0]

    if loi < 0.8 and dan < 0.8 :
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image1, "Khong phai Loi hoac Dan ", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("", image1)
        cv2.waitKey(0)


    else:

        if loi > dan:
            label = "Loi"
            score = loi*100
        else:
            label = "Dan"
            score = dan*100

        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image1, "{} with {}%".format(label, round(score,2)), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("",image1)
        cv2.waitKey(0)

