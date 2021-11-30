from imutils import paths
from imutils import face_utils
import argparse
import imutils
import cv2
import os
import dlib



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-o", "--output", required=True, help="path to output image")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

detector = dlib.get_frontal_face_detector()

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    try:
        i = i
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained
        # display an update to the user
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        # loop over the contours
        for (k, c) in enumerate(rects):
            # compute the bounding box for the contour then extract
            # the digit
            (x, y, w, h) = face_utils.rect_to_bb(c)
            roi = gray[y - 20:y + h + 20, x - 20:x + w + 20]
            # display the character, making it larger enough for us
            # to see, then wait for a keypress

            cv2.imshow("ROI", imutils.resize(roi, width=50))
            key = cv2.waitKey(0)
            print("nhap ")

            if key == ord("b"):
                print("[INFO] ignoring")
                continue

                # grab the key that was pressed and construct the path
                # the output directory
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["output"], key])

            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # write the labeled character to file

            # counts{} dùng để đếm số lượng của từng key
            count = counts.get(key, 1)  # đếm số lượng của key trong counts, nếu ko có trả về 1

            # Tạo file ảnh với tên theo biến đếm
            p = os.path.sep.join([dirPath, "1{}.jpg".format(str(count).zfill(5))])
            cv2.imwrite(p, roi)

            # increment the count for the current key
            counts[key] = count + 1    # Tăng giá trị của key , nếu chưa có thì tạo.

    # we are trying to control-c out of the script, so break from the
    # loop (you still need to press a key for the active window to
    # trigger this)
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break

    # an unknown error has occurred for this particular image
    except:
        print("[INFO] skipping image...")


