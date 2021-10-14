from keras_preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from lenet import LeNet
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from resize_correctly import preprocess
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

def step_decay(epoch):

    # initialize the base initial learning rate, drop factor, and epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 25

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)

callbacks = [LearningRateScheduler(step_decay)]

# khởi tạo data và label
data = []
labels = []

for imagePath in paths.list_images(args["dataset"]):

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 64, 64)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# convert data and labels to array
data = np.array(data, dtype=float) / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
classWeight = {0: 1.,
               1: classWeight[1]}

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.01, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")

model = LeNet.build(width=64, height=64, depth=1, classes=2)
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
epoch= 100
BS = 32
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY),
              class_weight=classWeight ,steps_per_epoch=len(trainX) // BS, epochs=epoch,
              callbacks=callbacks, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serializing network...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epoch), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
