import cv2
import numpy as np
import pickle

###################################################
width = 640
height = 480
threshold = 0.65  # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
###################################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)

#### LOAD THE TRAINNED MODEL
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

#### PREPORCESSING FUNCTION
def preProcessing(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Processsed 1", img)
    # img = (np.float32(img), cv2.COLOR_RGB2GRAY)
    # img = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_RGB2GRAY)

    # img = cv2.equalizeHist(img)
    # cv2.imshow("Processsed 2", img)

    # img = img/255
    # cv2.imshow("Processsed 3", img)

    return img


# while True:
# success, imgOriginal = cap.read()
imgOriginal = cv2.imread('test/oitenta.png')
img = np.asarray(imgOriginal)
img = cv2.resize(img, (32, 32))
img = preProcessing(img)
cv2.imshow("Processsed Image", img)
img = img.reshape(1, 32, 32, 1)
cv2.imshow("Processsed Image", img)

#### PREDICT
classIndex = int(model.predict_classes(img))

#print(classIndex)
predictions = model.predict(img)

#print(predictions)
probVal = np.amax(predictions)
print(classIndex, probVal)

if probVal > threshold:
    cv2.putText(imgOriginal,
                str(classIndex) + "   " + str(probVal),
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                1)

cv2.imshow("Original Image", imgOriginal)
cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
