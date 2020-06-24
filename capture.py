import cv2, time, os
from PIL import Image
import numpy as np
from keras.models import load_model

def nothing(x):
    # any operation
    pass

model = load_model('traffic_classifier.h5')
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
            10:'No passing',
            11:'No passing veh over 3.5 tons',
            12:'Right-of-way at intersection',
            13:'Priority road',
            14:'Yield',
            15:'Stop',
            16:'No vehicles',
            17:'Veh > 3.5 tons prohibited',
            18:'No entry',
            19:'General caution',
            20:'Dangerous curve left',
            21:'Dangerous curve right',
            22:'Double curve',
            23:'Bumpy road',
            24:'Slippery road',
            25:'Road narrows on the right',
            26:'Road work',
            27:'Traffic signals',
            28:'Pedestrians',
            29:'Children crossing',
            30:'Bicycles crossing',
            31:'Beware of ice/snow',
            32:'Wild animals crossing',
            33:'End speed + passing limits',
            34:'Turn right ahead',
            35:'Turn left ahead',
            36:'Ahead only',
            37:'Go straight or right',
            38:'Go straight or left',
            39:'Keep right',
            40:'Keep left',
            41:'Roundabout mandatory',
            42:'End of no passing',
            43:'End no passing veh > 3.5 tons' }

cv2.namedWindow("trackbars")
cv2.createTrackbar("L-H", "trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "trackbars", 185, 255, nothing)
cv2.createTrackbar("L-V", "trackbars", 128, 255, nothing)
cv2.createTrackbar("U-H", "trackbars", 36, 180, nothing)
cv2.createTrackbar("U-S", "trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "trackbars", 243, 255, nothing)


def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    print(sign)

def screenshot():
    global video
    path = os.path.join('inputImg', 'screenshot.png')
    pic = video.read()[1]
    #cv2.imshow("screenshot", video.read()[1])
    cv2.imwrite(path, pic)
    image = cv2.imread("inputImg/screenshot.png")

    signPresent = False

    if signPresent:
        classify(path)

if __name__ == '__main__':
    # Creates camera object
    video = cv2.VideoCapture(0)
    video.set(3, 640)
    video.set(4, 480)
    video.set(10, 100)

    a = 0
    while True:
        # Creates a frame object
        check, frame = video.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L-H", "trackbars")
        l_s = cv2.getTrackbarPos("L-S", "trackbars")
        l_v = cv2.getTrackbarPos("L-V", "trackbars")
        u_h = cv2.getTrackbarPos("U-H", "trackbars")
        u_s = cv2.getTrackbarPos("U-S", "trackbars")
        u_v = cv2.getTrackbarPos("U-V", "trackbars")

        lower_red = np.array([0, 120, 70])
        upper_red = np.array([30, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        finalMask = mask1 + mask2
        cv2.imshow("mask", finalMask)
        # Contours detection
        contours, _ = cv2.findContours(finalMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 5)




        # Show the frame
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        screenshot()
        a += 1

    print(a)

    # Shut down camera
    video.release()

    cv2.destroyAllWindows()
