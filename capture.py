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


def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    print(sign)

def findRed(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    finalRedMask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    finalRedMask = cv2.erode(finalRedMask, kernel)
    cv2.imshow("maskRed", finalRedMask)
    # Contours detection
    contours, _ = cv2.findContours(finalRedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    signPresent = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        if area > 1500:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

            if 3 < len(approx) and len(approx) < 20:
                signPresent = True

    return signPresent

def findBlue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100,150,0], np.uint8)
    upper_blue = np.array([140,255,255], np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    finalRedMask = cv2.erode(mask, kernel)
    cv2.imshow("maskBlue", mask)
    # Contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    signPresent = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        if area > 1500:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

            if 3 < len(approx) and len(approx) < 20:
                signPresent = True

    return signPresent

def screenshot():
    global video
    path = os.path.join('inputImg', 'screenshot.png')
    pic = video.read()[1]
    #cv2.imshow("screenshot", video.read()[1])
    cv2.imwrite(path, pic)
    image = cv2.imread("inputImg/screenshot.png")

    #image = cv2.imread("Road_Signs.png")
    #width = int(image.shape[1] * 30 / 100)
    #height = int(image.shape[0] * 30 / 100)
    #dsize = (width, height)
    #image = cv2.resize(image, dsize)
    #cv2.imshow("org", image)

    if findBlue(image) or findRed(image):
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
        #frame = cv2.imread("Road_Signs.png")
        #width = int(frame.shape[1] * 30 / 100)
        #height = int(frame.shape[0] * 30 / 100)
        #dsize = (width, height)
        #frame = cv2.resize(frame, dsize)


        # Show the frame
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(750)

        if key == ord('q'):
            break

        screenshot()
        a += 1

    print(a)

    # Shut down camera
    video.release()

    cv2.destroyAllWindows()
