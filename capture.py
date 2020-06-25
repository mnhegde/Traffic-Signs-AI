import cv2, time, os
from PIL import Image
import numpy as np
from sound import playString
from keras.models import load_model

def nothing(x):
    # any operation
    pass

model = load_model('traffic_classifier.h5')
previousSign = None
classes = { 1:'Speed limit of 20km/h',
            2:'Speed limit of 30km/h',
            3:'Speed limit of 50km/h',
            4:'Speed limit of 60km/h',
            5:'Speed limit of 70km/h',
            6:'Speed limit of 70km/h',
            7:'End of 80km/h speed limit',
            8:'Speed limit of 100km/h',
            9:'Speed limit of 120km/h',
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

sounds = {
    1:'Please ensure you are keeping under 20km/h',
            2:'Please ensure you are keeping under 30km/h',
            3:'Please ensure you are keeping under 50km/h',
            4:'Please ensure you are keeping under 60km/h',
            5:'Please ensure you are keeping under 70km/h',
            6:'Please ensure you are keeping under 80km/h',
            7:'Please acknowledge the changed speed limit',
            8:'Please ensure you are keeping under 100km/h',
            9:'Please ensure you are keeping under 120km/h',
            10:'Please ensure to not pass or overtake any vehicles except bicycles, motorcycles, and mopeds',
            11:'If your vehicle is 3.5 tons or over, you aren\'t permitted to overtake any vehicle',
            12:'Please yield to thsoe who arrived at the intersection beforehand',
            13:'Please notice the priority road and its occupants, and allow them to pass freely',
            14:'While merging, please give the right of way to a driver from another approach point',
            15:'Please come to a complete stop',
            16:'No vehicles are allowed in this area',
            17:'If your vehicle is above 3.5 tons, youa re not permitted in this area',
            18:'Please determine another route to your destination',
            19:'Please slow down if needed and pay attention to your surroundings ahead',
            20:'Please slow down when approaching this curve, and watch for any vehicles from the opposing direction',
            21:'Please slow down when approaching this curve, and watch for any vehicles from the opposing direction',
            22:'Please slow down when approaching this curve, and watch for any vehicles from the opposing direction',
            23:'Please slow down, and ensure to be aware of approaching vehicles',
            24:'Please slow down, and ensure to be aware of approaching vehicles',
            25:'Please keep this in mind',
            26:'Be wary of construction ahead, and keep under 30km/h',
            27:'Please slow down to follow these signals accordingly',
            28:'Look out for anyone ahead of you to avoid accidents',
            29:'Please reduce your speed and look for any children to yield to',
            30:'Please reduce your speed and allow bicycles to cross',
            31:'Slow down, and pay attention to oncoming traffic and/or vehicles',
            32:'Slow down, and watch for crossing animals',
            33:'Please slow down and pay attention to other vehicles in the intersection',
            34:'Slow down and take the respective turn',
            35:'Slow down and take the respective turn',
            36:'Please continue straight ahead',
            37:'Please go the respective direction for yoour destination',
            38:'Please go the respective direction for yoour destination',
            39:'Please merge into and keep in the right lane until designated otherwise',
            40:'Please merge into and keep in the left lane until designated otherwise',
            41:'Please use the roundabout, and find another route to your destination if needed',
            42:'This indicates you can now overtake vehicles on the road',
            43:'This indicates vehicles heavier than 3.5 tones can overtake vehicles'
}


def classify(file_path): # The AI function
    global previousSign
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    if sign != previousSign:
        previousSign = sign
        playString('I detected a ' + sign + ' sign. ' + sounds[pred+1])

def findRed(frame):
    # Finding Red/Orange/Light_Yellow in the picture
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    finalRedMask = mask1 + mask2

    # Gets rid of some static in the picture
    kernel = np.ones((5, 5), np.uint8)
    finalRedMask = cv2.erode(finalRedMask, kernel)

    cv2.imshow("maskRed", finalRedMask)
    # Contours detection
    contours, _ = cv2.findContours(finalRedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    signPresent = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        # if area of detected shape is too small(since it is most likelly static)
        if area > 2500:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

            # if the amount of sides is in between 3(triangle) and 20(most likelly a circle)
            if 3 < len(approx) and len(approx) < 20:
                signPresent = True

    return signPresent

def findBlue(frame):
    # Finding Blue in the picture
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110,50,25], np.uint8)
    upper_blue = np.array([140,255,255], np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Gets rid of some static in the picture
    kernel = np.ones((5, 5), np.uint8)
    finalRedMask = cv2.erode(mask, kernel)

    cv2.imshow("maskBlue", mask)
    # Contours detection
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    signPresent = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        # if area of detected shape is too small(since it is most likelly static)
        if area > 2500:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

            # if the amount of sides is in between 3(triangle) and 20(most likelly a circle)
            if 3 < len(approx) and len(approx) < 20:
                signPresent = True

    return signPresent

def screenshot():
    global video
    path = os.path.join('inputImg', 'screenshot.png')
    pic = video.read()[1] # Reads the video from main.
    cv2.imwrite(path, pic) # Puts the picture into the file_path
    image = cv2.imread("inputImg/screenshot.png")

    #image = cv2.imread("Road_Signs.png")
    #width = int(image.shape[1] * 30 / 100)
    #height = int(image.shape[0] * 30 / 100)
    #dsize = (width, height)
    #image = cv2.resize(image, dsize)
    #cv2.imshow("org", image)

    if findBlue(image) or findRed(image): # If we find Red / Blue in our screenshot
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

        # If the letter Q is pressed, the video stops recording.
        if key == ord('q'):
            break

        screenshot()
        a += 1

    print(a) # Num screenshots the program made

    # Shut down camera
    video.release()

    cv2.destroyAllWindows()
