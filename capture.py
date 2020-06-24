import cv2, time, os
from PIL import Image
import numpy
from sound import playString
from keras.models import load_model

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
            36:'Please continue straight ahead, 
            37:'Please go the respective direction for yoour destination', 
            38:'Please go the respective direction for yoour destination', 
            39:'Please merge into and keep in the right lane until designated otherwise', 
            40:'Please merge into and keep in the left lane until designated otherwise', 
            41:'Please use the roundabout, and find another route to your destination if needed', 
            42:'This indicates you can now overtake vehicles on the road', 
            43:'This indicates vehicles heavier than 3.5 tones can overtake vehicles'
}


def classify(file_path):
    global previousSign
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    if sign != previousSign:
        previousSign = sign
        playString('I detected a ' + sign + ' sign. ' + sounds[pred+1])


def screenshot():
    global video
    path = os.path.join('inputImg', 'screenshot.png')
    #cv2.imshow("screenshot", video.read()[1])
    cv2.imwrite(path, video.read()[1])
    classify(path, a)
    
if __name__ == '__main__':
    # Creates camera object
    video = cv2.VideoCapture(0)
    video.set(3, 640)
    video.set(4, 480)
    video.set(10, 100)

    a = 0
    while True:
        a += 1
        # Creates a frame object
        check, frame = video.read()

        # Show the frame
        cv2.imshow("Capturing", frame)

        key = cv2.waitKey(750)

        if key == ord('q'):
            break
        screenshot()

    print(a)

    # Shut down camera
    #video.release()

    cv2.destroyAllWindows()

