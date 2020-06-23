import cv2, time, os

def screenshot():
    global video
    path = "C:/Users/matve/Documents/BitHacksProject/inputImg"
    #cv2.imshow("screenshot", video.read()[1])
    cv2.imwrite(os.path.join(path , 'screenshot.png'), video.read()[1])

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
