import cv2
import os
import threading


# PATH TO FOLDER
path = os.path.dirname(os.path.abspath(__file__))
faces_dir = os.path.join(path, "faces")
os.makedirs(faces_dir, exist_ok=True)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cnt = 0
counter = 1
offset = 50

stop = threading.Event()

def face_catch(image):
    global cnt, counter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        for x, y, w, h in faces:
            if len(gray) > 0:
                cv2.imwrite(faces_dir + '/' + str(counter) + '.' + str(cnt) + '.jpg',
                            gray[y - offset: y + h + offset, x - offset: x + w + offset])
                cnt += 1
                counter += 1
                # FACE FRAME SIZE
                cv2.rectangle(image, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
                # SHOWING LAST KNOWN FRAME
                cv2.imshow('image', image[y - offset: y + h + offset, x - offset: x + w + offset])
                # PAUSE
                cv2.waitKey(500)
                if cnt == 10:
                    cnt = 0
                    continue
            else:
                 continue

    if stop.is_set():
        cv2.destroyAllWindows()

# GETTING VIDEO STREAM
# testing on webcam
video = cv2.VideoCapture(0)

def get_frames():
    while True:
        ret, frame = video.read()
        if not ret:
            break

        face_catch(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture = threading.Thread(target=get_frames())
capture.start()