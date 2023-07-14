import cv2
import os
import asyncio
import time
from datetime import datetime
from database.db_connection import async_session
from database.models import Events


path = os.path.dirname(os.path.abspath(__file__))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + '/model/trainer.yml')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

interval = 1

lock = asyncio.Lock()

async def save_recognized(session, face_area, face_id):
    label = face_id
    employees = os.path.join(path, 'employees', str(label))
    os.makedirs(employees, exist_ok=True)
    face_name = f'person_{label}:{datetime.now()}.jpg'
    face_path = os.path.join(employees, face_name)
    cv2.imwrite(face_path, face_area)
    event = Events(event_time=datetime.now(), photo_path=face_path, employee_id=label)
    session.add(event)
    await session.commit()

async def save_unrecognized(face_area):
    label = 'unknown'
    unrecognized = os.path.join(path, 'unrecognized')
    os.makedirs(unrecognized, exist_ok=True)
    face_name = f'face_{label}:{datetime.now()}.jpg'
    face_path = os.path.join(unrecognized, face_name)
    cv2.imwrite(face_path, face_area)


async def recognition_proces():
    start = time.time()
    save_time = start
    delay = 1.0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

        current = time.time()
        elapsed_time = current - start

        for x, y, w, h in faces:
            face_area = gray[y: y + h, x: x + w]
            face_id, conf = recognizer.predict(face_area)

            if conf < 100:
                async with lock:
                    if current - save_time >= delay:
                        session = async_session()
                        await save_recognized(session, face_area, face_id)
                        await session.close()


            elif elapsed_time >= interval:
                async with lock:
                    if current - save_time >= delay:
                        await save_unrecognized(face_area)
                        save_time = current


            else:
                async with lock:
                    if current - save_time >= delay:
                        await save_unrecognized(face_area)
                        save_time = current

            cv2.putText(frame, 'DETECTED', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video.release()
    cv2.destroyAllWindows()

async def run_recognition():
    await recognition_proces()

