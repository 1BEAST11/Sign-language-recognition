import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 9 # количество жестов для распознавания
dataset_size = 100 # количество собираемых изображений каждого жеста

cap = cv2.VideoCapture(0) # окно камеры

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()

        # Отзеркаливание кадра по горизонтали
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, ' Press "Q" to start', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) # отзеркаливание кадра по горизонтали

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) # полученное изображение записывается в папку с названием counter

        counter += 1

cap.release()
cv2.destroyAllWindows()
