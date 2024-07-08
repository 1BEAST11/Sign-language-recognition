import os
import pickle
import numpy as np
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb) # выделение руки на изображении
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x # координата х маркера
                    y = hand_landmarks.landmark[i].y # координата у маркера

                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux) # запись координат маркера

            # создание метки
            label_array = np.zeros(9, dtype=float)  # создаем массив из 3 ячеек, заполненных нулями
            label_index = int(dir_)  # получаем индекс метки из имени директории
            label_array[label_index] = 1.0  # устанавливаем 1.0 в соответствующей ячейке
            labels.append(label_array)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
