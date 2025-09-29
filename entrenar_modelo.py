# entrenar_modelo.py
import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = []
labels = []
carpetas = ['Passion_Red', 'Celeste_Pearl', 'Grey', 'SR150.50', 'Racing_Blue', 'Brown', 'Fluorecent_Pink', 'Lemon_Green', 'Light_Lilac', 'Light_Orange', 'Lilac', 'Violet', 'Blue', 'Green']

for carpeta in carpetas:
    path = f"muestras/{carpeta}"
    for archivo in os.listdir(path):
        img = cv2.imread(f"{path}/{archivo}")
        if img is None:
            continue
        img = cv2.resize(img, (50, 50))
        promedio_color = img.mean(axis=(0, 1))  # RGB promedio
        data.append(promedio_color)
        labels.append(carpeta)

data = np.array(data)
labels = np.array(labels)

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(data, labels)

with open('modelo_color.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("✅ Modelo entrenado y guardado como 'modelo_color.pkl'")
