import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

train_dir = 'D:\\SoftProject\\ImagenesRecog\\Output'
label_dir = 'D:\\SoftProject\\ImagenesRecog\\Output'

# Lista para almacenar las imágenes y las etiquetas
images = []
labels = []

# Recorre las subcarpetas dentro del directorio de entrenamiento
for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)
    # Recorre las imágenes dentro de cada subcarpeta
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        # Lee la imagen en escala de grises
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Agrega la imagen y la etiqueta a las listas correspondientes
        images.append(img)
        labels.append(label)

# Convierte las listas a matrices NumPy
images = np.array(images)
labels = np.array(labels)

# Divide los datos en conjuntos de entrenamiento, validación y prueba
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Realiza cualquier procesamiento adicional necesario en los datos de entrenamiento, validación y prueba
# ...

# Imprime la forma de los conjuntos de datos
print('Train images shape:', train_images.shape)
print('Train labels shape:', train_labels.shape)
print('Validation images shape:', val_images.shape)
print('Validation labels shape:', val_labels.shape)
print('Test images shape:', test_images.shape)
print('Test labels shape:', test_labels.shape)
#canal de colores RGB =3
num_channels =3
#numero de clases distintas: "pertenece a la cultura" y "no pertenece a la cultura".
num_classes = 2
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(800, 800, num_channels)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes)
])

# Compila el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrena el modelo
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Evalúa el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Realiza predicciones en nuevas imágenes
new_images = 'D:\SoftProject\ImagenesRecog\ImgEvaluacion'
predictions = model.predict(new_images)
print(predictions)