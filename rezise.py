
import cv2
import os

# Directorio que contiene las imágenes originales
input_dir = 'D:\SoftProject\ImagenesRecog\Input'

# Directorio para almacenar las imágenes redimensionadas
output_dir = 'D:\SoftProject\ImagenesRecog\Output'

# Dimensiones de redimensionamiento deseadas
new_width = 800
new_height = 800

# Recorre las imágenes en el directorio de entrada
for filename in os.listdir(input_dir): 
    input_path = os.path.join(input_dir, filename)
    img = cv2.imread(input_path)
    resized_img = cv2.resize(img, (new_width, new_height))
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, resized_img)