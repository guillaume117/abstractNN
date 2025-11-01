import numpy as np
from PIL import Image

"""Charge et prétraite une image"""
img = Image.open('fmnist_sample_1_Pullover.png')

# Détermine le nombre de canaux attendus
expected_channels = 3  # Par défaut RGB
if target_shape and len(target_shape) >= 2:
    if isinstance(target_shape[1], int):
        expected_channels = target_shape[1]

# Convertit dans le bon format
if expected_channels == 1:
    img = img.convert('L')  # Niveaux de gris
else:
    img = img.convert('RGB')

# Redimensionne si nécessaire
if target_shape and len(target_shape) >= 4:
    target_h, target_w = target_shape[2], target_shape[3]
    img = img.resize((target_w, target_h))

img_array = np.array(img, dtype=np.float32) / 255.0

# Ajoute la dimension channel si nécessaire (pour images en niveaux de gris)
if len(img_array.shape) == 2:
    img_array = np.expand_dims(img_array, axis=0)
else:
    # Transpose pour correspondre à (channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))

# Ajoute la dimension batch
img_array = np.expand_dims(img_array, axis=0)

return img_array
img = Image.open('fmnist_sample_1_Pullover.png').convert('L')
img = img.resize((28, 28))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = img_array.reshape(1, 1, 28, 28)

 
np.save('test.npy', img_array)