from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import cv2
from pathlib import Path
import numpy as np
import os
import time
from tqdm import tqdm
import yaml
from pyefd import elliptic_fourier_descriptors
from pyefd import normalize_efd
import shutil

model_file = r'C:\Users\w12j692\Desktop\MainFileCattle\EFD_Sorted\MOD3\shapes_cnn_backbone.h5'
image_folder = r'C:\Users\w12j692\Desktop\MainFileAurora\AuroraSet2'
out_dir = r'C:\Users\w12j692\Desktop\MainFileCattle\EFD_Sorted\Cattle_set_EFD\bad'

classifier = tf.keras.models.load_model(model_file)
classes = ['bad', 'good']

files = os.listdir(image_folder)

for file in tqdm(files):

    if file.upper().endswith('.TIF'):
        try:
            img = cv2.imread(os.path.join(image_folder, file), 0)
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(contours, key=cv2.contourArea)

            coeffs = elliptic_fourier_descriptors(np.squeeze(c), 50)
            coeffs = normalize_efd(coeffs)
            bn = coeffs.flatten()[3:]

            bn_ten = tf.convert_to_tensor(bn)
            val = classifier.predict(tf.expand_dims(bn_ten, 0, name=None))
            ind = val.argmax()
            prob = val[0][ind]
            class_name = classes[ind]

            output = os.path.join(out_dir, class_name, str(int(prob*10)*10))
            os.makedirs(output, exist_ok=True)
            shutil.move(os.path.join(image_folder, file), os.path.join(output, file))

        except:
            print('bad image')


