import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import kagglehub
import statsmodels.api as sm

# import tensorflow_addons as tfa
import cv2
from keras import backend as K

from sklearn.preprocessing import StandardScaler
import math
from sklearn.datasets import load_iris
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Activation, Concatenate
from keras.optimizers import SGD
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.datasets import mnist
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import SparseCategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model


import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from scipy.io import wavfile
from scipy.fftpack import fft
import cv2
import os
import librosa
import librosa.display
from glob import glob
import skimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint,LearningRateScheduler
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from pydub import AudioSegment
from tqdm import tqdm
import tempfile

from tensorflow.keras import regularizers
tf.debugging.set_log_device_placement(False)

import numpy as np
import os
import re

import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric

folder = './savefiles'

maxElements = 3000

def findFilesFromPattern(pattern, maxvalue = maxElements, maxdim = 2):
    pattern = re.compile(pattern + r'_(\d+)_(\d+)\.npy')
    heatmaps_dict = {}

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            i, j = map(int, match.groups())
            filepath = os.path.join(folder, filename)
            data = np.load(filepath)
            if j >= maxvalue * maxdim:
                continue
            
            if i not in heatmaps_dict:
                heatmaps_dict[i] = []
            
            while len(heatmaps_dict[i]) <= j:
                heatmaps_dict[i].append(None)
            
            heatmaps_dict[i][j] = data
    return [heatmaps_dict[i] for i in sorted(heatmaps_dict.keys())]

mfccwasserstein = findFilesFromPattern('wassersteinMfccHeat')
melwasserstein = findFilesFromPattern('wassersteinHeat')
meltimeeuclid = findFilesFromPattern('timeMetricHeat')
meleuclid = findFilesFromPattern('euclideanHeat')

def load_spectrograms(prefix, path='./savefiles'):
    pattern = os.path.join(path, f"{prefix}_*.npy")
    file_list = sorted(glob(pattern)) 
    return [np.load(file) for i, file in enumerate(file_list) if i < maxElements]

myRaw = [
    load_spectrograms("mel_spectrogram_dbs_angry"),
    load_spectrograms("mel_spectrogram_dbs_disgusted"),
    load_spectrograms("mel_spectrogram_dbs_fearful"),
    load_spectrograms("mel_spectrogram_dbs_happy"),
    load_spectrograms("mel_spectrogram_dbs_neutral"),
    load_spectrograms("mel_spectrogram_dbs_surprised"),
    load_spectrograms("mel_spectrogram_dbs_sad")
]
print(len(mfccwasserstein))
print(len(sum([x[1::2] for x in mfccwasserstein], [])))


myData = np.array([
                    sum([x for x in myRaw], [])
                    ])
print('finish data')
myData = myData.astype('float32')
myData = np.transpose(myData, (1, 2, 3, 0))
myY = np.array(sum([[i for x in range(len(melwasserstein[i]) // 2)] for i in range(7)], []))

myY = to_categorical(myY, num_classes=7)

myData2 = np.array([
                    sum([x[::2] for x in meleuclid], []),
                    sum([x[1::2] for x in meleuclid], []),
                    sum([x[::2] for x in meltimeeuclid], []),
                    sum([x[1::2] for x in meltimeeuclid], []),
                    sum([x[::2] for x in mfccwasserstein], []),
                    sum([x[1::2] for x in mfccwasserstein], []),
                    sum([x[::2] for x in melwasserstein], []),
                    sum([x[1::2] for x in melwasserstein], [])
                    ])
print('finish data')
myData2 = myData2.astype('float32')
myData2 = np.transpose(myData2, (1, 2, 3, 0))

X_train, X_test, X_train2, X_test2, y_train, y_test = train_test_split(
    myData, myData2, myY, test_size=0.2, shuffle=True, stratify=myY, random_state=20
)

print('start model')

checkpoint = ModelCheckpoint(
    'best_model.h5',             
    monitor='val_auc',
    save_best_only=True,         
    mode='max',                  
    verbose=1
)

input_128 = Input(shape=(128, 128, 1))
input_32 = Input(shape=(32, 32, 8))

x1 = Conv2D(32, (3, 3), activation='relu')(input_128)
x2 = Conv2D(32, (3, 3), activation='relu')(input_32)

x1 = BatchNormalization()(x1)
x2 = BatchNormalization()(x2)

x1 = Activation('relu')(x1)
x2 = Activation('relu')(x2)

x1 = MaxPooling2D()(x1)
x2 = MaxPooling2D()(x2)

x1 = Conv2D(32, (3, 3), activation='relu')(x1)
x2 = Conv2D(32, (3, 3), activation='relu')(x2)

x1 = MaxPooling2D()(x1)
x2 = MaxPooling2D()(x2)

x1 = Flatten()(x1)
x2 = Flatten()(x2)


merged = Concatenate()([x1, x2])
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.2)(merged)
output = Dense(7, activation='softmax')(merged)


model = Model(inputs=[input_128, input_32], outputs=output)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',
                  TopKCategoricalAccuracy(k=3, name='top_3_accuracy'), 
                  AUC(multi_label=True)])


model.summary()

plot_model(model, to_file='model_architecture_double.png', show_shapes=True, show_layer_names=True)

history = model.fit([X_train, X_train2], y_train, epochs=60, batch_size=256, validation_split=0.2, callbacks=[checkpoint])

model = load_model('best_model.h5')

from sklearn.metrics import confusion_matrix, classification_report
class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'surprised', 'sad']
y_pred = model.predict([X_test, X_test2])
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
report = classification_report(y_test_classes, y_pred_classes, target_names=class_labels)
print(report)

results = model.evaluate([X_test, X_test2], y_test, verbose=1)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# confusion matrix
from sklearn.metrics import confusion_matrix

import seaborn as sns

cm = confusion_matrix(y_test_classes, y_pred_classes)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
