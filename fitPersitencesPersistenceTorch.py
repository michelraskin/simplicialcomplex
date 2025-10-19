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

maxElements = 8000

def findFilesFromPattern(pattern, maxvalue = maxElements, maxdim = 2):
    pattern = re.compile(pattern + r'_(.*?)_(.*?)_(.*?)_(\d+)_(\d+)\.npy')
    heatmaps_dict = {}

    heatmaps_dict = {}

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            dataset, actor, emotion, i, j = map(str, match.groups())
            i, j = int(i), int(j)
            filepath = os.path.join(folder, filename)
            data = np.load(filepath)
            if j >= maxvalue * maxdim:
                continue
            
            if i not in heatmaps_dict:
                heatmaps_dict[i] = []
            
            while len(heatmaps_dict[i]) <= j:
                heatmaps_dict[i].append(None)
            
            heatmaps_dict[i][j] = {'data': data, 'dataset': dataset, 'actor': actor, 'emotion':emotion}
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
    load_spectrograms("savee"),
    load_spectrograms("tess"),
    load_spectrograms("radvess"),
    load_spectrograms("cremad"),
]
print(len(mfccwasserstein))
print(len([j['data'] for j in sum([x[1::2] for x in mfccwasserstein], [])]))


myData = np.array([
                    sum([x for x in myRaw], [])
                    ])
print('finish data')
myData = myData.astype('float32')
myData = np.transpose(myData, (1, 2, 3, 0))
myEmotionMap = {
    'neutral': 1, 'calm':2, 'happy':3, 'sad':4, 'angry':5, 'fearful':6, 'disgust':7, 'surprised':8
}
myY = np.array([myEmotionMap[j['emotion'].split('_')[-1]] - 1 for j in sum([x[::2] for x in meleuclid], [])])
myActors = np.array([j['actor']  for j in sum([x[::2] for x in meleuclid], [])])
print(np.unique(myActors))

print(np.unique(myY))

myY = to_categorical(myY, num_classes=8)

myData2 = np.array([
                    [j['data'] for j in sum([x[::2] for x in meleuclid], [])],
                    [j['data'] for j in  sum([x[1::2] for x in meleuclid], [])],
                    [j['data'] for j in  sum([x[::2] for x in meltimeeuclid], [])],
                    [j['data'] for j in  sum([x[1::2] for x in meltimeeuclid], [])],
                    [j['data'] for j in  sum([x[::2] for x in mfccwasserstein], [])],
                    [j['data'] for j in  sum([x[1::2] for x in mfccwasserstein], [])],
                    [j['data'] for j in  sum([x[::2] for x in melwasserstein], [])],
                    [j['data'] for j in  sum([x[1::2] for x in melwasserstein], [])]
                    ])
print('finish data')
myData2 = myData2.astype('float32')
myData2 = np.transpose(myData2, (1, 2, 3, 0))

X_train, X_test, y_train, y_test = train_test_split(
    myData2, myY, test_size=0.2, shuffle=True, stratify=myY, random_state=20
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ================================================================
# Model Definition
# ================================================================
class CNNModel(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),  # for input 32×32 after two poolings
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ================================================================
# Data Preparation
# ================================================================
# Assume X_train, y_train, X_test, y_test are numpy arrays
# Shapes: X: (N, 32, 32, 8), y: one-hot (N, 7)
X_train_tensor = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=256)

# ================================================================
# Training Setup
# ================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU (if any)
else:
    device = torch.device("cpu")   # fallback
model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100

best_val_auc = 0.0
auroc = MulticlassAUROC(num_classes=8).to(device)
top3acc = MulticlassAccuracy(num_classes=8, top_k=3).to(device)

# ================================================================
# Training Loop with Checkpoint
# ================================================================
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            preds = torch.softmax(outputs, dim=1)
            val_preds.append(preds)
            val_labels.append(y_val)
    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    val_auc = auroc(val_preds, val_labels).item()
    val_top3 = top3acc(val_preds, val_labels).item()

    print(f"Epoch {epoch+1}/{num_epochs} - val_auc: {val_auc:.4f} - top3_acc: {val_top3:.4f}")

    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved new best model.")

# ================================================================
# Evaluation
# ================================================================
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

# Classification report
class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'surprised', 'sad']
report = classification_report(all_labels, all_preds, target_names=class_labels)
print(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()