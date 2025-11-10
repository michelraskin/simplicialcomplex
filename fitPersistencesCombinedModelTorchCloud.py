import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns


from sklearn.preprocessing import StandardScaler
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.model_selection import GroupShuffleSplit
# from tensorflow.keras.utils import to_categorical

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report

import numpy as np
import os
import re


from google.cloud import storage
import numpy as np
import re
import io
from io import BytesIO

# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./strange-metrics-266115-d7ca05a26868.json"

# GCS folder path (no './' here)
GCS_BUCKET = "simplicialcomplex-outputbucket"
GCS_PREFIX = "savefiles/"  # path inside bucket

bucket_name = GCS_BUCKET
blob_name = "data/myData.npz"

# --- initialize client ---
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)

# --- download file into memory ---
data_bytes = blob.download_as_string()

# --- load npz into numpy ---
data_npz = np.load(BytesIO(data_bytes))

myData = data_npz['myData']
myData2 = data_npz['myData2']
myY = data_npz['myY']
myActors = data_npz['myActors']

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(myData, myY, groups=myActors))

X_train, X_test, X_train2, X_test2 = myData[train_idx], myData[test_idx], myData2[train_idx], myData2[test_idx]
y_train, y_test = myY[train_idx], myY[test_idx]


# X_train, X_test, X_train2, X_test2, y_train, y_test = train_test_split(
#     myData, myData2, myY, test_size=0.2, shuffle=True, stratify=myActors, random_state=20
# )

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Model definition ---
class DualInputCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(DualInputCNN, self).__init__()
        
        # Branch 1: for 128x128x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        
        # Branch 2: for 32x32x8
        self.branch2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )

        # Compute flattened sizes (lazy init: done in forward)
        self.flatten1_size = None
        self.flatten2_size = None

        # Merge + classifier head
        self.fc1 = nn.Linear(1, 1)  # placeholder — reset after knowing sizes
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x1, x2):
        # Pass through each branch
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        
        # Flatten
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        if not hasattr(self, 'fc1_initialized') or not self.fc1_initialized:
            merged_dim = x1.shape[1] + x2.shape[1]
            self.fc1 = nn.Linear(merged_dim, 256).to(x1.device)  
            self.fc1_initialized = True
        
        # Merge branches
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# --- Instantiate model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DualInputCNN(num_classes=8).to(device)

# --- Loss and optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train_tensor = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
X_train2_tensor = torch.tensor(X_train2.transpose(0, 3, 1, 2), dtype=torch.float32)
y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
X_test_tensor = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
X_test2_tensor = torch.tensor(X_test2.transpose(0, 3, 1, 2), dtype=torch.float32)
y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

dataset = TensorDataset(X_train_tensor, X_train2_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)
test_loader = DataLoader(TensorDataset(X_test_tensor, X_test2_tensor, y_test_tensor), batch_size=256)

# ================================================================
# Training Setup
# ================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU (if any)
else:
    device = torch.device("cpu")   # fallback
model = DualInputCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 20

best_val_auc = 0.0
auroc = MulticlassAUROC(num_classes=8).to(device)
top3acc = MulticlassAccuracy(num_classes=8, top_k=3).to(device)

# ================================================================
# Training Loop with Checkpoint
# ================================================================
for epoch in range(num_epochs):
    model.train()
    for X_batch, X2_batch, y_batch in train_loader:
        X_batch, X2_batch, y_batch = X_batch.to(device), X2_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch, X2_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for X_val, X2_val, y_val in val_loader:
            X_val, X2_val, y_val = X_val.to(device), X2_val.to(device), y_val.to(device)
            outputs = model(X_val, X2_val)
            preds = torch.softmax(outputs, dim=1)
            val_preds.append(preds)
            val_labels.append(y_val)
    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    val_auc = auroc(val_preds, val_labels).item()
    val_top3 = top3acc(val_preds, val_labels).item()

    y_pred = torch.argmax(val_preds, dim=1)

    accuracy = (y_pred == val_labels).float().mean()

    print(f"Epoch {epoch+1}/{num_epochs} - val_auc: {val_auc:.4f} - top3_acc: {val_top3:.4f} - val_acc: {accuracy.item():.4f}")

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
    for X_batch, X2_batch, y_batch in test_loader:
        X_batch, X2_batch = X_batch.to(device), X2_batch.to(device)
        outputs = model(X_batch, X2_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    val_preds = torch.cat(all_preds)
    val_labels = torch.cat(all_labels)
    val_auc = auroc(val_preds, val_labels).item()
    val_top3 = top3acc(val_preds, val_labels).item()

    y_pred = torch.argmax(val_preds, dim=1)

    accuracy = (y_pred == val_labels).float().mean()

    print(f"Epoch {epoch+1}/{num_epochs} - val_auc: {val_auc:.4f} - top3_acc: {val_top3:.4f} - val_acc: {accuracy.item():.4f}")

# Classification report
class_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']
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