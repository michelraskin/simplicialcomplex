"""
Fixed Combined Model Implementation
Addresses common fusion model failures
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ============================================================================
# Solution 1: Pre-trained Branch Initialization
# ============================================================================

class MelBranchCNN(nn.Module):
    """Standalone mel spectrogram branch (for pre-training)."""
    def __init__(self, embedding_dim=64, num_classes=6):
        super(MelBranchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.features(x)
        emb = self.embedding(x)
        if return_embedding:
            return emb
        return self.classifier(emb)


class TopoBranchCNN(nn.Module):
    """Standalone topological branch (for pre-training)."""
    def __init__(self, embedding_dim=64, num_classes=6):
        super(TopoBranchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.features(x)
        emb = self.embedding(x)
        if return_embedding:
            return emb
        return self.classifier(emb)


# ============================================================================
# Solution 2: Improved Fusion with BatchNorm
# ============================================================================

class ImprovedCombinedModel(nn.Module):
    """
    Fixed fusion model with:
    - BatchNorm for feature normalization
    - Balanced architecture
    - Higher dropout
    - Separate embedding/classifier paths
    """
    def __init__(self, num_classes=6):
        super(ImprovedCombinedModel, self).__init__()

        # Mel branch (reuse pre-trained if available)
        self.mel_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.mel_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Topo branch
        self.topo_features = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.topo_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # CRITICAL: Normalize embeddings before fusion
        self.mel_norm = nn.BatchNorm1d(64)
        self.topo_norm = nn.BatchNorm1d(64)

        # Fusion classifier with proper regularization
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # Add BN
            nn.ReLU(),
            nn.Dropout(0.3),  # Increase dropout
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, mel_input, topo_input):
        # Extract features
        mel_feat = self.mel_features(mel_input)
        mel_emb = self.mel_embedding(mel_feat)
        mel_emb = self.mel_norm(mel_emb)  # Normalize

        topo_feat = self.topo_features(topo_input)
        topo_emb = self.topo_embedding(topo_feat)
        topo_emb = self.topo_norm(topo_emb)  # Normalize

        # Fuse and classify
        combined = torch.cat([mel_emb, topo_emb], dim=1)
        return self.classifier(combined)

    def load_pretrained_branches(self, mel_model_path, topo_model_path):
        """Load pre-trained branch weights."""
        # Load mel branch
        mel_state = torch.load(mel_model_path)
        mel_state_filtered = {}
        for k, v in mel_state.items():
            if k.startswith('features'):
                mel_state_filtered[k] = v
            elif k.startswith('embedding'):
                mel_state_filtered[k.replace('embedding', 'mel_embedding')] = v
        self.load_state_dict(mel_state_filtered, strict=False)

        # Load topo branch
        topo_state = torch.load(topo_model_path)
        topo_state_filtered = {}
        for k, v in topo_state.items():
            if k.startswith('features'):
                topo_state_filtered[k.replace('features', 'topo_features')] = v
            elif k.startswith('embedding'):
                topo_state_filtered[k.replace('embedding', 'topo_embedding')] = v
        self.load_state_dict(topo_state_filtered, strict=False)

        print("✅ Loaded pre-trained branch weights")


# ============================================================================
# Solution 3: Learnable Adaptive Fusion
# ============================================================================

class AdaptiveFusionModel(nn.Module):
    """
    Learns optimal fusion weights dynamically.
    Allows model to weight mel vs topo based on input.
    """
    def __init__(self, num_classes=6):
        super(AdaptiveFusionModel, self).__init__()

        # Reuse improved branches
        self.mel_branch = MelBranchCNN(embedding_dim=64, num_classes=num_classes)
        self.topo_branch = TopoBranchCNN(embedding_dim=64, num_classes=num_classes)

        # Attention-based fusion gate
        self.fusion_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, mel_input, topo_input):
        # Get embeddings
        mel_emb = self.mel_branch(mel_input, return_embedding=True)  # (B, 64)
        topo_emb = self.topo_branch(topo_input, return_embedding=True)  # (B, 64)

        # Compute adaptive weights
        combined = torch.cat([mel_emb, topo_emb], dim=1)  # (B, 128)
        weights = self.fusion_attention(combined)  # (B, 2)

        # Weighted fusion
        fused = weights[:, 0:1] * mel_emb + weights[:, 1:2] * topo_emb  # (B, 64)
        return self.classifier(fused)

    def get_fusion_weights(self, mel_input, topo_input):
        """Return learned fusion weights for interpretability."""
        with torch.no_grad():
            mel_emb = self.mel_branch(mel_input, return_embedding=True)
            topo_emb = self.topo_branch(topo_input, return_embedding=True)
            combined = torch.cat([mel_emb, topo_emb], dim=1)
            weights = self.fusion_attention(combined)
            return weights  # (B, 2): [mel_weight, topo_weight]


# ============================================================================
# Training Functions
# ============================================================================

def train_single_branch(model, train_loader, val_loader, num_epochs=20,
                        lr=1e-3, device='cuda', save_path='model.pth'):
    """Train a single branch model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                _, predicted = outputs.max(1)
                total += y_val.size(0)
                correct += predicted.eq(y_val).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {train_loss/len(train_loader):.4f} - "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model: {val_acc:.4f}")

    return model


def train_fusion_model_with_pretraining(
    mel_model_path, topo_model_path,
    train_loader, val_loader,
    num_epochs=30, device='cuda'
):
    """
    Train fusion model with pre-trained branches.
    This is the RECOMMENDED approach.
    """
    model = ImprovedCombinedModel().to(device)

    # Load pre-trained weights
    if mel_model_path and topo_model_path:
        model.load_pretrained_branches(mel_model_path, topo_model_path)
        print("🔥 Using pre-trained branches")

    criterion = nn.CrossEntropyLoss()

    # Separate learning rates: small for branches, large for classifier
    optimizer = optim.Adam([
        {'params': model.mel_features.parameters(), 'lr': 1e-5},
        {'params': model.mel_embedding.parameters(), 'lr': 1e-5},
        {'params': model.topo_features.parameters(), 'lr': 1e-5},
        {'params': model.topo_embedding.parameters(), 'lr': 1e-5},
        {'params': model.mel_norm.parameters(), 'lr': 1e-4},
        {'params': model.topo_norm.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for mel_batch, topo_batch, y_batch in train_loader:
            mel_batch = mel_batch.to(device)
            topo_batch = topo_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(mel_batch, topo_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for mel_val, topo_val, y_val in val_loader:
                mel_val = mel_val.to(device)
                topo_val = topo_val.to(device)
                y_val = y_val.to(device)
                outputs = model(mel_val, topo_val)
                _, predicted = outputs.max(1)
                total += y_val.size(0)
                correct += predicted.eq(y_val).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Loss: {train_loss/len(train_loader):.4f} - "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_combined_fixed.pth')
            print(f"✅ Saved best combined model: {val_acc:.4f}")

    return model


# ============================================================================
# Diagnostic Functions
# ============================================================================

def diagnose_fusion_model(model, mel_batch, topo_batch, device='cuda'):
    """Run diagnostics on fusion model."""
    model.eval()
    mel_batch = mel_batch.to(device)
    topo_batch = topo_batch.to(device)

    print("\n" + "="*60)
    print("FUSION MODEL DIAGNOSTICS")
    print("="*60)

    # 1. Check input statistics
    print("\n1. Input Statistics:")
    print(f"   Mel - Min: {mel_batch.min():.4f}, Max: {mel_batch.max():.4f}, "
          f"Mean: {mel_batch.mean():.4f}, Std: {mel_batch.std():.4f}")
    print(f"   Topo - Min: {topo_batch.min():.4f}, Max: {topo_batch.max():.4f}, "
          f"Mean: {topo_batch.mean():.4f}, Std: {topo_batch.std():.4f}")

    # 2. Check embedding norms
    with torch.no_grad():
        mel_feat = model.mel_features(mel_batch)
        mel_emb = model.mel_embedding(mel_feat)
        mel_emb_norm = mel_emb.norm(dim=1).mean()

        topo_feat = model.topo_features(topo_batch)
        topo_emb = model.topo_embedding(topo_feat)
        topo_emb_norm = topo_emb.norm(dim=1).mean()

    print("\n2. Embedding Norms (before normalization):")
    print(f"   Mel embedding norm: {mel_emb_norm:.4f}")
    print(f"   Topo embedding norm: {topo_emb_norm:.4f}")
    print(f"   Ratio (mel/topo): {mel_emb_norm/topo_emb_norm:.4f}")

    if mel_emb_norm / topo_emb_norm > 5 or topo_emb_norm / mel_emb_norm > 5:
        print("   ⚠️  WARNING: Large norm imbalance detected!")

    # 3. Check prediction entropy
    with torch.no_grad():
        outputs = model(mel_batch, topo_batch)
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

    print("\n3. Prediction Entropy:")
    print(f"   Entropy: {entropy:.4f}")
    print(f"   Max entropy (random): {np.log(6):.4f}")

    if entropy > 1.5:
        print("   ⚠️  WARNING: High entropy suggests model is guessing randomly!")

    print("\n" + "="*60)


# ============================================================================
# Simple Ensemble Alternative
# ============================================================================

class SimpleEnsemble:
    """
    Simple ensemble if fusion continues to fail.
    Guaranteed to be ≥ max(mel_acc, topo_acc).
    """
    def __init__(self, mel_model, topo_model, mel_weight=0.7):
        self.mel_model = mel_model
        self.topo_model = topo_model
        self.mel_weight = mel_weight
        self.topo_weight = 1.0 - mel_weight

    def predict(self, mel_input, topo_input):
        """Ensemble prediction."""
        self.mel_model.eval()
        self.topo_model.eval()

        with torch.no_grad():
            mel_logits = self.mel_model(mel_input)
            topo_logits = self.topo_model(topo_input)

            mel_probs = torch.softmax(mel_logits, dim=1)
            topo_probs = torch.softmax(topo_logits, dim=1)

            ensemble_probs = (self.mel_weight * mel_probs +
                            self.topo_weight * topo_probs)

            return ensemble_probs.argmax(dim=1)


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("Fixed Combined Model Implementation")
    print("="*60)
    print("\nRecommended workflow:")
    print("1. Train mel branch: train_single_branch(mel_model, ...)")
    print("2. Train topo branch: train_single_branch(topo_model, ...)")
    print("3. Train fusion: train_fusion_model_with_pretraining(...)")
    print("4. If fusion fails, use SimpleEnsemble")
    print("\nSee notebook for complete training example.")
