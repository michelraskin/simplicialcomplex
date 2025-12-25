# Debugging the Combined Fusion Model

## Current Problem

**Expected**: Combined model ≥ max(mel_acc, topo_acc) = 68.6%
**Actual**: Combined model = 50.0% ❌

This is a **critical failure**. The fusion model is worse than flipping a coin (16.7% for 6 classes).

---

## Why Fusion Models Fail

### Issue 1: Feature Scale Mismatch
**Problem**: Mel spectrograms and topological features have different scales
- Mel: (1, 128, 256) with values [0, 1] (normalized spectrograms)
- Topology: (6, 32, 32) with values [0, ???] (persistence image scale unknown)

**Solution**: Normalize both inputs to same scale

### Issue 2: Branch Imbalance
**Problem**: One branch dominates during training
- Strong branch: High gradients, fast learning
- Weak branch: Low gradients, ignored by optimizer

**Solution**: Separate learning rates or gradient balancing

### Issue 3: Insufficient Regularization
**Problem**: Fusion layer overfits to training data
- 128-dim fusion vector → 64 → 64 → 6 (too many parameters)
- No batch normalization in fusion classifier

**Solution**: Add BatchNorm, increase dropout

### Issue 4: Cold Start Problem
**Problem**: Both branches start with random weights
- Takes many epochs for branches to learn useful features
- Fusion layer gets bad signals initially

**Solution**: Pre-train branches separately, then fine-tune fusion

### Issue 5: Architecture Issues
**Problem**: Branches may be unbalanced
- Mel branch: 7 conv blocks (deep)
- Topo branch: 3 conv blocks (shallow)
- Different representational capacity

**Solution**: Balance architectures or use adaptive weighting

---

## Diagnostic Steps

### Step 1: Check Input Statistics
```python
# Check data ranges
print("Mel stats:")
print(f"  Min: {X_train_tensor.min():.4f}")
print(f"  Max: {X_train_tensor.max():.4f}")
print(f"  Mean: {X_train_tensor.mean():.4f}")
print(f"  Std: {X_train_tensor.std():.4f}")

print("\nTopology stats:")
print(f"  Min: {X_train2_tensor.min():.4f}")
print(f"  Max: {X_train2_tensor.max():.4f}")
print(f"  Mean: {X_train2_tensor.mean():.4f}")
print(f"  Std: {X_train2_tensor.std():.4f}")
```

**Expected**: Both should be roughly [0, 1] with similar means/stds

### Step 2: Check Branch Learning
```python
# Monitor branch-specific losses
for epoch in range(num_epochs):
    # ... training loop ...

    # After each batch
    mel_pred_entropy = -torch.sum(mel_logits.softmax(1) * mel_logits.log_softmax(1), dim=1).mean()
    topo_pred_entropy = -torch.sum(topo_logits.softmax(1) * topo_logits.log_softmax(1), dim=1).mean()

    # Low entropy = confident predictions
    # High entropy = random guessing
    print(f"Mel entropy: {mel_pred_entropy:.4f}, Topo entropy: {topo_pred_entropy:.4f}")
```

**Expected**: Both entropies should decrease during training

### Step 3: Check Gradient Flow
```python
# After backward pass
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.4f}")
```

**Expected**: Gradients in both branches should be similar magnitude

### Step 4: Check Fusion Weights
```python
# After training
with torch.no_grad():
    # Get embeddings
    mel_emb = model.mel_branch(X_batch[:1])
    topo_emb = model.topo_branch(X_batch2[:1])

    print(f"Mel embedding norm: {mel_emb.norm():.4f}")
    print(f"Topo embedding norm: {topo_emb.norm():.4f}")
```

**Expected**: Similar norms; if one is 10x larger, it dominates

---

## Solutions (Ordered by Likelihood)

### Solution 1: Pre-train Branches (RECOMMENDED)
**Hypothesis**: Random initialization causes poor fusion learning

```python
# 1. Train mel branch alone
model_mel = MelBranch().to(device)
train(model_mel, train_loader_mel, epochs=20)
model_mel.eval()

# 2. Train topo branch alone
model_topo = TopoBranch().to(device)
train(model_topo, train_loader_topo, epochs=40)
model_topo.eval()

# 3. Initialize fusion model with pre-trained weights
model_combined = CombinedModel()
model_combined.mel_branch.load_state_dict(model_mel.state_dict())
model_combined.topo_branch.load_state_dict(model_topo.state_dict())

# 4. Fine-tune entire model (with lower LR for branches)
optimizer = torch.optim.Adam([
    {'params': model_combined.mel_branch.parameters(), 'lr': 1e-5},
    {'params': model_combined.topo_branch.parameters(), 'lr': 1e-5},
    {'params': model_combined.classifier.parameters(), 'lr': 1e-3}
])
```

### Solution 2: Feature Normalization
**Hypothesis**: Scale mismatch causes one branch to dominate

```python
class ImprovedCombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_branch = MelBranch()
        self.topo_branch = TopoBranch()

        # Add batch normalization before fusion
        self.mel_norm = nn.BatchNorm1d(64)
        self.topo_norm = nn.BatchNorm1d(64)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  # Add BN here too
            nn.ReLU(),
            nn.Dropout(0.3),  # Increase dropout
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)
        )

    def forward(self, mel_input, topo_input):
        mel_emb = self.mel_norm(self.mel_branch(mel_input))
        topo_emb = self.topo_norm(self.topo_branch(topo_input))
        combined = torch.cat([mel_emb, topo_emb], dim=1)
        return self.classifier(combined)
```

### Solution 3: Learned Fusion Weights
**Hypothesis**: One modality is simply better, should be weighted more

```python
class AdaptiveFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_branch = MelBranch()
        self.topo_branch = TopoBranch()

        # Learnable fusion weights
        self.fusion_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 2 weights (for mel and topo)
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Linear(64, 6)

    def forward(self, mel_input, topo_input):
        mel_emb = self.mel_branch(mel_input)  # (B, 64)
        topo_emb = self.topo_branch(topo_input)  # (B, 64)

        # Compute fusion weights
        combined = torch.cat([mel_emb, topo_emb], dim=1)  # (B, 128)
        weights = self.fusion_gate(combined)  # (B, 2)

        # Weighted combination
        fused = weights[:, 0:1] * mel_emb + weights[:, 1:2] * topo_emb  # (B, 64)
        return self.classifier(fused)
```

### Solution 4: Gradient Balancing
**Hypothesis**: Gradient magnitudes differ, causing training instability

```python
class GradBalancedFusion(nn.Module):
    # ... same architecture ...

    def forward(self, mel_input, topo_input):
        mel_emb = self.mel_branch(mel_input)
        topo_emb = self.topo_branch(topo_input)

        # Balance gradients during backprop
        mel_emb = mel_emb * torch.tensor(0.5).to(mel_emb.device)
        topo_emb = topo_emb * torch.tensor(0.5).to(topo_emb.device)

        combined = torch.cat([mel_emb, topo_emb], dim=1)
        return self.classifier(combined)

# Or use gradient scaling in training loop
def train_with_grad_balance(model, loader, optimizer):
    for mel_batch, topo_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(mel_batch, topo_batch)
        loss = criterion(output, y_batch)
        loss.backward()

        # Scale gradients
        for name, param in model.named_parameters():
            if 'mel_branch' in name and param.grad is not None:
                param.grad *= 0.5
            if 'topo_branch' in name and param.grad is not None:
                param.grad *= 0.5

        optimizer.step()
```

### Solution 5: Deeper Fusion
**Hypothesis**: Late fusion loses fine-grained interaction

```python
class EarlyFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fuse at feature map level, not embedding level
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.topo_conv = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Process fused features
        self.shared_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 16+16=32 channels
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... more layers ...
        )

        self.classifier = nn.Linear(64, 6)

    def forward(self, mel_input, topo_input):
        mel_feat = self.mel_conv(mel_input)
        topo_feat = self.topo_conv(topo_input)

        # Concatenate at feature map level
        combined = torch.cat([mel_feat, topo_feat], dim=1)
        features = self.shared_conv(combined)
        return self.classifier(features.mean([2, 3]))  # Global avg pool
```

---

## Recommended Debugging Protocol

### Phase 1: Diagnosis (30 minutes)
1. Run diagnostic code to check:
   - Input statistics
   - Branch entropies
   - Gradient norms
   - Embedding norms

2. Identify the issue:
   - Scale mismatch → Solution 2
   - One branch dead → Solution 1
   - Training instability → Solution 4

### Phase 2: Quick Fix (1-2 hours)
1. Try Solution 1 (pre-training) first
   - Train branches separately
   - Load weights into combined model
   - Fine-tune with small LR

2. If still fails, try Solution 2
   - Add BatchNorm
   - Standardize inputs
   - Increase dropout

### Phase 3: Advanced Fixes (if needed)
3. Try Solution 3 (learnable weights)
4. Try Solution 5 (early fusion)

---

## Expected Results After Fixes

**Minimum acceptable**: 68% (match mel baseline)
**Target**: 70-72% (modest improvement)
**Stretch goal**: 75% (significant gain)

If you can't reach 68%, that's still okay for the analysis paper—just frame it as:
> "While fusion does not improve over individual modalities in our current
> implementation, this suggests opportunities for future architectural
> innovations in multi-modal emotion recognition."

---

## Alternative: Ensemble Instead of Fusion

If fusion continues to fail, use an ensemble:

```python
# Train separate models
mel_model = train_mel_model()
topo_model = train_topo_model()

# Ensemble predictions
def ensemble_predict(mel_input, topo_input):
    mel_pred = mel_model(mel_input).softmax(1)
    topo_pred = topo_model(topo_input).softmax(1)

    # Average probabilities
    combined_pred = 0.7 * mel_pred + 0.3 * topo_pred  # Weight by performance
    return combined_pred.argmax(1)
```

This should guarantee ≥68% accuracy.

---

Next: I'll create a fixed implementation notebook!
