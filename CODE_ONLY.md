# CATALOG MODIFIED - CODE BLOCKS ONLY
## Show these on screen while explaining

---

## MODIFICATION 1: LEARNABLE ALPHA

```python
# Learnable fusion weight (MODIFICATION 1)
self.alpha = nn.Parameter(torch.tensor(0.6))

# In forward pass:
alpha = torch.sigmoid(self.alpha)
logits = alpha * W + (1 - alpha) * Q
```

---

## MODIFICATION 2: LEARNABLE TEMPERATURE

```python
# Learnable temperature (MODIFICATION 2)
self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
self.class_temps = nn.Parameter(torch.ones(num_classes))

# In forward pass:
logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
class_temps = torch.softmax(self.class_temps, dim=0)
logits = logits * logit_scale * (1 + class_temps).unsqueeze(0)
```

---

## MODIFICATION 3: ENHANCED MLP PROJECTION

```python
# Enhanced description projection (MODIFICATION 3)
self.desc_projection = nn.Sequential(
    nn.Linear(desc_dim, 1045),      # Expand to 1045
    nn.GELU(),                       # Non-linear activation
    nn.Linear(1045, feature_dim)     # Compress to 512
)

# In forward pass:
descriptions = self.desc_projection(descriptions)  # [B, 512]
```

---

## MODIFICATION 4: LAYER NORMALIZATION

```python
# Layer norms (MODIFICATION 4)
self.image_norm = nn.LayerNorm(feature_dim)
self.desc_norm = nn.LayerNorm(feature_dim)
self.text_norm = nn.LayerNorm(feature_dim)

# In forward pass:
images = self.image_norm(images)
descriptions = self.desc_norm(descriptions)
text_centroids = self.text_norm(text_centroids)
```

---

## MODIFICATION 5: DROPOUT REGULARIZATION

```python
# Dropout (MODIFICATION 5)
self.dropout = nn.Dropout(0.15)

# In forward pass:
logits = self.dropout(logits)
return logits
```

---

## COMPLETE FORWARD PASS PIPELINE

```python
def forward(self, images, descriptions, labels, text_centroids):
    # Step 1: Project descriptions
    descriptions = self.desc_projection(descriptions)
    
    # Step 2: Layer normalize
    images = self.image_norm(images)
    descriptions = self.desc_norm(descriptions)
    text_centroids = self.text_norm(text_centroids)
    
    # Step 3: L2 normalize
    images = F.normalize(images, p=2, dim=-1)
    descriptions = F.normalize(descriptions, p=2, dim=-1)
    text_centroids = F.normalize(text_centroids, p=2, dim=-1)
    
    # Step 4: Compute similarities
    W = images @ text_centroids.t()
    Q = descriptions @ text_centroids.t()
    
    # Step 5: Learnable fusion
    alpha = torch.sigmoid(self.alpha)
    logits = alpha * W + (1 - alpha) * Q
    
    # Step 6: Learnable temperature
    logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
    class_temps = torch.softmax(self.class_temps, dim=0)
    logits = logits * logit_scale * (1 + class_temps).unsqueeze(0)
    
    # Step 7: Dropout
    logits = self.dropout(logits)
    
    return logits
```

---

## CLASS INITIALIZATION

```python
class CALOGModified(nn.Module):
    """Modified CATALOG with learnable components"""
    
    def __init__(self, num_classes=10, feature_dim=512, desc_dim=768):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # All 5 modifications initialized here
        self.desc_projection = nn.Sequential(
            nn.Linear(desc_dim, 1045),
            nn.GELU(),
            nn.Linear(1045, feature_dim)
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.6))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.class_temps = nn.Parameter(torch.ones(num_classes))
        
        self.image_norm = nn.LayerNorm(feature_dim)
        self.desc_norm = nn.LayerNorm(feature_dim)
        self.text_norm = nn.LayerNorm(feature_dim)
        
        self.dropout = nn.Dropout(0.15)
```

