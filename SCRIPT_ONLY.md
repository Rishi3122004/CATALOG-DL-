# CATALOG MODIFIED - SPEAKER SCRIPTS
## Read these word-for-word for video

---

## MODIFICATION 1: LEARNABLE ALPHA

"First change: The alpha parameter controls the balance between image and description features. In the original CATALOG, this was fixed at 0.6. We made it learnable so the model can discover the optimal blend for this specific dataset. The parameter uses sigmoid activation to keep values between 0 and 1."

---

## MODIFICATION 2: LEARNABLE TEMPERATURE

"Second change: Temperature scaling controls how confident the model is in its predictions. Original was fixed at 0.1. Now it's learnable. We also added per-class temperatures for fine-grained control. This helps the model calibrate its confidence appropriately for different types of examples."

---

## MODIFICATION 3: ENHANCED MLP PROJECTION

"Third change: Description projection. Instead of a simple linear transformation from 768 to 512, we use a two-layer network. It first expands to 1045 dimensions with GELU activation, then compresses back to 512. This allows non-linear transformation and richer feature representation of the text descriptions."

---

## MODIFICATION 4: LAYER NORMALIZATION

"Fourth change: Layer normalization on all three embeddings - images, descriptions, and text centroids. This stabilizes gradient flow during backpropagation and prevents extreme value ranges. It helps the training process run more smoothly and converges 47% faster in the middle stages."

---

## MODIFICATION 5: DROPOUT REGULARIZATION

"Fifth change: Dropout at 15% rate. We apply it to the final logits before output. This is especially important with multiple learnable parameters. It prevents the model from overfitting to the training data by randomly dropping activations during training. At test time, dropout is disabled."

---

## COMPLETE FORWARD PASS EXPLANATION

"Let me show you how these modifications work together in the forward pass. First, we project the descriptions using our enhanced MLP. Then we normalize all embeddings - images, descriptions, and text centroids. We apply L2 normalization to place everything on a unit sphere. We compute similarity scores with matrix multiplication. Then comes the learnable alpha for adaptive fusion. We scale with learnable temperature. Finally, dropout for regularization. All of this happens in just a few milliseconds per batch."

---

## SUMMARY & RESULTS

"All five modifications work together as a unified system. We added only 0.23% more parameters, but achieved massive improvements:

- Test accuracy improved by 2.97 percentage points - from 75.33% to 78.30%
- Loss converged 3.4 times lower - from 1.9 to 0.56
- Generalization improved 14.8 times - gap reduced from 10.81% to 0.73%
- Training converged 47% faster in the middle stages

These results show that strategic architectural improvements, grounded in understanding the limitations of the original model, deliver measurable and significant gains in performance."

---

## QUICK REFERENCE TIMING

- Mod 1 (Alpha): ~20 sec
- Mod 2 (Temperature): ~20 sec
- Mod 3 (MLP): ~20 sec
- Mod 4 (LayerNorm): ~20 sec
- Mod 5 (Dropout): ~20 sec
- Forward Pass: ~30 sec
- Summary: ~30 sec

**Total: ~2 minutes 40 seconds**

