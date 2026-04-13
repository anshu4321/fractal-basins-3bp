# Experiment 3.5: Training Convergence Sanity Check

## Numbers

- MLP-raw at N=300k, seed=42
- F1 at epoch 60: 0.6779
- F1 at epoch 600: 0.6384
- Delta F1: -0.0396 (F1 DECREASED)
- Peak F1: ~0.678 near epoch 60, then gradual overfitting

## Pass criterion evaluation

Section 3.5: "If F1 increases by >0.01, the Phase A runs were undertrained."

F1 did not increase. It decreased by 0.040, indicating overfitting beyond epoch 60.
Phase A's 60-epoch budget was sufficient. The model was converged (arguably slightly
overtrained) at 60 epochs.

Note: the 600-epoch run used a cosine LR schedule matched to 600 epochs, so the LR
profile differs from Phase A's 60-epoch schedule. The F1 at epoch 60 (0.678) is
slightly higher than Phase A's (0.669) because the LR is still near its peak at
epoch 60 under the 600-epoch schedule. The subsequent decline is from overfitting
under continued high LR followed by slow annealing.

**D5 status: PASS. Undertraining is not a confound.**
