# SM / LC Training Analysis — 2026-06-13

## Results so far

| Config | Job | Run dir | Status | Best val loss | Best epoch |
|---|---|---|---|---|---|
| Baseline (no SM/LC) | — | run_20260612_175854 | Done | **−22.571** | ep96 |
| SM-only | 106636 | run_20260613_172500 | Early stopped ep87 | **−17.761** | ep77 |
| SM+LC | 106638 | run_20260613_172504 | Early stopped ep62 | **−14.071** | ep52 |
| Restricted baseline (2016-2022) | 106669 | run_20260613_201659 | Early stopped ep18 | **−13.061** | ep8 |
| Baseline+GPW (LR=5e-4, bias init) | 106671 | run_20260613_203006 | Killed @ ep38 | ~−12.9 | ep38 |
| Baseline+GPW (LR=1e-4, bias init) | 106680 | — | Running | — | — |

Loss metric: Poisson NLL (zone-level, logsumexp aggregation). More negative = better.

---

## Restricted baseline result (2016-2022 date range, LR=5e-4)

Best val loss: **−13.061** @ epoch 8. Early stopped at epoch 18.

**Why it stopped early**: val spiked at ep9 (−1.51 → effectively losing the best) and again catastrophically at ep12 (+14.57). The early-stopping counter started from ep8 (best) and never reset, reaching patience=10 at ep18. The model simply never recovered to −13.06.

**Key finding**: **SM-only (−17.76) beats the restricted baseline (−13.06)** on the same 2016-2022 window. This means soil moisture adds genuine predictive value beyond what ERA5+VIIRS provides on the restricted date range. The comparison isn't perfectly fair (restricted baseline stopped too early — it was converging and may have gone lower with more epochs), but SM-only's clear advantage at the same data window is encouraging.

**Val spike pattern**: The same instability observed in the GPW run (positive val loss at certain epochs) is present here in the restricted baseline at LR=5e-4. This strongly suggests the spikes are a property of the Poisson NLL loss landscape at this learning rate, not specific to population weighting. Both runs showed val spikes around ep9-ep12 that derailed convergence.

**Action**: The GPW run was resubmitted with LR=1e-4 (job 106680) to test whether the lower learning rate eliminates these spikes.

---

## Why did SM-only converge at a higher loss than baseline? (val −17.76 vs −22.57)

### Primary cause: training data restriction

`--add_sm` restricts training to **2016–2022** (Sentinel-1 availability).
The baseline trains on **2012–2023** — roughly 11 years vs 6 years of data.

This matters significantly:
- Fewer unique training samples → less exposure to epidemic cycle variation
- Fewer inter-annual patterns learned → worse generalisation to validation set
- The model may simply not have seen enough dengue-season diversity to reach the same NLL floor

This is almost certainly the dominant cause. To confirm: run the baseline restricted to
2016–2022. If it also converges to ~−17 to −18, the data restriction fully explains
the gap and SM adds no information. If the restricted baseline still reaches −22, then
the SM branch itself is the issue (noise, optimization difficulty, etc.).

### Secondary causes (smaller effect)

- **SM data has gaps**: many patches have near-zero validity (no Sentinel-1 swath
  coverage for that 6-day window). These contribute near-zero signal while still
  consuming model capacity in the SM branch.
- **More parameters, less data**: the SM branch adds parameters that need more
  samples to converge, but the training set is actually smaller.
- **SM early stopped at ep87 vs baseline ep96**: minor, ~9 extra epochs for the
  baseline may account for a small fraction of the gap.

### Takeaway

Do not interpret the val loss gap as "SM hurt the model." The comparison is unfair
because baseline and SM-only train on different amounts of data. A controlled experiment
(baseline restricted to 2016–2022) is needed before drawing any conclusions about
whether SM adds predictive value.

---

## Why did SM+LC perform worse than SM-only and baseline?

### 1. The comparison isn't fair yet

SM-only is still improving at epoch 76 (val −17.71), and the baseline ran 96 epochs.
SM+LC early-stopped at epoch 52. A more complex model likely needs more epochs to converge —
the SM-only trajectory suggests the same pattern.

### 2. Consistent positive bias in SM+LC

Throughout SM+LC training, the validation bias was persistently positive (+0.76 to +2.41).
SM-only's bias oscillates near zero. Persistent positive bias = systematic overprediction,
not just noise. This points to a specific failure mode, not random instability.

### 3. Possible causes (in order of likelihood)

**a) LC embedding random initialization corrupts fused representation**
The LC branch (nn.Embedding → CNN → [B,128]) starts with random weights and produces
noisy output. This noisy vector is concatenated with ERA5+VIIRS features already trained
from a checkpoint, injecting a spurious positive signal into the FC fusion head.
SM-only avoids this because its SM branch resumed from a checkpoint that had already
partially learned the SM signal.

**b) More branches = harder joint optimization**
VIIRS + ERA5 + SM + LC all fusing into the same FC head makes the loss landscape
more complex. With the same LR (5e-4) for all branches, the optimizer may not
allocate adequate gradient signal to each branch within 52 epochs.

**c) LC at 86×86 resolution adds limited orthogonal information**
The static risk raster already captures geographic variation. MODIS MCD12Q1 land cover
at 500m resampled to 86×86 may not add signal that isn't already explained by static features,
so the embedding learns to correlate with existing features rather than adding new signal.

**d) Early stopping was too aggressive**
patience=10 at epoch 52 is likely too early for a 4-branch model. The SM-only run
is still gaining at epoch 73. With patience=20 or more epochs, SM+LC might recover.

---

## How to diagnose

1. **Run LC-only (no SM)**: isolates whether LC helps/hurts relative to baseline.
   If LC-only beats baseline, the problem is joint SM+LC optimization, not LC itself.

2. **Train SM+LC from scratch with more epochs**: `--num_epochs 150 --patience 20`.
   If early stopping at ep52 was premature, it should converge further.

3. **Freeze LC branch for first 20–30 epochs, then unfreeze**: prevents random
   LC initialization from corrupting the ERA5/SM representations during early training.
   Implementation: set `lc_branch.requires_grad_(False)` for first N epochs.

4. **Differential learning rates**: use a lower LR for the LC branch (e.g., 1e-4)
   and normal LR (5e-4) for the rest, to slow down the noisy LC updates.

5. **Log per-branch gradient norms**: if LC branch has 10× larger gradients than ERA5,
   it's dominating updates and destabilizing the other branches.

---

## Bottom line

Do not conclude that LC is harmful. The most likely explanation is that:
- The LC branch's random initialization injected a positive noise signal
- The early stopping at ep52 didn't give it enough time to correct
- The training setup (single LR, patience=10) isn't well-suited to multi-branch fine-tuning

A fair test would be: train SM+LC from scratch for 100+ epochs with patience=20,
or freeze the LC branch for the first 20 epochs before joint training.
