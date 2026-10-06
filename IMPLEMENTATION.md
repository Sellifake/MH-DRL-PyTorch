# Implementation notes

This repository is a PyTorch re-implementation (2026) of MH-DRL. It follows the method of the paper; the original 2024 code is kept in the history of [jiefeng0109/MH-DRL](https://github.com/jiefeng0109/MH-DRL). This page maps the paper to the code and lists every place where the implementation differs from the paper, with the reason.

## Paper → code

| Paper | Code |
|---|---|
| Sec. 3.1 multi-agent modeling: one agent per band, actions select / deselect | `mhdrl/agent.py: MultiDQN, run_mhdrl` |
| Eq. (5)–(8) reward $R_t = P_c - \alpha\,\mathrm{Corr} + \beta P_b$ | `agent.py: observe` (written as $(P_c-\alpha\,\mathrm{Corr})(1-(n_s-n_d)^2/n_d^2)$, which is the same expression after substituting $\beta$) |
| Eq. (9) reward shared equally by the selecting agents | `agent.py: r_vec` |
| Sec. 3.2 / Table 1 PE-Net (3D conv 8/16/32, 2D conv 64, FC 256/128) | `mhdrl/nets.py: PENet` |
| Algorithm I Stage I: random band combination training, evaluation without fine-tuning | `mhdrl/masked.py: MaskedEvaluator` |
| Eq. (10) state = extracted features $F_l$ | `masked.py: evaluate` (mean of $F_l$ over training samples) |
| Sec. 3.3 / Fig. 2 participated, high- and low-confidence bands, $k=\lfloor (m+n)/2 \rfloor$, Eq. (12) | `agent.py: teacher_advice` |
| Eq. (13) teachers BS-Nets → SICNN → TWCNN in three periods | `agent.py: run_mhdrl` (`guide` steps), `mhdrl/teachers.py` |
| Sec. 3.4 DQN with target network and memory buffer | `agent.py: run_mhdrl` |
| Sec. 4.1 splits: 5% / 3% / 5% of labeled pixels per class for training | `mhdrl/data.py: split` |
| Sec. 4.3 compared methods: pixel-wise classifiers for BS-Nets, ABCNN, MR-SVM, DDCNN; spatial windows for TWCNN, SICNN, DRLBS | `mhdrl/evaluate.py` |

## Differences from the paper

| Item | Paper | This code | Reason |
|---|---|---|---|
| PE-Net evaluation | one PE-Net trained on the training set | the training set is split into 4 class-stratified folds; one PE-Net per fold, each training sample is scored only by the PE-Net that did not see it | a network scored on its own training samples is ~100% accurate for almost any band subset, so the reward cannot rank subsets |
| PE-Net input window | 25 × 25 | 15 × 15 inside the band search; 21 × 21 and 25 × 25 for the final classifier | faster search |
| PE-Net layers | Table 1 | Table 1 plus batch normalization and a linear embedding of the 0/1 band mask; batch-norm uses batch statistics at evaluation time | lets one network evaluate subsets of any size consistently |
| State | extracted features $F_l$ | $F_l$ concatenated with the current 0/1 band mask | gives each agent the current selection |
| $P_c$ | ratio of correctly classified samples | class-averaged probability of the true class | smoother signal; small classes count equally |
| $\mathrm{Corr}$ | sum of pairwise correlation coefficients | mean absolute pairwise correlation coefficient | scale-free; the scale is absorbed by $\alpha$ (0.1) |
| Reward sharing, Eq. (9) | $R/N_p$ | $R$ minus its running mean, divided by its running std, then shared by $n_d/N_p$ (`--no_center` gives the paper form) | gives a positive or negative signal to the selecting agents |
| Joint action | each agent acts independently; $P_b$ penalizes the number of bands | the $n_d$ agents with the largest $Q(\text{select})-Q(\text{deselect})$ select, with random swaps for exploration (`--act argmax` gives the paper form); bands added by a teacher are kept and the lowest-ranked others are dropped to keep $n_d$ bands | keeps exactly $n_d$ bands, as in the `AFS-任意波段数` version of the 2024 code |
| Teachers | the participated bands are fed to a teacher at every step | each teacher scores all bands once; at every step it takes the top-$k$ of the participated bands by that score | the three teachers are ranking methods; much faster |
| Output | bands selected at the last step | top-$n_d$ bands by Q-advantage at the last state and at the highest-reward state; the one with the higher reward on the training data is kept | more stable output |
| Low-confidence bands | text and Fig. 2: selected at $t-1$ and deselected at $t$ (the set formula below Fig. 2 swaps the two conditions) | text and Fig. 2 | low-confidence bands must be a subset of the participated bands |
| Hyper-parameters | 90 steps, memory 45, batch 16, learning rate 0.01 | 1500 steps (300 teacher-guided), batch 32, learning rate 0.001, 4 updates per step; $\gamma=0.9$ and target update every 10 updates as in the paper | longer search |
| Final classifier | PE-Net | ensemble of 4 PE-Nets (window 21 / 25, two initializations each) trained on the selected bands | — |
| Training samples | a fixed ratio per class | a fixed ratio per class, at least 3 | affects 3 small classes of Indian Pines |
| Runs | mean ± std of 30 runs | mean ± std of 3 random splits | compute |
| TWCNN threshold | 0.1 (Sec. 4.2) | 0.5, as in the group's TWCNN code | — |
| CNNeGA, RLFSR-Net | compared | not included | no public code |

## Reproducibility

The data split depends only on `--seed`. All random number generators used by a run (PyTorch and NumPy) are seeded from `--seed`, but cuDNN autotuning and GPU atomics are not bit-deterministic, so re-running the band search can give slightly different bands and accuracies. The released weights reproduce the reported numbers exactly with `mhdrl.infer`.

## Requirements

A CUDA GPU is required.
