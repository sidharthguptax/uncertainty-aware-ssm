# Results Log

Working record of experimental results, interpretations and caveats. Organised by experiment.
Numbers are copied from script output; the per-day tables and JSON files named in each section regenerate them.

---

## 1. Implementation: DS³M replication

Reference: Xu, Peng & Chen, arXiv:2106.02329 v3 (Jan 2025). Official code: github.com/Sherry-Xu/Deep-Switching-State-Space-Model.

The replication follows the **official code** rather than the paper's specification, so that published numbers can be reproduced. Deviations from the paper (documented in the code headers):

| # | Component | Paper | Official code (followed here) |
|---|---|---|---|
| 1 | Transition matrix $\Gamma$ | learned $K \times K$ matrix | $\Gamma = \tfrac12\,\mathrm{softmax}(\mathrm{Linear}(T_0)) + \tfrac12 I$, forcing $\Gamma_{ii} \ge 0.5$ |
| 2 | Variance output | networks output $\log\Sigma$ (eqs. 5, 7) | networks output a standard deviation via Softplus |
| 3 | Network structure | 2-layer MLP per mean and per variance | shared 2-layer ReLU trunk, separate mean and std heads |
| 4 | Emission input | separate $z_t$ per regime in eq. (21) | one $z_t$, sampled under the sampled $d_t$, feeds every regime's emission |
| 5 | Initial conditions | $d_1 \sim$ Uniform, $z_1 \sim N(0,1)$ | $d_0 \sim$ Uniform, $z_0 = 0$ |
| 6 | KL annealing | 0.01 → 1 | 0.01 for the first half of training, then rising to ≈0.51 |

Simplifications relative to the official code:
- No early stopping (theirs averages all past validation losses and rarely triggers).
- No retraining loop. The official `main.py` retrains until the model uses both regimes on the test set; this is not mentioned in the paper.

Error found in the official evaluation: `main.py` compares regime forecasts against `d[-500:]` of the 2001-value label file, while forecast targets are `y[-500:]`. Given the verified alignment (Section 2), their labels are shifted one step later than the observations being forecast. This replication uses correctly aligned labels.

Settings (Appendix C, toy example): GRU hidden size 10, latent dimension 2, $K = 2$, window length 20, Adam (lr 0.001, ReduceLROnPlateau factor 0.1, patience 10), batch size 64, 100 epochs. The backward RNN producing $A_t$ has the same size as $h_t$.

---

## 2. Toy data conventions (verified against the official CSV)

Script: `experiments/toy/check_official.py`.

- **Noise convention.** The paper's $N(0, s)$ in eq. (23) means **standard deviation $s$**, not variance. Evidence: the official data's regime-0 std of $y$ is 19.98; eq. (23) under the standard-deviation reading predicts ≈19.4, and under the variance reading ≈6.6–7.3.
- **Alignment.** In the official files `y[i]` pairs with `d[i]`; `d` has one extra value at the end (2001 vs 2000). With this alignment, regime-1 std of $y$ is 0.76; with the alternative alignment, regime-0 points leak into regime 1 at each switch and it rises to 6.3.
- **Regime-1 discrepancy.** The official regime-1 data (std 0.76) is quieter than eq. (23) predicts under either noise convention (≈1.24 or ≈1.34). Their simulation code is not published, so the source cannot be identified. Consequence: an exact simulator of eq. (23) does not reproduce their dataset; validation therefore uses their CSV directly.
- **Own simulator** (`experiments/toy/simulate_toy.py`, standard-deviation convention): regime stds 19.83 / 1.26; estimated $\Gamma$ diagonal 0.943 / 0.957; mean durations 17.5 / 22.8 (theory: 20).
- **Data split.** The official code uses the last 500 points for testing and splits the first 1500 into 75% training / 25% validation, not the 1000/480/500 subsequences stated in Section 4.1.

---

## 3. Toy replication: 5 seeds

Scripts: `experiments/toy/run_seeds.py`. Results: `experiments/toy/results_5seeds.json`.

| Seed | RMSE | 90% coverage | Regime accuracy | Mean predicted duration |
|---|---|---|---|---|
| 1 | 14.932 | 0.842 | 0.796 | 9.1 |
| 2 | 16.593 | 0.858 | 0.806 | 8.8 |
| 3 | 14.418 | 0.882 | 0.814 | 10.2 |
| 4 | 16.021 | 0.852 | 0.844 | 11.6 |
| 5 | 20.167 | 0.832 | 0.588 | 500 (collapsed) |

**Seed 5 collapsed to a single regime**: it forecast the same regime at all 500 test points, and its validation loss barely decreased. The official code's retraining loop would discard and silently rerun such a run. **Collapse rate: 1 of 5 runs.**

Comparison with the paper (Table 1, DS³M, mean ± std over 5 runs):

| | All 5 seeds | 4 non-collapsed seeds | Paper |
|---|---|---|---|
| RMSE | 16.43 ± 2.26 | 15.49 ± 0.99 | 14.57 ± 0.35 |
| Regime accuracy | 0.770 ± 0.103 | 0.815 ± 0.021 | 0.788 ± 0.033 |
| Mean duration | — | 9.9 | 7.5 / 7.6 |
| 90% coverage | 0.853 ± 0.019 | 0.859 ± 0.017 | not reported |

Learned $\Gamma$, seed 3: $\begin{pmatrix} 0.84 & 0.16 \\ 0.07 & 0.93 \end{pmatrix}$. With regime labels swapped this is $\begin{pmatrix} 0.93 & 0.07 \\ 0.16 & 0.84 \end{pmatrix}$, against the paper's $\begin{pmatrix} 0.91 & 0.09 \\ 0.18 & 0.82 \end{pmatrix}$.

Interpretation:
- Regime accuracy and durations match or exceed the paper (correct label alignment may account for part of the accuracy difference).
- RMSE is about one point higher and more variable across seeds. Training had not converged at 100 epochs in any run.
- Lowest validation loss does not imply lowest RMSE (seed 2: validation loss 159, RMSE 16.6).
- **Coverage is below nominal in every run (0.83–0.88)**, even in-domain on data generated by a switching state-space model. This is the in-domain baseline for later coverage comparisons.

---

## 4. SPY setup

Scripts: `experiments/spy/download_spy.py`, `ds3m/data.py` (`load_spy`).

- **Data:** SPY daily adjusted close (yfinance, `auto_adjust=True`), 1999-12-31 to 2024-12-31; daily log returns $r_t = \log P_t - \log P_{t-1}$; 6,289 returns.
- **Splits:** train 2000–2015 (4,005 windows), validation 2016–2018 (734 windows), test 2019–2024 (1,510 trading days).
- **Normalisation:** training-period mean and std only (mean 0.000155, std 0.01272).
- **Windows:** length 20, built as for the toy data; each test day is forecast from the preceding 20 returns.
- **Model settings:** toy settings (hidden 10, latent 2, $K = 2$, 100 epochs). One training run takes about 7 minutes on the MacBook CPU.
- **Evaluation periods, fixed before any results were seen:**

| Period | Dates |
|---|---|
| 2019 | full year |
| 2020 | full year |
| COVID crash | 2020-02-20 to 2020-04-30 |
| 2021 | full year |
| 2022 | full year |
| 2023–2024 | two years |

---

## 5. SPY: first DS³M fit (seed 3)

> **Superseded in part by Section 8.** Seed 3 turned out to be the only one of five runs in which training proceeded normally under the official recipe. The coverage figures below are not representative of DS³M trained with that recipe.

Scripts: `experiments/spy/train_spy.py`, `experiments/spy/evaluate_spy.py`. Results: `experiments/spy/forecasts_seed3.csv`.

Learned $\Gamma = \begin{pmatrix} 0.99993 & 0.00007 \\ 0.49991 & 0.50009 \end{pmatrix}$.

- Regime 0 is near-absorbing (expected duration ≈15,000 days).
- Regime 1's diagonal sits exactly at the 0.5 floor imposed by deviation 1; the learnable part of its row has gone entirely to "switch back". **The floor is binding.**
- Implied stationary share of regime 1 ≈ 0.01%.

**Regime collapse:** the model forecast regime 0 on every test day (regimes used: 1). In this run DS³M behaved as a single-regime model, with volatility changes handled by the continuous within-regime dynamics.

Point forecasts: RMSE 0.01251 against 0.01255 for always predicting zero. No meaningful point-forecast skill, consistent with the unpredictability of daily return direction.

| Period | Days | 90% coverage | Mean interval width |
|---|---|---|---|
| 2019 | 252 | 0.933 | 0.0286 |
| 2020 | 253 | 0.893 | 0.0482 |
| COVID crash | 50 | 0.780 | 0.0926 |
| 2021 | 252 | 0.917 | 0.0271 |
| 2022 | 251 | 0.865 | 0.0490 |
| 2023–2024 | 502 | 0.898 | 0.0269 |
| **Overall** | 1510 | **0.901** | |

Interpretation: overall coverage is nominal only because over-coverage in calm years offsets under-coverage in stress periods. Intervals widen about 3.4× in the crash relative to calm years, but not enough.

Caveats: single seed (see Section 8 for five seeds); training not converged (validation loss still falling at epoch 100); one value of $K$; 200 Monte Carlo samples per forecast.

---

## 6. SPY: GARCH(1,1) baselines

Script: `experiments/spy/garch_spy.py`. Results: `experiments/spy/forecasts_garch_normal.csv`, `experiments/spy/forecasts_garch_t.csv`.

Same protocol as DS³M: constant mean + GARCH(1,1), parameters estimated on 2000–2015 only and then frozen; one-step-ahead forecasts using data up to the previous day.

Fitted parameters (returns in percent):

| | $\mu$ | $\omega$ | $\alpha$ | $\beta$ | $\alpha+\beta$ | $\nu$ |
|---|---|---|---|---|---|---|
| GARCH-N | 0.0559 | 0.0196 | 0.0991 | 0.8864 | 0.986 | — |
| GARCH-t | 0.0731 | 0.0143 | 0.1016 | 0.8915 | 0.993 | 7.36 |

90% coverage by period, with DS³M (seed 3) for comparison:

| Period | DS³M seed 3 | GARCH-N | GARCH-t |
|---|---|---|---|
| 2019 | 0.933 | 0.944 | 0.937 |
| 2020 | 0.893 | 0.901 | 0.897 |
| COVID crash | 0.780 | 0.800 | 0.800 |
| 2021 | 0.917 | 0.921 | 0.917 |
| 2022 | 0.865 | 0.865 | 0.861 |
| 2023–2024 | 0.898 | 0.912 | 0.908 |
| Overall | 0.901 | 0.909 | 0.905 |

Mean interval width in the crash: DS³M 0.093, GARCH-N 0.116, GARCH-t 0.117.

Interpretation:
- **The qualitative pattern is shared by all models**: over-coverage in calm years, under-coverage in the crash and 2022. It is not specific to DS³M; across five DS³M seeds (Section 8) the crash under-coverage is far more severe than GARCH's.
- DS³M seed 3 vs GARCH in the crash: 39 vs 40 covered days out of 50. This near-parity does **not** hold across seeds: mean DS³M crash coverage over five seeds is 0.436 (Section 8).
- **Heavy tails do not explain the crash under-coverage at the 90% level.** GARCH-t ($\nu \approx 7.4$) and GARCH-N give near-identical coverage; at the 5%/95% points a unit-variance $t_{7.4}$ and a Gaussian nearly coincide. Tail shape should matter at 99%, which has not yet been evaluated.
- GARCH widens its intervals more in the crash than DS³M, with almost no coverage gain.
- GARCH is fitted by maximum likelihood and gives identical results on every run; DS³M's results vary strongly with the random seed (Section 8).

---

## 7. Formal coverage tests

Script: `experiments/spy/coverage_tests.py`; tests in `ds3m/evaluation.py`.

- **Kupiec (1995):** likelihood-ratio test that the miss rate equals the nominal 10%.
- **Christoffersen (1998):** likelihood-ratio test that misses are independent over time (no clustering).

| Model | Miss rate | Kupiec p (overall) | Christoffersen p (overall) | Kupiec p (COVID crash) | Kupiec p (2022) |
|---|---|---|---|---|---|
| DS³M seed 3 | 0.099 | 0.932 | 0.138 | 0.013 | 0.074 |
| GARCH-N | 0.091 | 0.223 | 0.632 | 0.035 | 0.074 |
| GARCH-t | 0.095 | 0.545 | 0.509 | 0.035 | 0.048 |

Interpretation:
- **Over the full test period, all three models pass both standard tests.**
- **In the COVID crash, all three show significant under-coverage** (p < 0.05). 2022 is borderline.
- The standard whole-period backtest therefore declares every model well calibrated while concealing a significant failure during the regime break. This supports the proposal's methodological claim that coverage must be measured around documented regime breaks rather than over an aggregate held-out period.
- No significant clustering overall: about 150 misses over six years dilute the crash-period clustering; 50 days is too short for a reliable clustering test.

Caveats:
- Kupiec assumes independent misses; within the crash misses are likely dependent, so its p-values there are probably too small.
- Six tests (two periods × three models): individual borderline p-values deserve limited weight, although the crash result is consistent across all models.
- Evaluation periods were fixed before results were seen.
- DS³M results are from one seed, and that seed is not representative (Section 8). The GARCH results and the main finding (whole-period tests pass, crash window fails) are unaffected.

---

## 8. SPY: DS³M over 5 seeds, official training recipe

Script: `experiments/spy/run_seeds_spy.py` (variant `official`: date-ordered batches, no gradient clipping). Results: `experiments/spy/summary_5seeds.json`, `experiments/spy/forecasts_seed{1..5}.csv`.

| Seed | Regimes used | Overall | 2019 | 2020 | COVID crash | 2021 | 2022 | 2023–2024 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.812 | 0.925 | 0.676 | 0.220 | 0.877 | 0.641 | 0.876 |
| 2 | 1 | 0.927 | 0.972 | 0.834 | 0.380 | 0.972 | 0.821 | 0.982 |
| 3 | 1 | 0.901 | 0.933 | 0.893 | 0.780 | 0.917 | 0.865 | 0.898 |
| 4 | 1 | 0.875 | 0.948 | 0.771 | 0.340 | 0.948 | 0.701 | 0.942 |
| 5 | 2 | 0.926 | 0.968 | 0.838 | 0.460 | 0.988 | 0.849 | 0.958 |
| **Mean ± std** | | 0.888 ± 0.048 | 0.949 ± 0.021 | 0.802 ± 0.083 | **0.436 ± 0.211** | 0.940 ± 0.045 | 0.775 ± 0.099 | 0.931 ± 0.043 |

GARCH-N / GARCH-t crash coverage for comparison: 0.800 / 0.800.

Learned $\Gamma$ by seed:

| Seed | $\Gamma$ |
|---|---|
| 1 | $\begin{pmatrix} 0.530 & 0.470 \\ 0.017 & 0.983 \end{pmatrix}$ |
| 2 | $\begin{pmatrix} 0.969 & 0.031 \\ 0.480 & 0.520 \end{pmatrix}$ |
| 3 | $\begin{pmatrix} 0.99993 & 0.00007 \\ 0.49991 & 0.50009 \end{pmatrix}$ |
| 4 | $\begin{pmatrix} 0.958 & 0.042 \\ 0.445 & 0.555 \end{pmatrix}$ |
| 5 | $\begin{pmatrix} 0.656 & 0.344 \\ 0.216 & 0.784 \end{pmatrix}$ |

In four seeds one regime is near-absorbing and the other's diagonal sits at or near the 0.5 floor.

Training behaviour (validation loss, printed every 10 epochs):

| Seed | Behaviour | Learning rate |
|---|---|---|
| 1 | fell to epoch 30, then rose and plateaued | cut repeatedly, 1e-8 by epoch 90 |
| 2 | rose almost from the start (17,067 → 21,307) | cut from epoch 30, 1e-8 by epoch 70 |
| 3 | fell steadily (16,835 → 12,473) | stayed at 1e-3 |
| 4 | rose from epoch 10 (14,950 → 15,251) | cut from epoch 30, 1e-8 by epoch 70 |
| 5 | fell slowly (18,344 → 14,217) | cut to 1e-4, then 1e-5 |

Interpretation:
- **Training under the official recipe is unstable on SPY.** In seeds 1, 2 and 4 the validation loss rose early, the scheduler cut the learning rate until training effectively stopped, and the best checkpoint comes from the first 10–30 epochs, i.e. barely-trained models. Only seed 3 trained normally.
- **Regime collapse is the norm on SPY: 4 of 5 seeds** (toy data: 1 of 5). The one seed using two regimes (5) still under-covers badly in the crash (0.46).
- **Crash coverage varies enormously with the seed** (0.22–0.78), and averages 0.436 against GARCH's 0.800. Seed 3's near-parity with GARCH (Sections 5–6) was not representative.
- The paper's undisclosed retraining loop (retrain until both regimes are used) would hide both the collapse rate and the seed sensitivity.

Hypothesised causes (tested in Section 9 — shuffling and clipping did not help; the actual cause is dead single-unit ReLUs in the emission networks):
- Mini-batches are taken in date order and never shuffled: each epoch ends on late-2015 data, and validation is on calmer 2016–2018 data.
- No gradient clipping: the official `main.py` defines `clip = 10` but never applies it.
- The validation loss includes sampled regime paths, so it is noisy; with patience 10 the scheduler may cut the learning rate on noise.

Status: rerun with shuffled batches and gradient clipping at 10 (variant `stable`) — see Section 9.

Two questions to keep separate in the thesis:
1. Does the published training recipe work on financial data? (Evidence so far: no — high collapse rate, high seed sensitivity, concealed by the retraining loop.)
2. Does DS³M work on financial data when trained stably? (Pending.)

---

## 9. SPY: stabilised training, and the dead-ReLU diagnosis

Script: `experiments/spy/run_seeds_spy.py` (variant `stable`: shuffled mini-batches, gradient-norm clipping at 10). Results: `experiments/spy/summary_stable_5seeds.json`, `experiments/spy/forecasts_stable_seed{1..5}.csv`, `experiments/spy/ds3m_spy_stable_seed{1..5}_gamma.json`.

| Seed | Regimes used | Overall | 2019 | 2020 | COVID crash | 2021 | 2022 | 2023–2024 | Train loss at epoch 40 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.849 | 0.937 | 0.727 | 0.280 | 0.917 | 0.673 | 0.920 | 1.452 |
| 2 | 1 | 0.931 | 0.976 | 0.838 | 0.400 | 0.976 | 0.825 | 0.986 | 1.118 |
| 3 | 1 | 0.875 | 0.901 | 0.862 | 0.840 | 0.889 | 0.857 | 0.871 | 1.189 |
| 4 | 1 | 0.901 | 0.956 | 0.802 | 0.380 | 0.960 | 0.765 | 0.960 | 1.419 |
| 5 | 1 | 0.929 | 0.968 | 0.838 | 0.400 | 0.964 | 0.833 | 0.986 | 1.417 |
| **Mean ± std** | | 0.897 ± 0.035 | 0.948 ± 0.030 | 0.813 ± 0.053 | **0.460 ± 0.218** | 0.941 ± 0.037 | 0.790 ± 0.074 | 0.945 ± 0.049 | |

Collapsed runs: **5 of 5**. Compared with the official recipe (crash 0.436 ± 0.211, 4 of 5 collapsed), stabilisation brought no improvement.

Learned $\Gamma$ (diagonal): seed 1 (0.563, 0.954); seed 2 (0.946, 0.537); seed 3 (0.9996, 0.501); seed 4 (0.954, 0.572); seed 5 (0.999, 0.502). In every seed one diagonal entry is at or near the 0.5 floor.

**Training-loss benchmark.** Training data are normalised to mean 0, std 1. A model that always predicts a fixed $N(0,1)$ has average loss $\tfrac12\log(2\pi e) \approx 1.419$, the loss of having learned nothing. Train losses are compared at epoch 40, before the KL weight starts rising (from epoch 50 the annealed KL term raises the training loss independently of fit).
- Seeds 4 and 5 sit at 1.417–1.419, flat from epoch 10: no learning.
- Seed 1 at 1.452: no learning.
- Seeds 2 and 3 are well below (1.118, 1.189): real volatility dynamics were learned.

**Dead-ReLU check** (`experiments/spy/check_dead_relu.py`): share of validation inputs for which each regime's emission trunk outputs exactly zero, and the variation of the predicted emission std across inputs.

| Seed | Regime 0 emission | Regime 1 emission |
|---|---|---|
| 1 | dead (100% zero, std constant) | dead |
| 2 | dead | alive, std nearly constant (variation 0.001) |
| 3 | **alive** (std variation 0.29) | dead |
| 4 | dead | dead |
| 5 | dead | dead |

**8 of 10 regime-specific emission networks are dead.**

Mechanism: in the official architecture (deviation 3), each network's hidden width equals its output dimension. For a univariate series ($D = 1$) the emission trunk is $\mathrm{Linear}(12 \to 1) \to \mathrm{ReLU} \to \mathrm{Linear}(1 \to 1) \to \mathrm{ReLU}$, so all information passes through a single ReLU unit. If that unit's input is negative for every data point, it outputs zero everywhere: the emission mean and std become constants, and no gradient reaches the encoder, $z_t$ or the regimes beneath it. Whether this happens depends on the random initialisation and early updates — hence the seed lottery. This is the known "dying ReLU" phenomenon in narrow networks.

Interpretation:
- **Failure to learn** (seeds 1, 4, 5) is caused by dead emission networks in both regimes.
- **Regime collapse is caused by a dead regime, not by the data.** In seed 3, regime 0's emission is alive and regime 1's is dead; a dead regime can only emit one fixed distribution, so the model avoids it and $\Gamma$ pushes it to the 0.5 floor. The SPY "single-regime" result should therefore not be read as evidence that SPY has one regime.
- In seed 2 the prior $\Gamma$ favours regime 0, whose emission is dead, so forecasts have near-constant width despite some learning elsewhere.
- **The only seed with a working emission in its forecasting regime (seed 3) reached 0.84 crash coverage, above GARCH's 0.80.**
- The paper's own univariate experiments (toy, sleep apnea, unemployment) are exposed to the same failure mode; the official retraining loop would conceal it. Toy seed 5's collapse (Section 3) is likely the same mechanism (not yet verified).

This is a precisely characterised failure mode of the published architecture, matching the scope of RQ3 (targeted modification addressing an empirically located failure).

Planned modification (variant `wide`): hidden width 16 in every Gaussian network (transition, emission, posterior), with shuffling and clipping retained. Success criteria: training loss below 1.419 in most seeds, few or no dead networks, and much lower seed-to-seed variation in crash coverage.

---

## 10. Ablation design and pre-specified selection criteria

Recorded while configuration C5 was still training on its first seed, before any of its results were available.

**Purpose.** Replace sequential trial-and-error with a planned ablation in which every change targets a diagnosed mechanism, and every configuration is judged by criteria fixed in advance using training and validation data only.

**Factors**

| Factor | Levels | Mechanism it targets |
|---|---|---|
| Network hidden width | 1 (official, $= D$), 16 | dead single-unit ReLU (Section 9) |
| KL schedule | official (0.01, peaking at ≈0.51), paper (linear 0.01 → 1, then 1), none (constant 1) | posterior copying $y_t$ into $z_t$ when the KL weight is below 1 |
| Training procedure | official (date-ordered batches, no clipping), stabilised (shuffled batches, gradient-norm clipping at 10) | optimisation instability |

**Configurations** (all five seeds, 1–5; toy hyperparameters otherwise; $K = 2$)

| ID | Variant name | Width | KL schedule | Training | Status |
|---|---|---|---|---|---|
| C1 | `official` | 1 | official | official | complete (Section 8) |
| C2 | `stable` | 1 | official | stabilised | complete (Section 9) |
| C3 | `wide` | 16 | official | stabilised | stopped on seed 1 at epoch 20 |
| C4 | `wide_paperkl` | 16 | paper | stabilised | stopped on seed 1 at epoch 30 |
| C5 | `wide_nokl` | 16 | none | stabilised | running |
| C6 | `stable_nokl` | 1 | none | stabilised | planned |

C6 is a control: C5 differs from C2 in both width and KL schedule, so C6 isolates the effect of the KL schedule at the official width (the dead-ReLU mechanism predicts it should not fix the narrow model).

**Observations from the stopped runs** (training and validation behaviour only; no test-period results were computed):
- C3, seed 1: training loss −1.95 (epoch 10) and −2.90 (epoch 20), i.e. near-zero predicted uncertainty on training data; validation loss 115,853 and 124,402 (≈8× earlier runs); learning rate cut to 1e-4 by epoch 20. Interpretation: with the KL weight at 0.01, the posterior (which sees $y_t$ through $A_t$) copies $y_t$ into $z_t$ and the widened emission reads it back. The narrow official networks cannot exploit this, so the official schedule only appears to work because of the bottleneck.
- C4, seed 1: training loss −0.58 (epoch 10, KL weight 0.19), 0.36 (epoch 20), 1.22 (epoch 30); validation loss 65,990 → 51,181 → 39,844; learning rate cut to 1e-4 by epoch 20 and 1e-5 by epoch 30. Copying occurred while the KL weight was low; recovery was too slow before the scheduler froze training.

Background: KL annealing (Bowman et al., 2016) was introduced to stop a latent variable from being ignored. The failure here is the reverse (the latent variable over-used). With a KL weight below 1 the objective is no longer a lower bound on the likelihood.

**Selection criteria — training and validation data only**

Requirements for a configuration to be considered trained reliably:
1. Training loss at epoch 40 below 1.419 (the loss of a fixed $N(0,1)$) in at least 4 of 5 seeds. For the official KL schedule, epoch 40 precedes any rise in the KL weight; for constant-weight configurations the comparison holds throughout.
2. In each seed, the emission network of the regime used for forecasting is not fully dead (`check_dead_relu`).

Among configurations meeting both requirements, ranking is by:
3. Mean validation loss over the five seeds (computed with KL weight 1, i.e. the negative ELBO, comparable across all configurations).

Also reported for every configuration, but not used for selection: regime collapse rate; seed-to-seed standard deviation of the validation loss.

**Test-period metrics** (overall and per-period coverage, Kupiec and Christoffersen tests) are computed for every completed configuration and reported in full, but **are not used to choose a configuration**. Note for transparency: test-period coverage of C1 and C2 had already been viewed before these criteria were written; the criteria above do not use it.

**Stopping rule.** The ablation is closed once C5 and C6 are complete. No configurations are added in response to test-period results. If no configuration meets the requirements, the finding is that DS³M cannot be trained reliably on SPY with these settings, with the documented mechanisms; the calibration comparison then proceeds with GARCH and the best-validated DS³M configuration, reported with that caveat.

---

## Open questions and pending experiments

- **Ablation C5 (`wide_nokl`, running) and C6 (`stable_nokl`, planned)** — Section 10. Selection by the pre-specified criteria; the selected configuration is then compared with GARCH on the test period.
- Verify the dead-ReLU mechanism for toy seed 5.
- **Regime collapse:** is it robust? Variants to test — longer training to convergence; the paper's $\Gamma$ parameterisation without the 0.5 floor; $K = 3$ and $K = 4$ (classical Markov-switching studies find 2–4 regimes for equity returns: Turner, Startz & Nelson 1989; Ang & Bekaert 2002; Guidolin & Timmermann 2005, 2007; Maheu, McCurdy & Song 2012). $K$ to be chosen on validation predictive log-likelihood only.
- **ACI / AgACI** (Gibbs & Candès 2021; Zaffran et al. 2022) on DS³M and GARCH: does online adjustment repair crash-period coverage? Requires saving DS³M's Monte Carlo samples.
- **99% intervals** to test tail shape; requires ≥1,000 Monte Carlo samples for DS³M.
- **Fixed model vs walk-forward refits** (committed experiment; reported whatever the outcome).
- Learned $\Gamma$ is now saved next to the model weights (`*_gamma.json`) for all new runs.
