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

Caveats: single seed; training not converged (validation loss still falling at epoch 100); one value of $K$; 200 Monte Carlo samples per forecast.

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
- **The coverage pattern is shared by all three models**: over-coverage in calm years, under-coverage in the crash and 2022. It is not specific to DS³M.
- DS³M vs GARCH in the crash: 39 vs 40 covered days out of 50, a one-day difference.
- **Heavy tails do not explain the crash under-coverage at the 90% level.** GARCH-t ($\nu \approx 7.4$) and GARCH-N give near-identical coverage; at the 5%/95% points a unit-variance $t_{7.4}$ and a Gaussian nearly coincide. Tail shape should matter at 99%, which has not yet been evaluated.
- GARCH widens its intervals more in the crash than DS³M, with almost no coverage gain.
- The collapsed DS³M behaves like a continuous-volatility model and performs like GARCH.

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
- DS³M results are from one seed.

---

## Open questions and pending experiments

- **DS³M on SPY, 5 seeds** (running): collapse rate on SPY vs 1/5 on the toy data; per-period coverage as mean ± std.
- **Regime collapse:** is it robust? Variants to test — longer training to convergence; the paper's $\Gamma$ parameterisation without the 0.5 floor; $K = 3$ and $K = 4$ (classical Markov-switching studies find 2–4 regimes for equity returns: Turner, Startz & Nelson 1989; Ang & Bekaert 2002; Guidolin & Timmermann 2005, 2007; Maheu, McCurdy & Song 2012). $K$ to be chosen on validation predictive log-likelihood only.
- **ACI / AgACI** (Gibbs & Candès 2021; Zaffran et al. 2022) on DS³M and GARCH: does online adjustment repair crash-period coverage? Requires saving DS³M's Monte Carlo samples.
- **99% intervals** to test tail shape; requires ≥1,000 Monte Carlo samples for DS³M.
- **Fixed model vs walk-forward refits** (committed experiment; reported whatever the outcome).
- Save learned $\Gamma$ alongside model weights.
