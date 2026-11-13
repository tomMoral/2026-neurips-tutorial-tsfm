# NeurIPS Paris tutorial: TSFM for forecasting (≈60 min)

Part 1 of 3 (forecasting, then Mantis classification, then EEG FM).
Format: mostly hands-on, with short slide interludes. Three notebooks, one
shared helper module, and the same `ml-time-series` env (Colab badges as the
fallback).

## Storyline

One running dataset, made progressively harder: **Paris bike counters**
(already in `data/bike-counter-data.parquet` and `02c`). It covers every axis
we want with no context switch:

- univariate: one counter
- multivariate: several channels at one site (both directions, or counts and weather)
- covariates: weather and holidays, known in the future or only in the past
- multiple sites: the whole panel of counters (global model, cross-learning)

CO₂ (`02b`) is a short univariate warm-up because its trend and seasonality are
clean, so AR versus direct errors are easy to see.

## Timeline

| Time | Block | Format |
|---|---|---|
| 0–5 | What a TSFM is, the landscape | slides |
| 5–15 | `01-forecasting` §1–4: zero-shot, one API, many architectures, evaluation | hands-on |
| 15–27 | §5–6: AR vs direct, patch size | hands-on + paper |
| 27–42 | §7–10: multivariate, calendar, weather, many sites | hands-on |
| 42–52 | Evaluating TSFMs: benchmark_tsfm (benchopt) | demo + results |
| 52–58 | Open problems | slides |
| 58–60 | Handoff to classification (Mantis) | slide |

### 0–5 · Landscape (slides)

One slide on the recipe: pretraining on a large corpus, then zero-shot
forecasting with a probabilistic output (quantiles or samples).
One map slide that places models on two axes, which the rest of the session
then reuses:

| Model | Backbone | Input | Output / decoding | Multivar / covariates |
|---|---|---|---|---|
| Chronos (T5) | encoder-decoder | quantized tokens | AR sampling over tokens | no |
| Chronos-Bolt | encoder-decoder | patches | direct multi-quantile | no |
| Chronos-2 | encoder-only, group attention | patches | direct | yes (both) |
| TimesFM | decoder-only | patches (32) | AR over long output patches (128) | covariates via regression |
| TiRex | xLSTM (recurrent) | patches | AR-style with state | no |
| Toto | decoder-only | patches | AR | multivariate |
| Moment | encoder-only (masked) | patches | reconstruction head | no |
| TabPFN-TS / TabICL | tabular FM | lag and calendar features | direct regression | covariates as columns |
| t0 (TFC) | API | | | |

The point to make: "foundation model" covers very different inductive biases.
Tokenization, patching, decoding strategy and how channels are mixed are the
real design choices.

### 5–15 · NB1 — zero-shot univariate

- Thin wrapper `forecast(model, context, horizon) -> quantiles` around each
  library. The audience compares models, not APIs.
- Run on one bike counter: Seasonal-naive, Chronos-Bolt, TimesFM, TiRex,
  TabPFN-TS. Plot the median and the 10–90% band.
- Metrics: MASE and WQL against seasonal naive. This reuses the evaluation
  lessons of `01b` (test on the future, rolling origin).
- Exercise: vary the context length (1 day, 1 week, 1 month). Which model
  uses the long context?

Reuse: the TSFM section at the end of `02c` already has TiRex, Chronos,
TimesFM and t0. Move it here and generalize it.

### 15–27 · NB2 — how the forecast is produced: AR vs direct

- Recall from `02a`/`02c`: the recursive AR(p) model versus direct
  multi-output regression. Same dichotomy inside TSFMs.
- Experiment: horizon sweep (1 h → 1 week) with Chronos (token AR) against
  Chronos-Bolt (direct), and TimesFM (long output patches, a hybrid).
  Plot error against horizon. AR error compounds, direct loses sharpness.
- Patch size (slides, our paper arXiv:2611.02750). Encoder-only models prefer
  small patches. Decoder-only models show a U-shape, and their best patch size
  grows with the horizon. Context helps encoders up to about 2048 points.
  Decoders sit on the cheap side of the compute and latency Pareto front.
- Exercise: measure CPU latency per model on the same context. Put it next to
  accuracy to reproduce the Pareto view in miniature.

### 27–42 · NB3 — beyond one series

1. **Multivariate**: forecast several channels jointly with Chronos-2 (group
   attention) or Toto, against independent univariate forecasts. Is there a
   gain? Often small, which is a good honest message.
2. **Covariates**: add weather and holidays (features from `02c` solutions).
   - Known future covariates (calendar, holidays) versus past-only (weather
     unless forecast).
   - Chronos-2 with native covariates, TabPFN-TS with covariates as columns,
     TimesFM plus covariate regression.
   - Exercise: shuffle the covariate as a leakage and sanity check.
3. **Multiple sites**: the full counter panel.
   - Zero-shot per site, against fine-tuning or cross-learning across sites.
   - Map of per-site error (folium is already in the env) to show where models
     fail: new stations, closures, regime changes.

### 42–52 · Evaluation: benchmark_tsfm

- What benchopt brings: one Objective, one Dataset, one Solver per model, and
  a run that is reproducible from a config. Show
  `benchopt run benchmark_tsfm -s chronos2 -d gifteval --config ...`.
- Suites: GIFT-Eval, fev-bench, Monash, plus our own (Enedis).
- Results: the GIFT-Eval table (Chronos2 best on WQL/CRPS, TabPFN-v2 best on
  MASE/sMAPE). The ranking depends on the metric.
- Pitfalls the benchmark makes explicit:
  - pretraining and test overlap (`benchmark_utils/leakage.py`),
  - MASE undefined on constant series (16 of 55 GIFT-Eval datasets),
  - aggregation choice (median, geometric mean, Elo, critical difference plot).
- Exercise (take-home): add your own model as a Solver in about 30 lines.

### 52–58 · Open problems

- **Design**: patch size and architecture depend on horizon and budget. There
  is no universal tokenization.
- **Evaluation**: leakage between pretraining corpora and benchmarks, metric
  disagreement, rapid saturation of public leaderboards.
- **Multivariate and covariates**: gains are inconsistent. When does channel
  mixing help?
- **Panels and hierarchies**: cross-series learning, coherence across levels.
- **Distribution shift**: regime changes and rare events, calibration of
  intervals.
- **Cost**: is a 200M model worth it over TabPFN-TS or a tuned GBM with lags?

### 58–60 · Handoff

Bridge to Mantis: same encoders, but the target is a label instead of the
future. Moment and Chronos-2 embeddings plus a linear probe make the link.

## Logistics

- Precompute and cache all forecasts in the notebooks (`.parquet` next to the
  data), so a slow wifi or CPU-only laptop still shows every plot.
- Small checkpoints only (Chronos-Bolt small, TiRex, TimesFM 2.5 200M) on CPU.
- Pin versions in `environment.yml`. Run the notebooks in CI, as the repo does.
- Coordinate with the other two parts on a shared env and a shared wrapper.

## Material to build

- [x] Flat layout: `01-forecasting.ipynb`, `tsfm_utils.py`, `data/`, `solutions/` at the root, old chapters in a local, untracked `inspiration/`
- [x] `01-forecasting`: the three draft notebooks merged in one, the context-length exercise dropped
- [x] Calendar with public and school holidays (Paris, zone C), at 1 h and 4 h resolution
- [ ] Cut the notebook to the time slot once the format (notebook-based slides or companion deck) is chosen
- [x] Trim `environment.yml` to the packages of the tutorial
- [ ] Slides: landscape map, patch-size results, benchmark results, open problems
- [ ] benchmark_tsfm: re-run GIFT-Eval with the current config, export plots
- [x] Colab: the badge and setup cell point to `tomMoral/2026-neurips-tutorial-tsfm`

## Findings from the first runs (16 origins, Sébastopol S-N)

- t0 and Chronos-2 lead on MASE and WQL. TabICL needs the calendar to compete.
- Rolling out t0 keeps the median (MASE 0.91 → 0.89) but narrows the intervals (WQL 0.163 → 0.174).
- Chronos (T5, 20 samples) covers only about 30% with its 80% interval.
- Resampling to 2 h beats 1 h and 4 h for 4-hourly totals: a U-shape, as in the patch-size paper.
- Joint forecast of the two directions: no gain.
- Calendar on 4 weeks with a weekday holiday (MASE at each resolution):
  - hourly, 2048 h of context: TabICL 1.65 → 1.43, Chronos-2 1.44 → 1.35, t0 1.40 → 1.39.
  - 4 h, 2048 steps (11 months, every holiday since Sept. 2020 in context): t0 1.61 → 1.29 (public) → 1.20 (+ school). Chronos-2 does not gain.
  - School holidays hurt on the origins every two weeks with a short context (t0 at 4 h, 512 steps: 0.93 → 1.00). Their only example in context is spring 2021, during the third lockdown.
- Observed future weather: MASE 0.92 → 0.76 (leakage upper bound). Past weather: 0.92 → 0.91. Placebo: no gain.
- All counters: cross-learning brings almost nothing (0.832 → 0.830 with 168 h of context).
