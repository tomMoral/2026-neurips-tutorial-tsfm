# Slide plan: time series foundation models (long version)

Opening slides of the NeurIPS Paris tutorial. They set up the three parts
(forecasting, Mantis classification, EEG FM), then dive into the design of
forecasting foundation models. This is the long version, meant to be trimmed.

Build: the deck goes in the `presentations` repo (`~/workspace/presentations`,
github.com/tomMoral/presentations, private) as `topics/tsfm-tutorial/`, with
reveal.js and the `tm` layer. Copy the layout of `topics/benchopt/`: one
`parts/*.md` per act, images in `img/`, `refs.bib`. The presentation skill
(`reveal_tm.md`) has the commands, the style and the PDF export. Figures are
matplotlib SVGs made by scripts in `img/` (no TikZ), diagrams use
`<tm-flow>` or SVG.

Conventions for the deck:

- Every frame title is a claim, not a topic.
- Each design decision gets the same layout. Left: what the choice is, with a
  small diagram. Right: one row per model, taken from `notes/model_cards.md`.
  Bottom: a `>` takeaway with what it changes in practice, backed by a
  notebook result when we have one.
- Notebook results are the 2025 Sébastopol runs (hourly, MASE unless stated).
  Each one points to the notebook section that reproduces it.
- Paris thread: t0 (The Forecasting Company), Timestral (Mistral), TabICL and
  Mantis (Inria). Flag them with a small Paris marker in the tables.

Rough size: about 60 slides, 50 to 60 min if presented in full. The trimming
notes at the end say what to cut for a 15 min and a 25 min version.

## Act 0. Opening (2 slides)

1. **Title.** Time series foundation models, from Paris bikes to EEG.
   Speakers, affiliations, repo QR code and Colab badge.
2. **One morning, three tasks, one family of models.** Map of the tutorial:
   forecasting (bike counts), classification (Mantis), EEG foundation model.
   Show the three datasets side by side as small plots.

## Act 1. What a time series is (7 slides)

3. **A time series is a set of values indexed by time, and the order carries
   the information.** Definition `y_t, t = 1..T`, sampling frequency, and the
   counterexample: shuffling rows of a table loses nothing, shuffling a series
   destroys it. Figure: Sébastopol counter, one week, original against
   shuffled.
4. **Time series are everywhere, at every scale of time.** Grid of examples
   with their frequency: EEG (ms), bike counts (hour), electricity (15 min),
   sales (day), CO2 (month), variable stars (irregular). Point: frequency
   ranges over 10 orders of magnitude.
5. **Real series stack several seasonalities on top of a trend.** Sébastopol
   decomposed: daily cycle, weekly cycle, yearly cycle, Covid break in 2020.
   Figure: four panels zooming from one day to five years.
6. **Real series are not stationary.** Level shifts (Covid, new bike lanes),
   changing variance, holidays. Figure: monthly mean and std of Sébastopol,
   2020 to 2025.
7. **Real series are messy: gaps, outliers, and counting artefacts.** Missing
   days of a counter, zero runs from a broken sensor, DST hours (23 or 25 h).
   Figure: the panel of 2025 counters as a heatmap of missing values.
8. **A series rarely comes alone.** Three structures: several channels of one
   system (both directions of a counter, EEG electrodes), covariates that drive
   it (weather, holidays), and many related series (all Paris counters).
   Diagram: one target, its channels, its covariates, its siblings.
9. **Covariates differ by what we know when we forecast.** Static (location),
   past-only (observed weather), known in the future (calendar, holidays,
   weather forecasts). Timeline diagram with the forecast origin. This
   vocabulary is reused in Act 4.

## Act 2. Setups and tasks (9 slides)

10. **The same data supports many tasks.** Overview table: forecasting,
    classification, regression, anomaly detection, imputation, clustering,
    with input, output and an example for each. Highlight the two covered
    later: forecasting (part 1), classification (parts 2 and 3).
11. **Forecasting predicts the future from the past, up to a horizon.**
    Context, origin, horizon, lead time. Figure: one forecast on Sébastopol
    with its context and horizon shaded.
12. **A point forecast hides the uncertainty, so we predict distributions.**
    Median, quantiles, prediction interval, sample paths. Figure: fan chart
    with the 10, 50 and 90% quantiles.
13. **Forecasting setups multiply: univariate, multivariate, with covariates,
    across many series.** 2×2 grid of the setups with the bike example of each.
    Notebook sections 4, 7, 8 to 10.
14. **Local models fit each series, global models learn across series.**
    ARIMA and ETS against one model trained on a panel. A foundation model is
    a global model trained on a very wide panel. Sets up Act 3.
15. **Classification assigns a label to a whole series.** Examples: activity
    from accelerometers, sleep stage from EEG, star type from light curves.
    Pointer to parts 2 and 3.
16. **Evaluation must respect time.** Random splits leak the future.
    Rolling-origin evaluation with several origins. Figure: the 2025 origins
    of the notebook on a timeline.
17. **Scaled metrics make series comparable.** MASE (error over the seasonal
    naive error), WQL and CRPS for quantiles, coverage of the interval.
    One formula per metric, one line on what it rewards.
18. **Always compare with the seasonal naive baseline.** On Sébastopol the
    seasonal naive gets MASE 1.155 at a 2 week horizon, and the best foundation
    model 0.926. A gain of 20%, not an order of magnitude (notebook §4).

## Act 3. A quick recap on foundation models (9 slides)

19. **A foundation model is pretrained once on broad data, then reused without
    training.** Pretrain, then zero-shot, fine-tune or in-context learning.
    Diagram: one model, many downstream tasks.
20. **Language models made the recipe famous: next-token prediction at scale.**
    Tokens, a transformer, a huge corpus, emergent in-context learning.
    One slide, enough to set up the analogy.
21. **Time series are harder to pool than text.** No shared vocabulary, scales
    from 0.01 to 10^9, frequencies from ms to years, and series are short.
    So each model must decide how to turn numbers into tokens. This motivates
    Act 4.
22. **Zero-shot forecasting: the context is the only training set.** The model
    reads the history and outputs the forecast in one call. No fit, no
    hyperparameters. Code snippet from the notebook: `model.predict(context,
    horizon)`.
23. **Real data is scarce, so synthetic data does much of the work.** Real
    corpora (Chronos corpus, GIFT-Eval pretrain, Wikipedia pageviews, Google
    Trends) against synthetic generators (KernelSynth Gaussian processes,
    TSMixup, causal SCMs for covariates). Timestral stage 1 is fully synthetic.
    TabICL never saw a time series.
24. **Tabular foundation models also forecast, by turning time into
    features.** TabPFN and TabICL learn from synthetic tables. Forecasting
    becomes regression on calendar features. A different road to the same goal.
25. **Three years of time series foundation models.** Timeline from 2023 to
    2026: TimeGPT, Lag-Llama, Moirai, Chronos, TimesFM, Moment, Toto, TiRex,
    Chronos-2, TimesFM 2.5 and 3, TiRex-2, t0, Timestral. Mark the Paris ones.
26. **Benchmarks are the scoreboard: GIFT-Eval, fev-bench, TIME.** What each
    covers (datasets, domains, frequencies, metrics). Leaderboards from the
    model cards: four models claim first place, depending on the date and the
    metric.
27. **Leaderboards need care: leakage, overlap and author-run results.**
    Training corpora overlap test sets, decontaminated checkpoints exist
    (TiRex-2, t0-beta), and most numbers are reported by the authors.
    Takeaway: test on your own data, as we do with the bikes.

## Act 4. A tour of design decisions (about 28 slides)

Opening and closing frames, then one block per decision. Each block has one
"what are the options" slide and one "what the models chose" slide. A third
slide is added when the notebook shows the effect.

28. **Foundation models differ less by size than by a handful of design
    choices.** The list of decisions in the order of the data flow: scaling,
    tokens, backbone, mixing across variates, decoding the horizon, output,
    context, training data, inference tricks. Diagram: a pipeline from raw
    series to quantiles, each stage labeled with its decision.

### 4.1 Scaling the input

29. **Every model rescales the context, because series have arbitrary
    units.** Options: mean scaling, standardization, arcsinh on top, causal
    per-patch statistics, reversible normalization (RevIN).
30. **The field converged on standardization plus arcsinh.** Table: Chronos
    (mean scaling), Bolt (standardization), Chronos-2, TiRex-2, t0, Timestral
    (standardization then arcsinh), TimesFM 3 (iterative RevIN and optional
    detrending), TabICL (inside the regressor). Takeaway: arcsinh tames
    spikes, such as event days for bike counts.

### 4.2 From numbers to tokens

31. **Three ways to make tokens: quantized values, patches, or table rows.**
    Diagram of each: Chronos bins values into 4096 tokens like words, patch
    models embed 16 or 32 steps at once, TabICL makes one row per time step.
32. **Patches won, and most models now use 32 steps.** Table of patch in and
    out sizes: Chronos 1/1, Bolt 16/64, Chronos-2 16/16, TimesFM 3 32/64,
    TiRex-2 32/32, Timestral 32/32, t0 32/32. Patches make long contexts
    affordable: 8192 steps are 256 tokens.
33. **The best patch size depends on the backbone and the horizon.** Our
    patch-size study (arXiv 2611.02750): encoder-only models prefer small
    patches, decoder-only models show a U-shape whose minimum grows with the
    horizon. Figure from the paper.
34. **Resolution is a patching decision you make on the data.** Notebook §6:
    forecasting 4 h totals from hourly, 2 h or 4 h data. MAE 246, 237 and 253.
    Takeaway: aggregating the input is not free, and neither is keeping
    every step.

### 4.3 The backbone

35. **Four backbones: encoder-decoder, encoder-only, decoder-only,
    recurrent.** One diagram each, with the attention mask.
36. **Every family is still in the race.** Table: Chronos and Bolt
    (encoder-decoder T5), Chronos-2 (encoder-only), TimesFM 3, t0 and
    Timestral (decoder-style, causal over time), TiRex-2 (xLSTM), TabICL
    (tabular in-context learner). Sizes from 5M to 330M. Takeaway: on our
    bikes the ranking is not the size ranking (Timestral 209M ahead of
    TimesFM 3 330M, notebook §4).
37. **A recurrent model scales linearly with the context.** TiRex-2 carries
    a state instead of attending to every patch. Cost against context length
    for attention and recurrence, a schematic curve.

### 4.4 Mixing variates and covariates

38. **Channel independence was the default, mixing is now the norm.** Options:
    univariate only, linear regression on covariates outside the model
    (TimesFM 2.5 XReg), attention across variates inside the model.
39. **Models alternate attention over time and attention across variates.**
    Diagram of the two axes. Table: Chronos-2 group attention, TimesFM 3
    variate attention (at most 32 variates), TiRex-2 variate mixer, t0 one
    variate layer every three, Timestral one every four, TabICL as columns.
    Asymmetric masks: covariates never read the targets.
40. **Known-future covariates help where the series alone cannot know.**
    Notebook §8, weeks with a public holiday at 4 h resolution: t0 1.02 to
    0.88, TabICL 0.98 to 0.81. Figure: Easter Monday 2025, forecast with and
    without the holiday flag.
41. **Past weather helps, and a placebo checks that the gain is real.**
    Notebook §9: Timestral 0.866 (none), 0.854 (calendar), 0.812 (+ past
    weather), 0.841 (+ shuffled weather). t0 shows the same pattern. TiRex-2
    gives the same forecast with or without covariates.
42. **Learning across many series is a mixing decision too.** Chronos-2
    `cross_learning` puts several series in one group. Notebook §10 on 118
    counters: Chronos-2 0.962 alone, 0.941 with cross-learning, t0 0.919
    without it. Grouping correlated counters brought no gain.

### 4.5 Decoding the horizon

43. **Two ways to reach the horizon: step by step, or all at once.**
    Autoregressive (feed predictions back) against direct (predict every step
    in one pass). Classic AR(p) recursive against direct multi-output, the same
    dichotomy as in classical forecasting.
44. **Direct decoding took over, with masked future patches or forecast
    tokens.** Table: Chronos (AR sampling over tokens), TimesFM 2.5 (AR over
    128-step patches), Bolt (direct 64), Chronos-2 and t0 (direct up to 1024),
    TimesFM 3 and TiRex (masked future patches), Timestral (decoupled forecast
    tokens), TabICL (direct on features).
45. **Every model has a maximum horizon, and beyond it they roll out.**
    Max horizon per model: Bolt 64, TiRex-2 320 (truncated with a warning),
    Chronos-2, t0 and Timestral 1024. Rollout feeds quantile paths back and
    reduces them.
46. **Rolling out costs accuracy and, above all, calibration.** Notebook §5:
    t0 direct against 8 rollout steps. MASE 0.890 to 0.932, coverage of the 80%
    interval 0.685 to 0.414. Figure: the two fan charts.

### 4.6 The output

47. **Models output samples or quantiles, and the grid matters.** Chronos
    samples 20 paths, Bolt, TimesFM 3 and TiRex-2 output 9 quantiles,
    Chronos-2, t0 and Timestral 21. Quantile crossing and how each fixes it
    (non-crossing by construction for Timestral).
48. **Samples give joint paths, quantiles give marginals.** What you can and
    cannot compute from each: totals over the horizon, peak timing. Link to
    the resolution exercise, where 4 h totals come from hourly quantiles.

### 4.7 How much history

49. **Context windows grew from 512 to 32768 steps.** Bar chart of max context
    per model: Chronos 512, Bolt 2048, TiRex-2 2048 (released config),
    Chronos-2 8192, TimesFM 3 15360, Timestral 32768, t0 no hard cap (trained
    on 8192), TabICL 4096 rows by default. One year of hourly data is 8760
    steps.
50. **Several models drop the extra history without telling you.** Chronos-2
    keeps 8192 steps, TiRex and TiRex-2 2048. Takeaway: check the
    effective context before concluding that history does not help.
51. **Seeing more than a year pays off for the models that can.** Notebook §11,
    MASE from 2048 to 16384 steps: TimesFM 3 0.93 to 0.85 (0.79 with
    calendar), Timestral 0.85 to 0.75, TabICL with calendar 0.95 to 0.78.
    Chronos-2, TiRex-2 and t0 stay flat. Figure: MASE against context length
    with a line at one year.

### 4.8 Training data and recipe

52. **Each model is a bet on a data mix.** Table from the cards: real versus
    synthetic, corpus size when stated, context length at training, number
    of stages (pretrain then long-context post-training for TiRex-2,
    Timestral, Chronos-2). Many cards do not state the corpus size.
53. **Synthetic multivariate data teaches models to use covariates.** All of
    Chronos-2's multivariate data is synthetic. TiRex-2 and t0 couple series
    with causal models. This is how covariate handling is learned without
    labeled real data.

### 4.9 Inference tricks

54. **Small tricks at inference change the numbers.** Symmetric averaging
    with the negated series (Timestral, t0 option), clipping at zero for
    counts, differencing test-time augmentation (TiRex-2), trimming leading
    missing values (t0). Takeaway: compare models with the same options.
55. **Cost matters as much as accuracy for deployment.** Parameters, latency
    from the cards (Timestral 5M at 17 ms on CPU), our notebook timings on GPU
    and CPU. Licenses: TimesFM 3 weights are non-commercial.

### Closing the tour

56. **One table to place every model.** The overview table of
    `notes/model_cards.md`, one column per decision of the tour, with the
    Paris models flagged.
57. **What the choices buy on our bikes.** Summary of the notebook results by
    decision: decoding (§5), resolution (§6), covariates (§8 and §9),
    cross-learning (§10), context (§11). One line and one number each.

## Act 5. Taking stock (3 slides)

58. **Foundation models are strong baselines, not oracles.** About 20% better
    than seasonal naive on our counter. Gains depend on covariates and history
    more than on the model. When a local model or a tabular model is enough.
59. **Open problems.** Evaluation without leakage, long horizons beyond the
    native window, irregular sampling, joint probabilistic forecasts, using
    covariates the model never saw in training.
60. **Now, your turn.** Open the notebook. QR code, Colab badge, CPU fast mode
    and GPU runtime. Handoff to part 1 hands-on, then to Mantis.

## Figures to make

| Slide | Figure | Source |
|---|---|---|
| 3 | week of Sébastopol, original and shuffled | `data/sebastopol.parquet` |
| 5 | zoom from a day to five years | same |
| 6 | monthly mean and std | same |
| 7 | missing values of the 2025 counters | `data/bike-counters-2025.parquet` |
| 11, 12 | forecast with context and fan chart | notebook §2 |
| 16 | rolling origins on a timeline | notebook §4 |
| 25 | model timeline | model cards |
| 28 | pipeline with one stage per decision | SVG or `<tm-flow>` |
| 31, 35, 39, 43 | tokens, backbones, mixing axes, decoding | SVG or `<tm-flow>` |
| 33 | patch size against horizon | arXiv 2611.02750 |
| 40 | Easter Monday with and without holidays | notebook §8 |
| 46 | direct against rollout fan charts | notebook §5 |
| 49 | max context per model | model cards |
| 51 | MASE against context length | notebook §11 |

## Trimming notes

- **25 min version**: keep 1 to 3, 8, 9, 11 to 13, 16 to 18, 19, 21, 23, 25,
  26, 28, then the "what the models chose" slide of each decision plus 40, 46
  and 51, then 56 to 60.
- **15 min version**: keep 1, 2, 3, 8, 11, 12, 16, 19, 21, 28, 32, 39, 44, 49,
  51, 56, 60. The notebook then carries the evidence.
- Act 1 and Act 2 can shrink to 4 slides if the other parts of the tutorial
  introduce time series themselves.
