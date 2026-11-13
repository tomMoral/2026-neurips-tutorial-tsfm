# Time series foundation models: a hands-on tutorial

Hands-on material for the NeurIPS Paris tutorial on time series foundation
models (TSFMs). It has three parts: forecasting, classification with Mantis,
and EEG foundation models.

## Layout

The repository is flat, with one notebook per part at its root.

| Notebook | Topic |
|---|---|
| [01-forecasting](01-forecasting.ipynb) | Zero-shot forecasting with Chronos, Chronos-2, TimesFM 3, TiRex-2, Timestral, t0 and TabICL on the Paris bike counters: evaluation, autoregressive or direct decoding, patch size, multivariate, covariates, many sites and long contexts |

- `tsfm_utils.py`: the data loaders, the metrics and one wrapper per model,
  all with the same `predict(context, horizon)` method.
- `data/`: the bike counts and the Paris weather (Open-Meteo, CC BY 4.0).
- `solutions/`: the exercise solutions.
- `notes/`: the plan of the session and the model cards.

The notebooks are meant for a GPU runtime on Colab. They also run on CPU, more
slowly.

## Setup

```bash
mamba env create -f environment.yml
conda activate ml-time-series
```

Each notebook also opens on Colab from the badge at its top.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The `Notebooks` workflow checks the
notebook conventions and runs every notebook with its exercises replaced by
their solutions.
