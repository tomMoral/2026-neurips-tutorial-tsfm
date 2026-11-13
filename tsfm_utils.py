"""Data, models and metrics shared by the TSFM forecasting notebooks.

Every model has the same interface::

    quantiles = model.predict(context, horizon)

Models load their weights at their first forecast. ``use_cache`` stores the
forecasts on disk, so that a second run of a notebook reuses them.

- ``context``: array ``(n_series, n_timesteps)``, NaN for a missing value.
  Models that forecast several variates jointly also take
  ``(n_series, n_variates, n_timesteps)``.
- ``horizon``: number of future timesteps to forecast.
- ``quantiles``: array ``(n_series, [n_variates,] horizon, len(QUANTILES))``.

Models that use covariates also take ``past_covariates``, shaped
``(n_series, n_covariates, n_timesteps)``, known only over the context, and
``future_covariates``, shaped ``(n_series, n_covariates, n_timesteps + horizon)``,
known over the context and the horizon.
"""
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN = QUANTILES.index(0.5)
WEEK = 24 * 7
BIKE_DATA = "data/sebastopol.parquet"  # both directions of one counter, 2020 to 2025
SITES_DATA = "data/bike-counters-2025.parquet"  # every counter, 2024 and 2025


# --- Data -------------------------------------------------------------------

def load_bike_counts(path=BIKE_DATA):
    """Hourly bike counts in UTC, one column per counter, NaN for a missing hour."""
    df = pd.read_parquet(path)
    counts = df.pivot_table(
        index="date", columns="counter_name", values="bike_count", observed=True
    )
    return counts.asfreq("h")


def load_counter_locations(path=SITES_DATA):
    df = pd.read_parquet(path)
    return df.drop_duplicates("counter_name").set_index("counter_name")[["latitude", "longitude"]]


def load_weather():
    """Hourly weather in Paris in UTC (Open-Meteo, CC BY 4.0)."""
    return pd.read_parquet("data/paris-weather.parquet").set_index("date").asfreq("h")


def french_holidays(start, end):
    from pandas.tseries.holiday import AbstractHolidayCalendar, Easter, EasterMonday, Holiday
    from pandas.tseries.offsets import Day

    class FrenchHolidays(AbstractHolidayCalendar):
        rules = [
            Holiday("New Year", month=1, day=1),
            EasterMonday,
            Holiday("Labour Day", month=5, day=1),
            Holiday("Victory Day", month=5, day=8),
            Holiday("Ascension", month=1, day=1, offset=[Easter(), Day(39)]),
            Holiday("Whit Monday", month=1, day=1, offset=[Easter(), Day(50)]),
            Holiday("Bastille Day", month=7, day=14),
            Holiday("Assumption", month=8, day=15),
            Holiday("All Saints", month=11, day=1),
            Holiday("Armistice", month=11, day=11),
            Holiday("Christmas", month=12, day=25),
        ]

    return FrenchHolidays().holidays(start, end)


# School holidays in Paris (zone C), first and last day off, from data.education.gouv.fr
SCHOOL_HOLIDAYS = [
    ("2019-07-06", "2019-09-01"), ("2019-10-19", "2019-11-03"), ("2019-12-21", "2020-01-05"),
    ("2020-02-08", "2020-02-23"), ("2020-04-04", "2020-04-19"), ("2020-05-20", "2020-05-24"),
    ("2020-07-04", "2020-08-31"), ("2020-10-17", "2020-11-01"), ("2020-12-19", "2021-01-03"),
    ("2021-02-13", "2021-02-28"), ("2021-04-10", "2021-04-25"), ("2021-05-13", "2021-05-16"),
    ("2021-07-06", "2021-09-01"), ("2021-10-23", "2021-11-07"), ("2021-12-18", "2022-01-02"),
    ("2022-02-19", "2022-03-06"), ("2022-04-23", "2022-05-08"), ("2022-05-26", "2022-05-29"),
    ("2022-07-07", "2022-08-31"), ("2022-10-22", "2022-11-06"), ("2022-12-17", "2023-01-02"),
    ("2023-02-18", "2023-03-05"), ("2023-04-22", "2023-05-08"), ("2023-05-18", "2023-05-21"),
    ("2023-07-08", "2023-09-03"), ("2023-10-21", "2023-11-05"), ("2023-12-23", "2024-01-07"),
    ("2024-02-10", "2024-02-25"), ("2024-04-06", "2024-04-21"), ("2024-05-09", "2024-05-12"),
    ("2024-07-06", "2024-09-01"), ("2024-10-19", "2024-11-03"), ("2024-12-21", "2025-01-05"),
    ("2025-02-15", "2025-03-02"), ("2025-04-12", "2025-04-27"), ("2025-05-29", "2025-06-01"),
    ("2025-07-05", "2025-08-31"), ("2025-10-18", "2025-11-02"), ("2025-12-20", "2026-01-04"),
]


def school_holidays(index):
    days = index.normalize()
    return np.any([(days >= start) & (days <= end) for start, end in SCHOOL_HOLIDAYS], axis=0)


def calendar_features(index):
    """Calendar covariates known in advance, in Paris time: hour, day of week, public and school holidays.

    ``index`` is in UTC, like the counts.
    """
    local = index.tz_localize("UTC").tz_convert("Europe/Paris").tz_localize(None)
    holidays = french_holidays(local.min().normalize(), local.max())
    return pd.DataFrame({
        "hour": local.hour,
        "weekday": local.dayofweek,
        "holiday": local.normalize().isin(holidays),
        "school holiday": school_holidays(local),
    }, index=index).astype("float32")


def make_windows(data, origins, context_length, horizon):
    """Cut a (context, future) pair before and after each forecast origin.

    ``data`` is a DataFrame indexed by time. Returns the contexts and futures
    as arrays ``(n_origins, n_columns, n_timesteps)``.
    """
    positions = data.index.get_indexer(origins)
    values = data.to_numpy(dtype="float32").T
    context = np.stack([values[:, p - context_length:p] for p in positions])
    future = np.stack([values[:, p:p + horizon] for p in positions])
    return context, future


# --- Metrics ----------------------------------------------------------------

def mase(y_true, quantiles, context, season=WEEK):
    """Mean absolute scaled error of the median, one value per series.

    The error is divided by the error of the seasonal naive forecast
    on the context: below 1, the model beats repeating last week.
    It is undefined (NaN) for a constant context, such as a broken counter.
    """
    scale = np.nanmean(np.abs(context[..., season:] - context[..., :-season]), axis=-1)
    scale = np.where(scale > 0, scale, np.nan)
    error = np.nanmean(np.abs(y_true - quantiles[..., MEDIAN]), axis=-1)
    return error / scale


def wql(y_true, quantiles):
    """Weighted quantile loss, one value per series: the probabilistic counterpart of the WAPE."""
    q = np.array(QUANTILES)
    diff = y_true[..., None] - quantiles
    pinball = np.maximum(q * diff, (q - 1) * diff)
    return 2 * np.nanmean(np.nansum(pinball, axis=-2), axis=-1) / np.nansum(np.abs(y_true), axis=-1)


def coverage(y_true, quantiles):
    """Fraction of true values inside the 10-90% interval: 0.8 when calibrated."""
    return np.nanmean((y_true >= quantiles[..., 0]) & (y_true <= quantiles[..., -1]))


def score(y_true, quantiles, context, season=WEEK):
    """MASE, WQL and coverage of the 80% interval, averaged over the series."""
    return {
        "MASE": np.nanmean(mase(y_true, quantiles, context, season)),
        "WQL": np.nanmean(wql(y_true, quantiles)),
        "coverage": coverage(y_true, quantiles),
    }


# --- Plots ------------------------------------------------------------------

def plot_forecast(index, context, future, quantiles, ax=None, label="forecast", n_context=WEEK):
    """Plot the end of the context, the truth and the 10-90% forecast band."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    horizon = len(future)
    past, ahead = index[-horizon - n_context:-horizon], index[-horizon:]
    ax.plot(past, context[-n_context:], color="k", lw=1)
    ax.plot(ahead, future, color="k", lw=1, ls=":", label="truth")
    line, = ax.plot(ahead, quantiles[:, MEDIAN], label=label)
    ax.fill_between(ahead, quantiles[:, 0], quantiles[:, -1], color=line.get_color(), alpha=0.25)
    ax.legend(loc="upper left")
    return ax


# --- Models -----------------------------------------------------------------

_memory = joblib.Memory(None, verbose=0)


def use_cache(location=".cache"):
    """Store the forecasts on disk, so that the next run reuses them.

    Delete the folder to recompute them, or pass ``location=None`` to turn the cache off.
    """
    global _memory
    _memory = joblib.Memory(location, verbose=0)


def _forecast(name, settings, model, context, horizon, kwargs):
    if not model.loaded:
        model.load()
        model.loaded = True
    return model.forecast(context, horizon, **kwargs)


def predict_in_chunks(model, context, horizon, size=8, **kwargs):
    """Forecast ``size`` series at a time, to bound the memory of long contexts.

    The arguments with one entry per series, such as covariates, are cut the same way.
    """
    quantiles = []
    for start in range(0, len(context), size):
        part = {k: v if isinstance(v, str) else v[start:start + size] for k, v in kwargs.items()}
        quantiles.append(model.predict(context[start:start + size], horizon, **part))
    return np.concatenate(quantiles)


class Forecaster:
    """The common interface. ``predict`` loads the weights at the first forecast
    and goes through the cache of ``use_cache``.

    Each model defines ``load``, which loads its weights, and ``forecast``, the
    native call of its library.
    """

    name = ""
    loaded = False

    def load(self):
        pass

    def predict(self, context, horizon, **kwargs):
        settings = {k: v for k, v in vars(self).items() if k != "loaded" and isinstance(v, (int, float, str))}
        return _memory.cache(_forecast, ignore=["model"])(self.name, settings, self, context, horizon, kwargs)


class SeasonalNaive(Forecaster):
    """Repeat the last week. No uncertainty: every quantile is the same."""

    name = "Seasonal naive"

    def forecast(self, context, horizon):
        last_week = context[..., -WEEK:]
        forecast = np.concatenate([last_week] * (horizon // WEEK + 1), axis=-1)[..., :horizon]
        return np.repeat(forecast[..., None], len(QUANTILES), axis=-1)


class Chronos(Forecaster):
    """Chronos (Amazon, 2024): T5 encoder-decoder over quantized values.

    Each value is scaled and binned into one of 4096 tokens, then the decoder
    samples the future one token at a time, like a language model.
    """

    name = "Chronos"

    def __init__(self, checkpoint="amazon/chronos-t5-small"):
        self.checkpoint = checkpoint

    def load(self):
        import chronos.chronos
        from chronos import BaseChronosPipeline

        # Chronos warns at every call with a horizon above 64 steps
        logging.getLogger(chronos.chronos.__file__).setLevel(logging.ERROR)
        self.pipeline = BaseChronosPipeline.from_pretrained(self.checkpoint, device_map=DEVICE)

    def forecast(self, context, horizon, chunk_size=4):
        # Small chunks of series bound the memory of the sampled paths
        quantiles = [
            self.pipeline.predict_quantiles(
                torch.tensor(chunk), prediction_length=horizon, quantile_levels=QUANTILES
            )[0]
            for chunk in np.split(context, range(chunk_size, len(context), chunk_size))
        ]
        return torch.cat(quantiles).numpy()


class ChronosBolt(Chronos):
    """Chronos-Bolt (Amazon, 2024): T5 encoder-decoder over patches of 16 steps.

    The decoder outputs the quantiles of the next 64 steps at once.
    """

    name = "Chronos-Bolt"

    def __init__(self, checkpoint="amazon/chronos-bolt-small"):
        super().__init__(checkpoint)


class Chronos2(Forecaster):
    """Chronos-2 (Amazon, 2025): encoder-only over patches of 16 steps.

    Its group attention mixes the variates and covariates of a series, or
    several series when ``cross_learning=True``.
    """

    name = "Chronos-2"

    def load(self):
        from chronos import Chronos2Pipeline

        self.pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None,
                 cross_learning=False):
        inputs = [{"target": target, "past_covariates": {}, "future_covariates": {}} for target in context]
        for name, covariates in [("past", past_covariates), ("future", future_covariates)]:
            for row, values in zip(inputs, covariates if covariates is not None else []):
                for i, v in enumerate(values):
                    row["past_covariates"][f"{name}{i}"] = v[:context.shape[-1]]
                    if name == "future":
                        row["future_covariates"][f"{name}{i}"] = v[context.shape[-1]:]
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=QUANTILES,
            cross_learning=cross_learning,
        )
        # Chronos-2 keeps a variate axis, which univariate inputs do not have.
        return torch.stack(quantiles).squeeze(1).numpy()


class TimesFM3(Forecaster):
    """TimesFM 3 (Google, 2026): decoder-only over patches of 32 steps.

    Attention alternates over time and across variates, so it takes covariates.
    The whole horizon is predicted in one pass, from masked future patches.
    """

    name = "TimesFM 3"

    def load(self):
        from timesfm3 import TimesFM3Forecaster

        self.model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device=DEVICE)

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        n_series = len(context)
        out = self.model.predict_batch(
            contexts=list(context), horizon=horizon, return_quantiles=True, make_positive=True,
            past_only_covariates=None if past_covariates is None else list(past_covariates),
            past_future_covariates=None if future_covariates is None else list(future_covariates),
        )
        # Deciles 0.1 to 0.9, as QUANTILES
        return np.stack([o.quantiles for o in out]).reshape(n_series, *np.shape(context)[1:-1], horizon, -1)


class TiRex2(Forecaster):
    """TiRex-2 (NX-AI, 2026): xLSTM, a recurrent network, over patches of 32 steps.

    The recurrent state carries the context and masked future patches, with
    variates and covariates. It keeps the last 2048 steps and forecasts at
    most 320.
    """

    name = "TiRex-2"

    def load(self):
        from tirex2 import load_model

        self.model = load_model("NX-AI/TiRex-2", device=DEVICE)

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        from tirex2 import TimeseriesType

        univariate = context.ndim == 2
        targets = context[:, None] if univariate else context
        series = [TimeseriesType(
            target=torch.tensor(targets[i]),
            past_covariates=None if past_covariates is None else torch.tensor(past_covariates[i]),
            future_covariates=None if future_covariates is None else torch.tensor(future_covariates[i]),
        ) for i in range(len(context))]
        # Each forecast is (variates, quantiles, horizon)
        quantiles = np.stack(self.model.forecast(series, prediction_length=horizon, output_type="numpy"))
        quantiles = quantiles.transpose(0, 1, 3, 2)
        return quantiles[:, 0] if univariate else quantiles


class Timestral(Forecaster):
    """Timestral (Mistral, 2026): causal transformer over patches of 32 steps.

    Learned forecast tokens read the context, up to 32768 steps, and give
    1024 steps at once. Layers alternate attention over time and over the
    variates and covariates.
    """

    name = "Timestral"

    def __init__(self, checkpoint="mistralai/Timestral-209M"):
        self.checkpoint = checkpoint

    def load(self):
        from timestral import TimestralForecaster

        self.model = TimestralForecaster.from_pretrained(self.checkpoint, device=DEVICE)

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        with torch.no_grad():
            forecast = self.model.forecast(
                context, prediction_length=horizon,
                past_covariates=past_covariates, known_covariates=future_covariates,
            )
        quantiles = torch.stack([forecast.quantile(q) for q in QUANTILES], dim=-1).float().cpu().numpy()
        # Timestral keeps a variate axis, which univariate inputs do not have.
        return quantiles[:, 0] if context.ndim == 2 else quantiles


class T0(Forecaster):
    """t0 (The Forecasting Company, 2026): decoder over patches of 32 steps.

    Its layers alternate attention over time and over the variates, so it takes
    multivariate inputs and covariates, and decodes up to 1024 steps at once.
    """

    name = "t0"

    def __init__(self, checkpoint="theforecastingcompany/t0-gamma"):
        self.checkpoint = checkpoint

    def load(self):
        from t0 import T0Forecaster

        self.model = T0Forecaster.from_pretrained(self.checkpoint).to(DEVICE).eval()

    def forecast(self, context, horizon, past_covariates=None, future_covariates=None):
        if past_covariates is not None:
            # Past-only covariates are extra variates, forecast with the target.
            context = np.concatenate([context[:, None], past_covariates], axis=1)
        with torch.no_grad():
            out = self.model.predict(
                context, horizon=horizon, quantile_levels=QUANTILES,
                future_covariates=future_covariates,
                trim_leading_missing=True, clip_nonnegative=True,
            )
        quantiles = out.quantiles.float().cpu().numpy()
        return quantiles if past_covariates is None else quantiles[:, 0]


class TabICL(Forecaster):
    """TabICL (Inria, 2025): a tabular foundation model, not a time series one.

    Each timestep is a row of features (position, calendar, detected seasonal
    phases, covariates). The model predicts the future rows from the past ones
    in context, all horizons at once.
    """

    name = "TabICL"

    def __init__(self, max_context=2048, n_estimators=8):
        self.max_context = max_context
        self.n_estimators = n_estimators  # ensemble of views of the table, averaged

    def load(self):
        from tabicl.forecast import TabICLForecaster, _dispatch

        _dispatch.tqdm = lambda iterable, **kwargs: iterable  # no progress bar per series
        self.forecaster = TabICLForecaster(
            max_context_length=self.max_context, tabicl_config={"n_estimators": self.n_estimators},
        )

    def forecast(self, context, horizon, future_covariates=None, origins=None, freq="1h"):
        """``origins`` are the UTC times of the first forecast step, one per series."""
        n_series, n_context = context.shape
        step = pd.Timedelta(freq)
        if origins is None:
            origins = [pd.Timestamp("2000-01-01") + n_context * step] * n_series
        frames = []
        for i in range(n_series):
            # Paris time, so that the calendar features follow the local day
            timestamps = (pd.date_range(origins[i] - n_context * step, periods=n_context + horizon, freq=freq)
                          .tz_localize("UTC").tz_convert("Europe/Paris").tz_localize(None))
            frame = pd.DataFrame({"item_id": i, "timestamp": timestamps, "target": np.nan})
            frame.loc[:n_context - 1, "target"] = context[i]
            for j, values in enumerate(future_covariates[i] if future_covariates is not None else []):
                frame[f"covariate{j}"] = values
            frames.append(frame)
        frames = pd.concat(frames)
        past = frames.groupby("item_id").head(n_context)
        future = frames.groupby("item_id").tail(horizon).drop(columns="target")
        pred = self.forecaster.predict_df(past, future_df=future, quantiles=QUANTILES)
        return pred[QUANTILES].to_numpy().reshape(n_series, horizon, len(QUANTILES))
