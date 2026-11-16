"""Figures of Act 1, as PDFs in the colours of the tm theme. Run from the repo root."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, PRIMARY, ACCENT, RULE = "#24221d", "#716b5e", "#1f6f50", "#a8680f", "#c4b89a"
OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": RULE, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.transparent": True, "savefig.bbox": "tight",
})

# Both directions of the Sébastopol counter, summed into one hourly series
seb = pd.read_parquet("data/sebastopol.parquet")
y = seb.groupby("date")["bike_count"].sum(min_count=1).asfreq("h")

# Slide 3: one week, in order and shuffled
week = y["2025-03-10":"2025-03-16"]
shuffled = week.sample(frac=1, random_state=0).to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.9), sharey=True)
axes[0].plot(week.index, week.to_numpy(), color=PRIMARY, lw=1)
axes[1].plot(week.index, shuffled, color=ACCENT, lw=1)
axes[0].set_title("hourly counts, in order", color=PRIMARY, fontsize=9)
axes[1].set_title("the same values, shuffled", color=ACCENT, fontsize=9)
axes[0].set_ylabel("bikes per hour")
for ax in axes:
    ax.set_xticks(week.index[12::24], week.index[12::24].strftime("%a"))
fig.savefig(OUT / "week_shuffled.pdf")

# Slide 5: zoom from one day to six years
zooms = [
    ("one day", y["2025-03-11"], "%Hh", 6),
    ("one week", y["2025-03-10":"2025-03-16"], "%a", 48),
    ("one year (daily)", y["2025"].resample("D").sum(), "%b", 61),
    ("2020 to 2025 (weekly)", y.resample("W").sum(), "%Y", 104),
]
fig, axes = plt.subplots(1, 4, figsize=(6.4, 1.6))
for ax, (title, s, fmt, step) in zip(axes, zooms):
    ax.plot(s.index, s.to_numpy(), color=PRIMARY, lw=.8)
    ax.set_title(title, fontsize=8, color=MUTED)
    ax.set_xticks(s.index[::step], s.index[::step].strftime(fmt), fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
fig.tight_layout(w_pad=.6)
fig.savefig(OUT / "seasonalities.pdf")

# Slide 6: monthly mean and std of the hourly counts
monthly = y.resample("MS").agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(5.2, 1.9))
ax.plot(monthly.index, monthly["mean"], color=PRIMARY, lw=1.2, label="monthly mean")
ax.plot(monthly.index, monthly["std"], color=ACCENT, lw=1.2, label="monthly std")
ax.axvspan(pd.Timestamp("2020-03-17"), pd.Timestamp("2020-05-11"), color=RULE, alpha=.4, lw=0)
ax.text(pd.Timestamp("2020-04-14"), monthly["std"].max(), "lockdown", ha="center",
        va="top", fontsize=7, color=MUTED)
ax.set_ylabel("bikes per hour")
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.savefig(OUT / "nonstationary.pdf")

# Slide 7: missing and zero days of every counter, 2024 and 2025
sites = pd.read_parquet("data/bike-counters-2025.parquet")
daily = (sites.set_index("date").groupby("counter_name", observed=True)["bike_count"]
         .resample("D").agg(lambda c: np.nan if c.isna().all() else c.sum()).unstack())
state = np.where(daily.isna(), 2, np.where(daily == 0, 1, 0))
order = np.argsort(state.sum(axis=1))
fig, ax = plt.subplots(figsize=(6.4, 2.2))
cmap = plt.matplotlib.colors.ListedColormap(["#e8e0cb", ACCENT, INK])
ax.imshow(state[order], aspect="auto", cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
months = daily.columns[daily.columns.day == 1][::3]
ax.set_xticks([daily.columns.get_loc(m) for m in months], months.strftime("%b %y"), fontsize=7)
ax.set_yticks([])
ax.set_ylabel(f"{len(daily)} counters")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ACCENT, INK)]
ax.legend(handles, ["zero all day", "missing"], frameon=False, fontsize=7,
          loc="upper left", bbox_to_anchor=(1, 1))
fig.savefig(OUT / "missing_2025.pdf")
