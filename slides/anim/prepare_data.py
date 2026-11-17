"""Small JSON of the bike data for the tasks animation. Run from the repo root."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "bikes.json"


def paris(index):
    """UTC timestamps of the data to Paris wall-clock time."""
    return index.tz_localize("UTC").tz_convert("Europe/Paris").tz_localize(None)


seb = pd.read_parquet("data/sebastopol.parquet")
y = seb.groupby("date")["bike_count"].sum(min_count=1).asfreq("h").interpolate(limit=3)
y.index = paris(y.index)
weather = pd.read_parquet("data/paris-weather.parquet").set_index("date")[["precipitation", "wind_speed"]]
weather.index = paris(weather.index)

# The week of the 18 September 2025 transport strike, its weather, and the days after it to forecast
week = y["2025-09-15":"2025-09-21"]
future = y["2025-09-22":"2025-09-24"]
# The covariates are known over the forecast window too
week_weather = weather["2025-09-15":"2025-09-23 23:00"]

counters = pd.read_parquet("data/bike-counters-2025.parquet")
counters["date"] = paris(pd.DatetimeIndex(counters["date"]))
sites = counters.groupby("counter_name", observed=True)[["latitude", "longitude"]].first()
# One weekday of every counter, scaled by its own maximum, to pulse the map
day = counters[counters.date.between("2025-09-16", "2025-09-16 23:00")]
profiles = day.pivot_table(index="counter_name", columns="date", values="bike_count", observed=True)
profiles = profiles.div(profiles.max(axis=1), axis=0).fillna(0).reindex(sites.index).fillna(0)

data = {
    "week": week.tolist(),
    "future": future.tolist(),
    "rain": week_weather.precipitation.round(1).tolist(),
    "wind": week_weather.wind_speed.round(1).tolist(),
    "sites": {
        "lat": sites.latitude.round(5).tolist(),
        "lon": sites.longitude.round(5).tolist(),
        "profile": np.round(profiles.to_numpy(), 3).tolist(),
    },
}
OUT.write_text(json.dumps(data))
print(f"{len(week)} hours, {len(week_weather)} weather hours, {len(sites)} sites -> {OUT}")
