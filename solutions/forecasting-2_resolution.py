errors = {}
for hours in [1, 2, 4]:
    resampled = counts[[counter]].resample(f"{hours}h").sum(min_count=hours)
    ctx, fut = tu.make_windows(resampled, origins, context_length // hours, tu.WEEK // hours)
    median = t0.predict(ctx[:, 0], tu.WEEK // hours)[..., tu.MEDIAN]
    # Sum the medians over blocks of 4 hours to compare on the same scale
    blocks = 4 // hours
    median = median.reshape(len(origins), -1, blocks).sum(axis=-1)
    truth = fut[:, 0].reshape(len(origins), -1, blocks).sum(axis=-1)
    errors[f"{hours}h"] = np.nanmean(np.abs(truth - median))

pd.Series(errors, name="MAE on 4-hourly totals")
