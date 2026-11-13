worst = site_mase.sort_values(ascending=False).index[:3]
contexts, futures = tu.make_windows(sites[worst], site_origins[:1], context_length, horizon)
q = t0.predict(contexts[0], horizon)
index = pd.date_range(end=site_origins[0] + pd.Timedelta(hours=horizon - 1), periods=context_length + horizon, freq="h")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i, (ax, name) in enumerate(zip(axes, worst)):
    tu.plot_forecast(index, contexts[0, i], futures[0, i], q[i], ax=ax, label="t0", n_context=2 * tu.WEEK)
    ax.set_title(f"{name}: MASE {site_mase[name]:.2f}")
plt.tight_layout()
