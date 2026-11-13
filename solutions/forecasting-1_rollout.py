def rollout(model, context, horizon, step):
    """Forecast `horizon` steps, `step` at a time, feeding the median back into the context."""
    quantiles = []
    for _ in range(0, horizon, step):
        q = model.predict(context, step)
        quantiles.append(q)
        context = np.concatenate([context, q[..., tu.MEDIAN]], axis=-1)
    return np.concatenate(quantiles, axis=-2)[..., :horizon, :]
