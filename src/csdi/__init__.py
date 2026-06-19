"""CSDI: multivariate price-direction forecaster for Penny.

Encodes the observed past LOB window — a multivariate ``(2R, T_past)`` series of
per-level order flow / price and depth — with stacked feature-axis and time-axis
transformers (the CSDI two-axis design, used here as a deterministic forecaster
rather than a diffusion imputer) and predicts the future mid-return series.  The
DeepLOB trend head turns the forecast into a direction.  Self-contained: its own
features, preprocessing, model, training, evaluation, and inference.
"""
