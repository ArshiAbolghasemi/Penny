"""TimesFM: univariate price-direction forecaster for Penny.

Forecasts the mid-price (as window-relative returns) from the observed past mids
using a pretrained TimesFM foundation model when available, blended with a
trainable residual transformer (and falling back to it entirely if the optional
``timesfm`` dependency is missing).  The DeepLOB trend head turns the forecast
into a direction.  Self-contained: features, preprocessing, model, training,
evaluation, and inference all live here.
"""
