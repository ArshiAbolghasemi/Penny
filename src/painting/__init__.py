"""Painting: the image-inpainting diffusion approach for Penny.

Concatenates a past + future LOB window into a 2-channel image and inpaints the
future region with a 2D UNet (DDPM forward, DDIM+RePaint reverse), guided by an
auxiliary DeepLOB trend loss.  Self-contained: features, preprocessing, model,
training, evaluation, and inference all live in this package.
"""
