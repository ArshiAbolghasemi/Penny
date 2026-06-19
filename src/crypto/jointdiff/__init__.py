"""JointDiffusion — a joint denoising-diffusion + trend classifier for LOB data.

Implements the *Joint Diffusion Models* idea (Deja et al., 2023, "Learning Data
Representations with Joint Diffusion Models"): a single U-Net is trained
simultaneously as (1) a denoising diffusion model over the past LOB feature
window and (2) a 3-class trend classifier reading the U-Net's pooled bottleneck
representation.  The denoising objective regularizes the shared encoder and
improves the learned representation used for classification.

Features are identical to DeepLOB (configurable ``ofi`` / ``lob``); the data
pipeline (loader, features, labels, windowing) is reused verbatim.
"""
