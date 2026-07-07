"""JointDiffusion-Lévy: a joint jump-diffusion generative model + trend classifier
for high-frequency limit-order-book (LOB) mid-price direction forecasting.

Package layout (clear separation of concerns):

    levy.config          — dataclass config + JSON load/merge
    levy.diffusion       — forward process (Gaussian / Lévy jump-diffusion),
                           noise schedules, generalized-score tables
    levy.backbone        — U-Net + timestep conditioning (FiLM / adaLN)
    levy.heads           — diffusion (score) head + multi-horizon trend heads
    levy.losses          — generalized-score MSE, Kendall-Gal uncertainty, PCGrad
    levy.data            — synthetic LOB generator, Binance/Nobitex adapters,
                           windowing + per-horizon labels, Dataset interface
    levy.model           — assembles backbone + heads -> JointDiffusionLevy
    levy.train           — runnable training script (config-driven)
    levy.evaluate        — per-horizon accuracy / precision / recall / F1

Build order (each component is unit-tested in tests/levy/ before wiring):
    1. diffusion (schedules -> generalized_score -> forward)
    2. data (synthetic -> windowing -> real crypto adapters -> dataset)
    3. backbone + heads + losses
    4. model + train + evaluate + feature-only inference
"""

__all__ = ["__version__"]
__version__ = "0.0.1"
