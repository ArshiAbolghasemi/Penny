# StochLOB stochastic forecaster

StochLOB retains the AlphaStableLOB sequence encoder but does not construct or train
its score head. The optional BiN, two-layer BiGRU, temporal multi-head attention,
AdaLN timestep conditioning, and attention pool encode the observed `(60, 31)` window
once into `z0`. A custom PyTorch Euler rollout then models predictive latent futures;
the historical samples are not solver steps.

At every one of the default 15 steps over `τ ∈ [0,1]`, horizon-conditioned networks
predict one shared drift, stable scale `γ`, Poisson intensity `λ`, deterministic jump
magnitude, and two router logits. Softmax weights sum to one and mix both increments:

```text
z_next = z + μ dt
           + π_alpha γ dt^(1/1.62) ξ_alpha
           + π_jump N J,                 N ~ Poisson(λ dt)
```

The router is recomputed from the evolving state. Alpha is fixed at `1.62`. Stable
draws, scales, intensity, jump magnitudes, and latent states have configurable safety
bounds; LayerNorm and gradient clipping provide additional control. Poisson counts are
true samples in the forward pass with a straight-through gradient for their mean.

Training draws four trajectories by default and supports two classification
objectives. `pathwise` averages cross-entropy over trajectories, requiring every
sampled future to remain predictive. `marginal` applies NLL to the mean trajectory
probability, exactly matching inference-time Monte Carlo aggregation:

```text
L_path     = mean_m [-log p_m(y)]
L_marginal = -log(mean_m p_m(y))
```

The supplied configs select `classification_objective: "marginal"` so predictive
training and inference use the same mixture distribution. Both losses are logged on
every batch for a controlled diversity/performance ablation. The CLI option
`--classification-objective pathwise|marginal` overrides the config. The optional
route entropy term is added to the selected classification loss; there is no
score-matching objective in StochLOB.

The route entropy term defaults to zero. If enabled, it anneals away and encourages
early exploration without imposing a 50/50 marginal route.

Inference remains stochastic. The default 20 trajectories are converted to class
probabilities and averaged. Evaluation reports predictive entropy, squared trajectory
disagreement, error-detection AUROC/AUPRC, ECE, Brier score, risk-coverage curves,
per-horizon behavior, and routing/intensity deciles for stress, realized variance, and
variance-controlled jump residual. It also evaluates `M = 1, 5, 10, 20, 50`.

`latent_dynamics` selects `deterministic`, `gaussian`, `stable`, `jump`, or `routed`,
so the central fixed-stable versus fixed-jump versus adaptive-routing comparison shares
one encoder and classifier implementation. The legacy AlphaStableLOB, JumpGateLOB,
and GaussGateLOB trainers remain available as separate score-matching baselines.

```bash
uv run python -m crypto.train_stochlob configs/crypto/coinbase/stochlob/btcirt_ofi_k10.json
```
