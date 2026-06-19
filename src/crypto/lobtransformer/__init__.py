"""LOBTransformer: transformer classifier over windowed LOB feature matrices.

Accepts the same ``(B, 1, T_past, F)`` input as DeepLOB and JointDiffusion
and produces ``(B, 3)`` direction logits (0=down, 1=stationary, 2=up).
"""
