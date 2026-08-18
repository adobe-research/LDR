"""
Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.

This file constitutes Licensed Materials under the Adobe Research License.
Use is limited to noncommercial research purposes.
See the LICENSE file at the project root for the complete license terms and disclaimer.

Latent Dynamics Reasoning (LDR). See ldr/rollout.md for the rollout derivation.
"""

from .model import build_ldr, LDR, PerceptualPyramidLoss

__all__ = ["build_ldr", "LDR", "PerceptualPyramidLoss"]
