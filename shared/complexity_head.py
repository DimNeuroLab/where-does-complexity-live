"""Target-conditioned complexity prediction head.

Architecture::

    z_cond = MLP(cat_embed || clip_text)          # target conditioning
    FiLM layers modulate an input representation with z_cond
    (mu, log_sigma^2) = final MLP(modulated representation)

The head predicts *within-category residuals* (``score - category_mean``).
Category conditioning uses both a learned embedding and CLIP text embeddings.
FiLM conditioning forces the category signal to modulate the input
representation rather than being concatenated with it, which would let the
model learn a category-only shortcut.

Heteroscedastic regression: the head outputs both a mean prediction and a
learned log-variance per sample, so that predictions with an inherently
noisy mapping to complexity are automatically downweighted by the loss.

This head is reused, unmodified, by Route A's embedding-conditioned
predictor (``route_a/embedding``) and by Route B's neural readout, which is
why it lives in ``shared/`` rather than with either route.
"""

from __future__ import annotations

import torch
from torch import nn


class CategoryFiLM(nn.Module):
  """Produces a per-element scale (gamma) and shift (beta) from a
  conditioning vector, applied as ``h = (1 + gamma) * h + beta``.

  The linear layer is zero-initialised, so a freshly constructed FiLM layer
  starts as the identity.
  """

  def __init__(self, cond_dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.fc = nn.Linear(cond_dim, hidden_dim * 2)
    nn.init.zeros_(self.fc.bias)
    nn.init.zeros_(self.fc.weight)

  def forward(self, h: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
    gamma, beta = self.fc(z_cond).chunk(2, dim=-1)
    return (1 + gamma) * h + beta


class ComplexityHead(nn.Module):
  """FiLM-conditioned heteroscedastic regression head.

  :param brain_dim: Dimensionality of the input representation (a brain
    embedding for Route B, or an image embedding for Route A's embedding
    predictor).
  :param clip_text_dim: Dimensionality of the CLIP text conditioning vector.
  :param n_categories: Number of target categories, for the learned
    category embedding.
  :param cat_embed_dim: Dimensionality of the learned category embedding.
  :param hidden_dim: Width of the first FiLM-conditioned block.
  :param dropout: Dropout applied after each FiLM block.
  """

  def __init__(
    self,
    brain_dim: int,
    clip_text_dim: int = 768,
    n_categories: int = 16,
    cat_embed_dim: int = 32,
    hidden_dim: int = 512,
    dropout: float = 0.2,
  ) -> None:
    super().__init__()
    self.cat_embedding = nn.Embedding(n_categories, cat_embed_dim)
    cond_input_dim = cat_embed_dim + clip_text_dim
    self.cond_proj = nn.Sequential(
      nn.Linear(cond_input_dim, hidden_dim),
      nn.GELU(),
    )
    cond_dim = hidden_dim

    self.brain_proj = nn.Sequential(
      nn.Linear(brain_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
    )
    self.film1 = CategoryFiLM(cond_dim, hidden_dim)
    self.act1 = nn.Sequential(nn.GELU(), nn.Dropout(dropout))

    self.fc2 = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim // 2),
      nn.LayerNorm(hidden_dim // 2),
    )
    self.film2 = CategoryFiLM(cond_dim, hidden_dim // 2)
    self.act2 = nn.Sequential(nn.GELU(), nn.Dropout(dropout))

    self.out_mu = nn.Linear(hidden_dim // 2, 1)
    self.out_logvar = nn.Linear(hidden_dim // 2, 1)
    # Small initial variance: log(0.04) ~= -3.2.
    nn.init.constant_(self.out_logvar.bias, -3.2)
    nn.init.zeros_(self.out_logvar.weight)

  def forward(
    self,
    z_brain: torch.Tensor,
    category_idx: torch.Tensor,
    clip_text: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the head.

    :param z_brain: Input representation, shape ``(batch, brain_dim)``.
    :param category_idx: Target category index per sample, shape ``(batch,)``.
    :param clip_text: CLIP text conditioning vector per sample, shape
      ``(batch, clip_text_dim)``.
    :returns: ``(mu, log_var)``, each shape ``(batch,)``.
    """
    z_cat = self.cat_embedding(category_idx)
    z_cond = self.cond_proj(torch.cat([z_cat, clip_text], dim=-1))

    h = self.brain_proj(z_brain)
    h = self.film1(h, z_cond)
    h = self.act1(h)

    h = self.fc2(h)
    h = self.film2(h, z_cond)
    h = self.act2(h)

    mu = self.out_mu(h).squeeze(-1)
    log_var = self.out_logvar(h).squeeze(-1)
    return mu, log_var


class HeteroscedasticLoss(nn.Module):
  """Gaussian negative log-likelihood with a learned per-sample variance.

  ``L = 0.5 * [(y - mu)^2 / sigma^2 + log(sigma^2)]``. The model learns
  which samples have a noisy mapping to complexity and downweights them
  automatically.
  """

  def __init__(self, min_log_var: float = -7.0, max_log_var: float = 2.0) -> None:
    super().__init__()
    self.min_log_var = min_log_var
    self.max_log_var = max_log_var

  def forward(self, mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_var = log_var.clamp(self.min_log_var, self.max_log_var)
    precision = torch.exp(-log_var)
    nll = 0.5 * (precision * (target - mu) ** 2 + log_var)
    return nll.mean()
