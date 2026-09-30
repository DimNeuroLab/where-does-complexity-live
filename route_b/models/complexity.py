"""Target-conditioned complexity prediction head (Phase 3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from route_b.models.encoder import FMRIEncoder, ProjectionHead
from route_b.types import TensorMap


class CategoryFiLM(nn.Module):
  """FiLM layer: produces per-element scale (γ) and shift (β) from conditioning."""

  def __init__(self, cond_dim: int, hidden_dim: int) -> None:
    super().__init__()
    self.fc = nn.Linear(cond_dim, hidden_dim * 2)
    nn.init.zeros_(self.fc.bias)
    nn.init.zeros_(self.fc.weight)

  def forward(self, h: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
    params = self.fc(z_cond)
    (gamma, beta) = params.chunk(2, dim=-1)
    return (1 + gamma) * h + beta


class ComplexityHead(nn.Module):
  """FiLM-conditioned heteroscedastic regression."""

  def __init__(
    self,
    brain_dim: int,
    clip_text_dim: int = 768,
    n_categories: int = 16,
    cat_embed_dim: int = 32,
    hidden_dim: int = 512,
    dropout: float = 0.4
  ) -> None:
    super().__init__()
    self.cat_embedding = nn.Embedding(n_categories, cat_embed_dim)
    cond_input_dim = cat_embed_dim + clip_text_dim
    self.cond_proj = nn.Sequential(nn.Linear(cond_input_dim, hidden_dim), nn.GELU())
    cond_dim = hidden_dim
    self.brain_proj = nn.Sequential(nn.Linear(brain_dim, hidden_dim), nn.LayerNorm(hidden_dim))
    self.film1 = CategoryFiLM(cond_dim, hidden_dim)
    self.act1 = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
    self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2))
    self.film2 = CategoryFiLM(cond_dim, hidden_dim // 2)
    self.act2 = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
    self.out_mu = nn.Linear(hidden_dim // 2, 1)
    self.out_logvar = nn.Linear(hidden_dim // 2, 1)
    nn.init.constant_(self.out_logvar.bias, -3.2)
    nn.init.zeros_(self.out_logvar.weight)

  def forward(
    self,
    z_brain: torch.Tensor,
    category_idx: torch.Tensor,
    clip_text: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (mu, log_var), each (B,)."""
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
    return (mu, log_var)


class HeteroscedasticLoss(nn.Module):
  """Gaussian negative log-likelihood with learned per-sample variance."""

  def __init__(self, min_log_var: float = -7.0, max_log_var: float = 2.0) -> None:
    super().__init__()
    self.min_lv = min_log_var
    self.max_lv = max_log_var

  def forward(self, mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_var = log_var.clamp(self.min_lv, self.max_lv)
    precision = torch.exp(-log_var)
    nll = 0.5 * (precision * (target - mu) ** 2 + log_var)
    return nll.mean()


class BrainComplexityModel(nn.Module):
  """End-to-end: PCA fMRI → encoder → FiLM-conditioned complexity head + aux heads."""

  def __init__(
    self,
    fmri_dim: int,
    dino_dim: int = 1024,
    clip_dim: int = 768,
    clip_text_dim: int = 768,
    n_categories: int = 16,
    cat_embed_dim: int = 32,
    n_subjects: int = 8,
    subject_embed_dim: int = 0,
    encoder_hidden: int = 2048,
    encoder_latent: int = 2048,
    encoder_dropout: float = 0.4,
    head_hidden: int = 512,
    head_dropout: float = 0.4
  ) -> None:
    super().__init__()
    self.encoder = FMRIEncoder(
      input_dim=fmri_dim,
      n_subjects=n_subjects,
      hidden_dim=encoder_hidden,
      latent_dim=encoder_latent,
      dropout=encoder_dropout,
      subject_embed_dim=subject_embed_dim
    )
    self.dino_early_head = ProjectionHead(encoder_latent, dino_dim, hidden_dim=dino_dim)
    self.dino_mid_head = ProjectionHead(encoder_latent, dino_dim, hidden_dim=dino_dim)
    self.dino_late_head = ProjectionHead(encoder_latent, dino_dim, hidden_dim=dino_dim)
    self.clip_head = ProjectionHead(encoder_latent, clip_dim, hidden_dim=clip_dim)
    self.complexity_head = ComplexityHead(
      brain_dim=encoder_latent,
      clip_text_dim=clip_text_dim,
      n_categories=n_categories,
      cat_embed_dim=cat_embed_dim,
      hidden_dim=head_hidden,
      dropout=head_dropout
    )

  def forward(
    self,
    fmri: torch.Tensor,
    subj_id: torch.Tensor,
    category_idx: torch.Tensor,
    clip_text: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = self.encoder(fmri, subj_id)
    (mu, log_var) = self.complexity_head(z, category_idx, clip_text)
    return (mu, log_var, z)

  def forward_with_auxiliary(
    self,
    fmri: torch.Tensor,
    subj_id: torch.Tensor,
    category_idx: torch.Tensor,
    clip_text: torch.Tensor
  ) -> TensorMap:
    z = self.encoder(fmri, subj_id)
    (mu, log_var) = self.complexity_head(z, category_idx, clip_text)
    return {
      'mu': mu,
      'log_var': log_var,
      'z': z,
      'dino_early': self.dino_early_head(z),
      'dino_mid': self.dino_mid_head(z),
      'dino_late': self.dino_late_head(z),
      'clip': self.clip_head(z)
    }

  def load_pretrained_encoder(self, state_dict: TensorMap) -> None:
    """Load encoder + aux heads from Phase 2 BrainFeatureDecoder."""
    own = self.state_dict()
    (loaded, skipped) = (0, [])
    for (k, v) in state_dict.items():
      if k in own and own[k].shape == v.shape:
        own[k] = v
        loaded += 1
      else:
        skipped.append(k)
    self.load_state_dict(own, strict=False)
    print(f'  Loaded {loaded}/{len(state_dict)} pretrained → {loaded}/{len(own)} model params')
    if skipped:
      print(f'  Skipped: {skipped}')
