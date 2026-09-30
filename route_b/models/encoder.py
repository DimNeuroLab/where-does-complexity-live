"""DNN fMRI encoder with per-subject input heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from route_b.types import TensorMap


class FMRIEncoder(nn.Module):
  """Per-subject input heads + shared deeper MLP → latent."""

  def __init__(
    self,
    input_dim: int,
    n_subjects: int = 8,
    hidden_dim: int = 2048,
    latent_dim: int = 2048,
    dropout: float = 0.4,
    n_hidden_layers: int = 3,
    subject_embed_dim: int = 0
  ) -> None:
    super().__init__()
    self.n_subjects = n_subjects
    self.hidden_dim = hidden_dim
    self.subject_heads = nn.ModuleList([nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout)
    ) for _ in range(n_subjects)])
    layers: list[nn.Module] = []
    in_d = hidden_dim
    for _ in range(n_hidden_layers - 1):
      layers += [nn.Linear(in_d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
      in_d = hidden_dim
    layers.append(nn.Linear(in_d, latent_dim))
    layers.append(nn.LayerNorm(latent_dim))
    self.shared_net = nn.Sequential(*layers)
    self.latent_dim = latent_dim

  def forward(self, fmri: torch.Tensor, subj_id: torch.Tensor) -> torch.Tensor:
    """Map each subject's fMRI rows through its adapter and the shared encoder.

    :param fmri: Tensor with shape ``(batch, input_dim)``.
    :param subj_id: Integer tensor of shape ``(batch,)`` with zero-based subject IDs.
    """
    batch_size = fmri.shape[0]
    h = torch.zeros(batch_size, self.hidden_dim, device=fmri.device, dtype=fmri.dtype)
    for sid in subj_id.unique():
      mask = subj_id == sid
      h[mask] = self.subject_heads[sid.item()](fmri[mask])
    return self.shared_net(h)


class ProjectionHead(nn.Module):
  """2-layer MLP mapping latent → target feature space."""

  def __init__(self, in_dim: int, target_dim: int, hidden_dim: int = 1024) -> None:
    super().__init__()
    self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, target_dim))

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    return self.net(z)


class ReconstructionHead(nn.Module):
  """2-layer MLP mapping latent → PCA fMRI reconstruction."""

  def __init__(self, in_dim: int, target_dim: int, hidden_dim: int = 2048) -> None:
    super().__init__()
    self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, target_dim))

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    return self.net(z)


class BrainFeatureDecoder(nn.Module):
  """Encoder + per-target projection heads."""

  def __init__(
    self,
    fmri_dim: int,
    dino_dim: int = 1024,
    clip_dim: int = 768,
    n_subjects: int = 8,
    hidden_dim: int = 2048,
    latent_dim: int = 2048,
    dropout: float = 0.4,
    subject_embed_dim: int = 0
  ) -> None:
    super().__init__()
    self.encoder = FMRIEncoder(
      input_dim=fmri_dim,
      n_subjects=n_subjects,
      hidden_dim=hidden_dim,
      latent_dim=latent_dim,
      dropout=dropout,
      subject_embed_dim=subject_embed_dim
    )
    self.dino_early_head = ProjectionHead(latent_dim, dino_dim, hidden_dim=dino_dim)
    self.dino_mid_head = ProjectionHead(latent_dim, dino_dim, hidden_dim=dino_dim)
    self.dino_late_head = ProjectionHead(latent_dim, dino_dim, hidden_dim=dino_dim)
    self.clip_head = ProjectionHead(latent_dim, clip_dim, hidden_dim=clip_dim)
    self.fmri_recon_head = ReconstructionHead(latent_dim, fmri_dim, hidden_dim=hidden_dim)

  def forward(self, fmri: torch.Tensor, subj_id: torch.Tensor) -> TensorMap:
    z = self.encoder(fmri, subj_id)
    return {
      'z': z,
      'dino_early': self.dino_early_head(z),
      'dino_mid': self.dino_mid_head(z),
      'dino_late': self.dino_late_head(z),
      'clip': self.clip_head(z),
      'fmri_recon': self.fmri_recon_head(z)
    }

  def encode(self, fmri: torch.Tensor, subj_id: torch.Tensor) -> torch.Tensor:
    """Return the shared latent z (for downstream heads like complexity)."""
    return self.encoder(fmri, subj_id)


class InfoNCELoss(nn.Module):
  """Symmetric InfoNCE (i2t + t2i) with fixed temperature."""

  def __init__(self, temperature: float = 0.07) -> None:
    super().__init__()
    self.temperature = temperature

  def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.t() / self.temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


class FeatureDecodingLoss(nn.Module):
  """Combined cosine + MSE + InfoNCE loss per output head."""

  def __init__(
    self,
    lambda_cos: float = 1.0,
    lambda_mse: float = 1.0,
    lambda_infonce: float = 0.5,
    temperature: float = 0.07,
    visual_weight: float = 1.0,
    fmri_recon_weight: float = 0.0,
    dino_weight: float = 1.0,
    clip_weight: float = 1.0,
    dino_reduction: str = 'sum'
  ) -> None:
    super().__init__()
    if dino_reduction not in {'sum', 'mean'}:
      raise ValueError("dino_reduction must be 'sum' or 'mean'")
    self.cos = nn.CosineSimilarity(dim=-1)
    self.mse = nn.MSELoss()
    self.infonce = InfoNCELoss(temperature)
    self.lc = lambda_cos
    self.lm = lambda_mse
    self.li = lambda_infonce
    self.visual_weight = visual_weight
    self.fmri_recon_weight = fmri_recon_weight
    self.dino_weight = dino_weight
    self.clip_weight = clip_weight
    self.dino_reduction = dino_reduction

  def _single_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    cos = (1 - self.cos(pred, gt)).mean()
    mse = self.mse(pred, gt)
    nce = self.infonce(pred, gt)
    return (self.lc * cos + self.lm * mse + self.li * nce, {'cos': cos.item(), 'mse': mse.item(), 'nce': nce.item()})

  def forward(
    self,
    outputs: TensorMap,
    targets: TensorMap,
    fmri: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, dict[str, float | dict[str, float]]]:
    total_loss = torch.tensor(0.0, device=outputs['z'].device)
    details = {}
    if self.visual_weight > 0:
      visual_total = torch.tensor(0.0, device=outputs['z'].device)
      if self.dino_weight > 0:
        dino_total = torch.tensor(0.0, device=outputs['z'].device)
        for key in ['dino_early', 'dino_mid', 'dino_late']:
          (loss, info) = self._single_loss(outputs[key], targets[key])
          dino_total = dino_total + loss
          details[key] = info
        if self.dino_reduction == 'mean':
          dino_total = dino_total / 3
        visual_total = visual_total + self.dino_weight * dino_total
        details['dino_total'] = float(dino_total.detach().item())
      if self.clip_weight > 0:
        (clip_loss, info) = self._single_loss(outputs['clip'], targets['clip'])
        visual_total = visual_total + self.clip_weight * clip_loss
        details['clip'] = info
      total_loss = total_loss + self.visual_weight * visual_total
      details['visual_total'] = float(visual_total.detach().item())
    if self.fmri_recon_weight > 0:
      if fmri is None:
        raise ValueError('fmri target is required when fmri_recon_weight > 0')
      recon_loss = self.mse(outputs['fmri_recon'], fmri)
      total_loss = total_loss + self.fmri_recon_weight * recon_loss
      details['fmri_recon'] = {'mse': float(recon_loss.detach().item())}
    return (total_loss, details)
