"""Fused Proxy Loss: unifies VAPL variation vectors with SCDL distributional proxy.

Score function  g(z, c):
    rep_term(z, c) = (1/S) Σ_s cos(z, normalize(μ_c + σ_c ⊙ ε_s))   MC sampling, grad → μ_c, σ_c
    var_term(z, c) = max_k  cos(z, normalize(v_{c,k}))                 hard max → forces specialisation
    g(z, c) = rep_term + lambda_var * var_term

CDBA losses (no ground-truth needed — run on ALL pixels, labeled + unlabeled):
    P(c|z)  = softmax_c( g(z, c) )
    L_E2P   = mean_n [ sum_c  P(c|z_n) * (1 - g(z_n, c)) ]
    L_P2E   = (1/C) sum_c  exp( -mean_n[ (2*P(c|z_n) - 1) * g(z_n, c) ] )
    L_cdba  = L_E2P + L_P2E

SAC loss (labeled pixels only — anchors proxy μ_c toward labeled class centroids):
    anchor_c = mean( z_{i,l}.detach() | label == c )
    L_SAC    = (1/C) sum_c  [ 1 - cos(μ_c, anchor_c) ]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FusedProxyStats:
    loss_cdba: torch.Tensor
    loss_e2p: torch.Tensor
    loss_p2e: torch.Tensor
    loss_sac: torch.Tensor
    g_pos_mean: torch.Tensor
    valid_tokens: torch.Tensor


def _groups_for(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class FusedProxyLoss(nn.Module):
    """Unified distributional + variation proxy loss for 3D segmentation features."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_dim: int = 256,
        num_variations: int = 5,
        lambda_var: float = 1.0,
        proxy_samples: int = 8,
        ignore_index: int = 255,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_variations = num_variations
        self.lambda_var = lambda_var
        self.proxy_samples = proxy_samples
        self.ignore_index = ignore_index
        self.eps = eps

        self.projector = nn.Sequential(
            nn.Conv3d(in_channels, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_groups_for(embedding_dim), num_channels=embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1, bias=True),
        )
        self.proxies = nn.Parameter(torch.empty(num_classes, embedding_dim * 2))
        self.variation_vectors = nn.Parameter(torch.empty(num_classes, num_variations, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proxies)
        nn.init.xavier_uniform_(self.variation_vectors)

    def forward(
        self,
        features: torch.Tensor,
        label_l: torch.Tensor | None = None,
        labeled_bs: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, FusedProxyStats]:
        """
        Args:
            features:   [B, C_in, D, H, W]  raw backbone features (full batch)
            label_l:    [labeled_bs, D, H, W] ground-truth labels (labeled portion only)
            labeled_bs: number of labeled samples in features

        Returns:
            loss_cdba, loss_sac, FusedProxyStats
        """
        embeddings = F.normalize(self.projector(features), p=2, dim=1)  # [B, D, sD, sH, sW]
        flat_all = embeddings.movedim(1, -1).reshape(-1, self.embedding_dim)  # [N_all, D]

        g_all = self._cdba_compute_g(flat_all)               # [N_all, C] — μ/σ detached
        loss_e2p, loss_p2e = self._cdba_loss(g_all)
        loss_cdba = loss_e2p + loss_p2e

        loss_sac = features.sum() * 0.0
        g_pos_mean = features.sum().detach() * 0.0

        if labeled_bs > 0 and label_l is not None:
            emb_labeled = embeddings[:labeled_bs]             # [labeled_bs, D, ...]
            targets = self._resize_targets(label_l, emb_labeled.shape[2:])
            flat_emb_l, flat_tgt_l = self._flatten_valid(emb_labeled, targets)

            if flat_emb_l.numel() > 0:
                loss_sac = self._sac_loss(flat_emb_l, flat_tgt_l)
                g_l = self._compute_g(flat_emb_l)
                arange = torch.arange(flat_tgt_l.numel(), device=flat_tgt_l.device)
                g_pos_mean = g_l[arange, flat_tgt_l].detach().mean()

        stats = FusedProxyStats(
            loss_cdba=loss_cdba.detach(),
            loss_e2p=loss_e2p.detach(),
            loss_p2e=loss_p2e.detach(),
            loss_sac=loss_sac.detach(),
            g_pos_mean=g_pos_mean,
            valid_tokens=torch.as_tensor(
                flat_all.shape[0], device=features.device, dtype=torch.float32
            ),
        )
        return loss_cdba, loss_sac, stats

    def forward_pseudo(
        self,
        features_u: torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """FusedProxy CE loss on pseudo-labeled unlabeled pixels.

        Proxy (μ/σ) is detached so pseudo-label noise only trains the projector.
        _flatten_valid auto-excludes ignore_index=255 (low-confidence pixels).
        """
        embeddings = F.normalize(self.projector(features_u), p=2, dim=1)
        targets = self._resize_targets(pseudo_labels, embeddings.shape[2:])
        flat_emb, flat_tgt = self._flatten_valid(embeddings, targets)
        if flat_emb.numel() == 0:
            return features_u.sum() * 0.0
        g = self._cdba_compute_g(flat_emb)   # proxies detached
        return F.cross_entropy(g, flat_tgt)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _proxy_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.proxies[:, :self.embedding_dim]
        sigma = F.softplus(self.proxies[:, self.embedding_dim:]).clamp_min(self.eps)
        return mu, sigma

    def _compute_g(self, z: torch.Tensor) -> torch.Tensor:
        """g(z, c) for every pixel z and every class c.  Returns [N, C]."""
        mu, sigma = self._proxy_params()   # [C, D], [C, D]

        # rep_term: Monte-Carlo E_u~N(μ_c,σ_c²)[cos(z, u)]
        noise = torch.randn(
            self.num_classes, self.proxy_samples, self.embedding_dim,
            device=mu.device, dtype=mu.dtype,
        )  # [C, S, D]
        samples = F.normalize(
            mu.unsqueeze(1) + sigma.unsqueeze(1) * noise, p=2, dim=-1
        )  # [C, S, D]
        # z: [N, D];  samples_flat: [C*S, D]  →  matmul → [N, C*S] → [N, C, S] → mean → [N, C]
        samples_flat = samples.view(self.num_classes * self.proxy_samples, self.embedding_dim)
        rep_term = (
            torch.matmul(z, samples_flat.t())
            .view(-1, self.num_classes, self.proxy_samples)
            .mean(dim=2)
        )  # [N, C]

        # var_term: hard-max cosine to variation vectors  →  forces specialisation
        var_norm = F.normalize(self.variation_vectors, p=2, dim=-1)  # [C, K, D]
        var_flat = var_norm.view(self.num_classes * self.num_variations, self.embedding_dim)
        var_term = (
            torch.matmul(z, var_flat.t())
            .view(-1, self.num_classes, self.num_variations)
            .max(dim=2).values
        )  # [N, C]

        return rep_term + self.lambda_var * var_term  # [N, C]

    def _cdba_compute_g(self, z: torch.Tensor) -> torch.Tensor:
        """g for CDBA path: μ and σ are detached so CDBA only trains projector + variation_vectors."""
        mu, sigma = self._proxy_params()
        mu = mu.detach()
        sigma = sigma.detach()

        noise = torch.randn(
            self.num_classes, self.proxy_samples, self.embedding_dim,
            device=mu.device, dtype=mu.dtype,
        )
        samples = F.normalize(
            mu.unsqueeze(1) + sigma.unsqueeze(1) * noise, p=2, dim=-1
        )
        samples_flat = samples.view(self.num_classes * self.proxy_samples, self.embedding_dim)
        rep_term = (
            torch.matmul(z, samples_flat.t())
            .view(-1, self.num_classes, self.proxy_samples)
            .mean(dim=2)
        )

        var_norm = F.normalize(self.variation_vectors, p=2, dim=-1)
        var_flat = var_norm.view(self.num_classes * self.num_variations, self.embedding_dim)
        var_term = (
            torch.matmul(z, var_flat.t())
            .view(-1, self.num_classes, self.num_variations)
            .max(dim=2).values
        )

        return rep_term + self.lambda_var * var_term

    def _cdba_loss(
        self, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """L_E2P and L_P2E.  No ground-truth labels needed."""
        p = torch.softmax(g, dim=1)  # [N, C]

        # E2P: each embedding aligns with all proxies weighted by soft assignment
        loss_e2p = (p * (1.0 - g)).sum(dim=1).mean()

        # P2E: each proxy discriminates its softly-assigned embeddings
        margin = ((2.0 * p - 1.0) * g).mean(dim=0)  # [C]
        loss_p2e = torch.exp(-margin).mean()

        return loss_e2p, loss_p2e

    def _sac_loss(
        self,
        flat_emb: torch.Tensor,
        flat_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Anchor each proxy μ_c toward the labeled class centroid (SAC)."""
        mu, _ = self._proxy_params()
        mu_norm = F.normalize(mu, p=2, dim=1)

        losses = []
        for c in torch.unique(flat_targets):
            mask = flat_targets == c
            if not mask.any():
                continue
            anchor = F.normalize(flat_emb[mask].mean(dim=0).detach(), p=2, dim=0)
            losses.append(1.0 - (anchor * mu_norm[c]).sum())

        if not losses:
            return mu.sum() * 0.0
        return torch.stack(losses).mean()

    def _resize_targets(self, targets: torch.Tensor, size: tuple) -> torch.Tensor:
        if targets.shape[1:] == size:
            return targets.long()
        resized = F.interpolate(targets.float().unsqueeze(1), size=size, mode="nearest")
        return resized[:, 0].long()

    def _flatten_valid(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_emb = embeddings.movedim(1, -1).reshape(-1, embeddings.shape[1])
        flat_tgt = targets.reshape(-1)
        valid = (
            (flat_tgt != self.ignore_index)
            & (flat_tgt >= 0)
            & (flat_tgt < self.num_classes)
        )
        return flat_emb[valid], flat_tgt[valid]
