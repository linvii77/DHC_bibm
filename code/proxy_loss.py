"""Proxy learning modules for DHC integration.
Extracted from BIBM_medical/vap_pidnet/vapl.py – logic unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


SoftmaxScope = Literal["per_class", "global"]


class ProjectionHead(nn.Module):
    """Small projection head used only during training."""

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 256,
        hidden_channels: int | None = None,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        if spatial_dims not in {2, 3}:
            raise ValueError("spatial_dims must be 2 or 3.")
        hidden_channels = hidden_channels or embedding_dim
        conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        norm = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
        self.proj = nn.Sequential(
            conv(in_channels, hidden_channels, kernel_size=1, bias=False),
            norm(hidden_channels),
            nn.ReLU(inplace=True),
            conv(hidden_channels, embedding_dim, kernel_size=1, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(features), p=2, dim=1)


@dataclass(frozen=True)
class VAPLStats:
    loss_cs: torch.Tensor
    loss_attraction: torch.Tensor
    loss_repulsion: torch.Tensor
    positive_probability: torch.Tensor
    negative_probability: torch.Tensor
    hard_fraction: torch.Tensor
    valid_pixels: torch.Tensor
    proxy_assignment_accuracy: torch.Tensor
    proxy_sigma_mean: torch.Tensor
    p_sub_entropy: float  # entropy of variation sub-distribution; 0.0 when use_variation=False


class CompositionalSimilarityLoss(nn.Module):
    """Compositional Similarity Loss from the paper methodology.

    ``softmax_scope="per_class"`` is the literal formula in the PDF:
    p_sub is normalized across variation vectors inside each class.
    ``softmax_scope="global"`` is kept only for diagnostics and is not
    used by the default reproduction path.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        num_variations: int = 5,
        lambda_var: float = 1.0,
        tau: float = 10.0,
        gamma: float = 2.0,
        tau_r: float = 0.8,
        lambda_r: float = 1.0,
        ignore_index: int = 255,
        softmax_scope: SoftmaxScope = "per_class",
        proxy_sigma_min: float = 0.05,
        use_variation: bool = True,
        eps: float = 1.0e-7,
        max_samples_per_class: int = 0,
        num_proxy_samples: int = 4,
        tau_var_cdba: float = 5.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if num_variations < 1:
            raise ValueError("num_variations must be positive.")
        if softmax_scope not in {"per_class", "global"}:
            raise ValueError("softmax_scope must be 'per_class' or 'global'.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_variations = num_variations
        self.lambda_var = lambda_var
        self.tau = tau
        self.gamma = gamma
        self.tau_r = tau_r
        self.lambda_r = lambda_r
        self.ignore_index = ignore_index
        self.softmax_scope = softmax_scope
        self.proxy_sigma_min = proxy_sigma_min
        self.use_variation = use_variation
        self.max_samples_per_class = max_samples_per_class
        self.num_proxy_samples = num_proxy_samples
        self.tau_var_cdba = tau_var_cdba
        # Runtime-toggleable switch for a "variation warmup" schedule: even
        # when use_variation=True (variation_vectors exists), forward()
        # behaves as combined=q_c (no p_sub term, no gradient to
        # variation_vectors) while this is False. The training script flips
        # it to True after a warmup period so (mu_c, sigma_c) stabilizes
        # before the variation sub-distribution is introduced.
        self.variation_active = True
        self.eps = eps

        # Replaces the single-point representative proxy with an SCDL-style
        # per-class Gaussian (mu, sigma), stored as a single [C, 2*D] tensor.
        self.proxy_dist = nn.Parameter(
            torch.empty(num_classes, embedding_dim * 2)
        )
        # use_variation=False is a clean single-Gaussian (SCDL-style) proxy
        # ablation: no variation_vectors parameter is created at all, and
        # forward() sets combined = q_c directly (no p_sub term).
        if self.use_variation:
            self.variation_vectors = nn.Parameter(
                torch.empty(num_classes, num_variations, embedding_dim)
            )
        else:
            self.variation_vectors = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proxy_dist[:, :self.embedding_dim])
        nn.init.constant_(self.proxy_dist[:, self.embedding_dim:], -2.0)
        if self.use_variation:
            nn.init.xavier_uniform_(self.variation_vectors)

    def forward(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, VAPLStats]:
        if embeddings.ndim not in {4, 5}:
            raise ValueError("embeddings must have shape [B, D, H, W] or [B, D, Z, H, W].")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected embedding dim {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        targets = self._resize_targets(targets, embeddings.shape[2:])
        flat_embeddings, flat_targets = self._flatten_valid(embeddings, targets)
        if self.max_samples_per_class > 0:
            flat_embeddings, flat_targets = self._balance_classes(flat_embeddings, flat_targets)

        if flat_embeddings.numel() == 0:
            zero = embeddings.sum() * 0.0
            stats = VAPLStats(
                loss_cs=zero,
                loss_attraction=zero,
                loss_repulsion=zero,
                positive_probability=zero,
                negative_probability=zero,
                hard_fraction=zero,
                valid_pixels=torch.zeros((), device=embeddings.device),
                proxy_assignment_accuracy=zero,
                proxy_sigma_mean=zero,
                p_sub_entropy=0.0,
            )
            return zero, stats

        x = F.normalize(flat_embeddings, p=2, dim=1)
        arange = torch.arange(flat_targets.numel(), device=flat_targets.device)

        # SCDL-style distribution proxy: q[:, c] is the probability that a
        # token belongs to class c, derived from the per-class Gaussian
        # (mu_c, sigma_c). This replaces the single-point representative
        # proxy from the original factorized similarity score.
        q, sigma_c = self._proxy_assignment_probabilities(x)
        if self.use_variation and self.variation_active:
            # Per-class softmax over variation vectors (unchanged from the
            # original formulation, minus the additive proxy_sim term that
            # was cancelled by this same softmax).
            p_sub = self._variation_subdistribution(x)
            # Entropy of the variation sub-distribution (for mechanism logging).
            p_sub_entropy = -(p_sub * (p_sub + 1e-8).log()).sum(dim=-1).mean().item()
            # Joint distribution over (class, variation): combined[n, c, k] =
            # q_c(x_n) * p_sub(x_n, v_{c,k} | c). Sums to 1 over (c, k).
            combined = q.unsqueeze(-1) * p_sub
        else:
            # Clean single-Gaussian (SCDL-style) degeneration: no variation
            # sub-distribution at all, combined = q_c.
            p_sub_entropy = 0.0
            combined = q.unsqueeze(-1)

        p_pos = combined[arange, flat_targets].amax(dim=1).clamp_min(self.eps)
        loss_attraction = -torch.log(p_pos).mean()

        p_neg = self._negative_probability(combined, flat_targets)
        p_neg_for_log = p_neg.clamp(min=0.0, max=1.0 - self.eps)
        ratio = p_neg_for_log / p_pos.clamp_min(self.eps)
        hard_mask = ratio > self.tau_r

        if hard_mask.any():
            focal_weight = p_neg_for_log[hard_mask].pow(self.gamma)
            loss_repulsion = -(
                focal_weight * torch.log1p(-p_neg_for_log[hard_mask])
            ).mean()
        else:
            loss_repulsion = loss_attraction.new_zeros(())

        loss_cs = loss_attraction + self.lambda_r * loss_repulsion
        proxy_assignment_accuracy = (q.argmax(dim=1) == flat_targets).float().mean()
        stats = VAPLStats(
            loss_cs=loss_cs,
            loss_attraction=loss_attraction,
            loss_repulsion=loss_repulsion,
            positive_probability=p_pos.detach().mean(),
            negative_probability=p_neg_for_log.detach().mean(),
            hard_fraction=hard_mask.float().detach().mean(),
            valid_pixels=torch.as_tensor(
                flat_targets.numel(), device=embeddings.device, dtype=torch.float32
            ),
            proxy_assignment_accuracy=proxy_assignment_accuracy.detach(),
            proxy_sigma_mean=sigma_c.detach().mean(),
            p_sub_entropy=p_sub_entropy,
        )
        return loss_cs, stats

    def _resize_targets(
        self, targets: torch.Tensor, size: tuple[int, ...]
    ) -> torch.Tensor:
        if targets.ndim in {4, 5} and targets.shape[1] == 1:
            targets = targets[:, 0]
        expected_ndim = len(size) + 1
        if targets.ndim != expected_ndim:
            raise ValueError(
                "targets must have shape [B, ...] or [B, 1, ...] matching embeddings."
            )
        if targets.shape[1:] == size:
            return targets.long()
        resized = F.interpolate(
            targets.unsqueeze(1).float(), size=size, mode="nearest"
        )
        return resized[:, 0].long()

    def _balance_classes(
        self, emb: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample at most max_samples_per_class voxels per class for equal gradient contribution."""
        balanced_emb, balanced_tgt = [], []
        for c in range(self.num_classes):
            mask = targets == c
            n = mask.sum()
            if n == 0:
                continue
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            if n > self.max_samples_per_class:
                perm = torch.randperm(n, device=emb.device)[:self.max_samples_per_class]
                idx = idx[perm]
            balanced_emb.append(emb[idx])
            balanced_tgt.append(targets[idx])
        if not balanced_emb:
            return emb, targets
        return torch.cat(balanced_emb, 0), torch.cat(balanced_tgt, 0)

    def _flatten_valid(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_embeddings = embeddings.movedim(1, -1).reshape(-1, embeddings.shape[1])
        flat_targets = targets.reshape(-1)
        valid = (
            (flat_targets != self.ignore_index)
            & (flat_targets >= 0)
            & (flat_targets < self.num_classes)
        )
        return flat_embeddings[valid], flat_targets[valid]

    def _proxy_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.proxy_dist[:, : self.embedding_dim]
        sigma = F.softplus(self.proxy_dist[:, self.embedding_dim :])
        sigma = sigma.clamp_min(self.proxy_sigma_min)
        return mu, sigma

    def _proxy_assignment_probabilities(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SCDL-style class assignment probability q_c(x) = softmax_c(sim(x, mu_c) / sigma_c)."""
        mu, sigma = self._proxy_params()
        mu_norm = F.normalize(mu, p=2, dim=1)
        sigma_c = sigma.mean(dim=1)
        proxy_logits = torch.matmul(x, mu_norm.t()) / sigma_c.unsqueeze(0)
        q = torch.softmax(proxy_logits, dim=1)
        return q, sigma_c

    def _variation_subdistribution(self, x: torch.Tensor) -> torch.Tensor:
        variations = F.normalize(self.variation_vectors, p=2, dim=-1)
        variation_sim = torch.einsum("nd,ckd->nck", x, variations)
        scores = self.lambda_var * variation_sim

        if self.softmax_scope == "per_class":
            return torch.softmax(self.tau * scores, dim=2)

        flat_scores = scores.flatten(1)
        flat_prob = torch.softmax(self.tau * flat_scores, dim=1)
        return flat_prob.view(-1, self.num_classes, self.num_variations)

    def sac_loss(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Semantic Anchor Constraint: align proxy means with mean labeled embeddings.

        anchor is detached so gradients only update proxy_dist, not backbone/proj_head.
        """
        targets = self._resize_targets(targets, embeddings.shape[2:])
        flat_emb, flat_tgt = self._flatten_valid(embeddings, targets)
        if flat_emb.numel() == 0:
            return embeddings.sum() * 0.0
        x = F.normalize(flat_emb, p=2, dim=1)
        mu, _ = self._proxy_params()
        mu_norm = F.normalize(mu, p=2, dim=1)
        total = x.new_zeros(())
        count = 0
        for c in range(self.num_classes):
            mask = flat_tgt == c
            if mask.sum() == 0:
                continue
            anchor_c = F.normalize(x[mask].mean(0).detach(), p=2, dim=0)
            total = total + (1.0 - (mu_norm[c] * anchor_c).sum())
            count += 1
        return total / max(count, 1)

    def _compute_g(self, x: torch.Tensor) -> torch.Tensor:
        """[N, C] factorized score g(z,c) = rep_term + lambda_var * var_term."""
        mu, sigma = self._proxy_params()
        S = self.num_proxy_samples
        eps = torch.randn(S, self.num_classes, self.embedding_dim, device=x.device)
        samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # [S, C, D]
        samples = F.normalize(samples, p=2, dim=-1)
        rep_term = torch.einsum('nd,scd->nc', x, samples) / S  # [N, C]
        if self.use_variation and self.variation_active and self.variation_vectors is not None:
            variations = F.normalize(self.variation_vectors, p=2, dim=-1)  # [C, K, D]
            var_sims = torch.einsum('nd,ckd->nck', x, variations)         # [N, C, K]
            var_term = torch.logsumexp(self.tau_var_cdba * var_sims, dim=-1) / self.tau_var_cdba
            return rep_term + self.lambda_var * var_term
        return rep_term

    def forward_cdba(self, embeddings: torch.Tensor, targets=None) -> torch.Tensor:
        """CDBA: E2P + P2E.

        targets=None → unlabeled mode (all voxels, no label filtering).
        targets given → labeled mode (filter valid pixels, apply class balance).
        """
        if targets is not None:
            targets = self._resize_targets(targets, embeddings.shape[2:])
            flat_emb, flat_tgt = self._flatten_valid(embeddings, targets)
            if self.max_samples_per_class > 0:
                flat_emb, flat_tgt = self._balance_classes(flat_emb, flat_tgt)
        else:
            flat_emb = embeddings.movedim(1, -1).reshape(-1, embeddings.shape[1])
        if flat_emb.numel() == 0:
            return embeddings.sum() * 0.0
        x = F.normalize(flat_emb, p=2, dim=1)
        g = self._compute_g(x)               # [N, C]
        P = torch.softmax(g, dim=1)          # [N, C]
        loss_e2p = (P * (1.0 - g)).sum(dim=1).mean()
        E_diff = ((2.0 * P - 1.0) * g).mean(dim=0)  # [C]
        loss_p2e = torch.exp(-E_diff).mean()
        return loss_e2p + loss_p2e

    def _negative_probability(
        self, joint_prob: torch.Tensor, flat_targets: torch.Tensor
    ) -> torch.Tensor:
        if self.num_classes == 1:
            return torch.zeros_like(joint_prob[:, 0, 0])

        neg = joint_prob.clone()
        arange = torch.arange(flat_targets.numel(), device=flat_targets.device)
        neg[arange, flat_targets, :] = -torch.inf
        return neg.flatten(1).amax(dim=1)
