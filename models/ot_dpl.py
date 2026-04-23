"""
models/ot_dpl.py
──────────────────────────────────────────────────────────────
★ NOVELTY 1 — OT Dynamic Prompt Learner (OT-DPL)

Key idea:
  - Learn N prompts that explicitly encode different k-space
    undersampling patterns (R values) via cross-attention between
    encoder deep features Fd and learnable prompt tokens.
  - Optimal Transport (Sinkhorn) regularization forces each prompt
    to specialize to a distinct degradation region in latent space,
    maximizing inter-prompt discrimination without any R labels.

Components:
  1. Self-attention  : prompts interact with each other
  2. Cross-attention : prompts (Q) attend to encoder features (K, V)
  3. Sinkhorn OT     : compute ideal balanced assignment T*
  4. L_ot loss       : BCE( M, T* ) — enforces prompt specialization
  5. FFN             : refine updated prompts

Reference architecture: DaPT (Wei et al., TIP 2025) Section III-C
Extended with: Xavier init, per-layer residual norms, batched Sinkhorn

Fixes applied for AMP + torch.compile compatibility:
  - @torch._dynamo.disable on sinkhorn() and _ot_loss():
    Both contain Python-level loops with dynamic exit conditions
    (convergence check, batch for-loop) that torch.compile cannot
    trace. Decorated to run in eager mode safely.
  - Sinkhorn runs in FP32 for numerical stability; output cast back
    to input dtype so the graph stays in float16 under AMP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo


# ── Sinkhorn-Knopp OT Solver ──────────────────────────────────

@torch._dynamo.disable   # Python loop with dynamic convergence check — not compilable
def sinkhorn(
    M: "torch.Tensor",
    eps: float = 0.05,
    iters: int = 100,
    tol: float = 1e-4,
) -> "torch.Tensor":
    """
    Fast Sinkhorn-Knopp algorithm for Optimal Transport.

    Solves:  T* = argmax_{T ∈ Polytope} <T, M> + eps·H(T)
    where H(T) = -sum_ij T_ij log T_ij  (entropy smoothing)

    Constraints (uniform prior):
      T · 1      = mu  = 1/N · ones(N)   (each prompt gets equal mass)
      T^T · 1   = nu  = 1/HW · ones(HW) (each pixel distributes fully)

    Args:
        M   : [N, HW]  attention / similarity matrix (unnormalized logits)
        eps : entropy regularization strength  (0.05 recommended)
        iters: max Sinkhorn iterations
        tol : convergence threshold on u update

    Returns:
        T*  : [N, HW]  optimal transport plan (soft assignment)
    """
    N, HW = M.shape
    device = M.device

    # Stabilized log-domain Sinkhorn
    log_K = M / eps                              # [N, HW]

    mu = torch.full((N,),  1.0 / N,  device=device)   # row marginal
    nu = torch.full((HW,), 1.0 / HW, device=device)   # col marginal

    log_u = torch.zeros(N,  device=device)
    log_v = torch.zeros(HW, device=device)

    for _ in range(iters):
        log_u_prev = log_u.clone()
        # Row normalization
        log_u = torch.log(mu + 1e-8) - torch.logsumexp(
            log_K + log_v.unsqueeze(0), dim=1)
        # Col normalization
        log_v = torch.log(nu + 1e-8) - torch.logsumexp(
            log_K + log_u.unsqueeze(1), dim=0)
        # Convergence check
        if (log_u - log_u_prev).abs().max() < tol:
            break

    log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
    T = torch.exp(log_T)                         # [N, HW]
    return T.clamp(0.0, 1.0)


# ── Feed-Forward Network ──────────────────────────────────────

class PromptFFN(nn.Module):
    def __init__(self, C: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, C * expansion),
            nn.GELU(),
            nn.Linear(C * expansion, C),
        )

    def forward(self, x):
        return self.net(x)


# ── OT Dynamic Prompt Learner ─────────────────────────────────

class OT_DPL(nn.Module):
    """
    OT-regularized Dynamic Prompt Learner.

    Args:
        C       : channel dimension of encoder features
        N       : number of prompts (one per degradation cluster)
        eps     : Sinkhorn entropy regularization parameter
        iters   : Sinkhorn iterations
        n_layers: number of (self-attn → cross-attn → FFN) layers

    Forward inputs:
        Fd : [B, C, H, W]  deep features from encoder bottleneck

    Forward outputs:
        P_hat  : [B, N, C]  learned degradation-aware prompts
        L_ot   : scalar     OT regularization loss
    """

    def __init__(
        self,
        C: int,
        N: int = 5,
        eps: float = 0.05,
        iters: int = 100,
        n_layers: int = 1,
    ):
        super().__init__()
        self.N     = N
        self.eps   = eps
        self.iters = iters

        # Learnable initial prompts — Xavier init
        self.prompts = nn.Parameter(torch.empty(N, C))
        nn.init.xavier_uniform_(self.prompts.unsqueeze(0))

        # Self-attention projections (prompt ↔ prompt)
        self.sa_q = nn.Linear(C, C)
        self.sa_k = nn.Linear(C, C)
        self.sa_v = nn.Linear(C, C)
        self.sa_norm = nn.LayerNorm(C)

        # Cross-attention projections (prompt Q, feature K/V)
        self.WQ = nn.Linear(C, C)
        self.WK = nn.Linear(C, C)
        self.WV = nn.Linear(C, C)
        self.ca_norm = nn.LayerNorm(C)

        # Feed-forward
        self.ffn      = PromptFFN(C)
        self.ffn_norm = nn.LayerNorm(C)

        self.scale = C ** -0.5

    # ── Self-attention among prompts ──────────────────────────
    def _self_attn(self, P: torch.Tensor) -> torch.Tensor:
        """P: [B, N, C] → [B, N, C]"""
        Q = self.sa_q(P)
        K = self.sa_k(P)
        V = self.sa_v(P)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # [B, N, N]
        attn = attn.softmax(dim=-1)
        return self.sa_norm(P + torch.bmm(attn, V))

    # ── Cross-attention: prompts attend to encoder features ───
    def _cross_attn(
        self,
        P: torch.Tensor,
        Fd_flat: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        P       : [B, N, C]
        Fd_flat : [B, HW, C]
        Returns:
            P_updated : [B, N, C]
            M         : [B, N, HW]  attention weights (response map)
            raw_logits: [B, N, HW]  unnormalized (for OT)
        """
        Q = self.WQ(P)                                          # [B, N, C]
        K = self.WK(Fd_flat)                                    # [B, HW, C]
        V = self.WV(Fd_flat)                                    # [B, HW, C]

        logits = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, N, HW]
        M      = logits.softmax(dim=-1)                         # response map

        context   = torch.bmm(M, V)                            # [B, N, C]
        P_updated = self.ca_norm(P + context)
        return P_updated, M, logits

    # ── OT loss computation ───────────────────────────────────
    @torch._dynamo.disable   # batch for-loop — not compilable
    def _ot_loss(
        self,
        M: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each sample in batch:
          1. Compute T* via Sinkhorn on raw logits (detached)
          2. Compute BCE( M_b, T*_b )

        M      : [B, N, HW]
        logits : [B, N, HW]
        Returns scalar loss.

        Note: Sinkhorn runs in FP32 for numerical stability regardless
        of AMP dtype. Loss is cast back to M.dtype before returning
        so the overall graph stays in float16 under AMP.
        """
        B = M.shape[0]
        loss = torch.tensor(0.0, device=M.device)

        for b in range(B):
            # T* from logits (detached — OT is target, not part of grad)
            # Cast to float32 for Sinkhorn numerical stability
            T_star = sinkhorn(
                logits[b].detach().float(), self.eps, self.iters
            )                                                   # [N, HW]

            M_b    = M[b].float().clamp(1e-6, 1 - 1e-6)        # [N, HW] FP32
            T_star = T_star.clamp(0.0, 1.0)

            bce = -(T_star * torch.log(M_b)
                    + (1 - T_star) * torch.log(1 - M_b))
            loss = loss + bce.mean()

        # Cast back to input dtype (float16 under AMP, float32 otherwise)
        return (loss / B).to(M.dtype)

    # ── Forward ───────────────────────────────────────────────
    def forward(
        self,
        Fd: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Fd : [B, C, H, W]
        Returns:
            P_hat : [B, N, C]   degradation-aware prompts
            L_ot  : scalar      OT regularization loss
        """
        B, C, H, W = Fd.shape

        # Flatten spatial dims
        Fd_flat = Fd.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]

        # Expand initial prompts for batch
        P = self.prompts.unsqueeze(0).expand(B, -1, -1)         # [B, N, C]

        # Self-attention (prompts interact)
        P = self._self_attn(P)

        # Cross-attention (prompts attend to encoder features)
        P, M, logits = self._cross_attn(P, Fd_flat)

        # OT regularization loss
        L_ot = self._ot_loss(M, logits)

        # FFN refinement
        P_hat = self.ffn_norm(P + self.ffn(P))                  # [B, N, C]

        return P_hat, L_ot