"""
Sharpness-Aware Minimization (SAM) optimizer for PatchTST.

Based on:
  - davda54/sam (https://github.com/davda54/sam)          — PyTorch SAM/ASAM
  - romilbert/samformer (https://github.com/romilbert/samformer) — SAMformer (ICML 2024)

References:
  [1] Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
      Generalization", ICLR 2021.
  [2] Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization for
      Scale-Invariant Learning of Deep Neural Networks", ICML 2021.
  [3] Ilbert et al., "SAMformer: Unlocking the Potential of Transformers in
      Time Series Forecasting with Sharpness-Aware Minimization and
      Channel-Wise Attention", ICML 2024.
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    SAM wraps any base optimizer (Adam, SGD, …) to perform sharpness-aware
    parameter updates.  Each training step requires **two** forward-backward
    passes:

        1. ``first_step``  — perturb weights toward the steepest ascent
           direction within a ρ-ball (find the adversarial neighbor).
        2. ``second_step`` — revert the perturbation and apply the base
           optimizer update using the gradient computed *at* the perturbed
           point.

    Args:
        params:          iterable of parameters or param-group dicts.
        base_optimizer:  optimizer **class** (e.g. ``torch.optim.Adam``).
        rho (float):     perturbation radius.  SAMformer uses 0.5–0.7 for
                         single-layer transformers; for deeper models like
                         PatchTST start with 0.1–0.3.
        adaptive (bool): if True, use element-wise adaptive SAM (ASAM) which
                         scales ε per parameter by |θ|², making the
                         perturbation scale-invariant.
        **kwargs:        forwarded to ``base_optimizer.__init__`` (lr, betas,
                         weight_decay, …).
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Instantiate the real optimizer on the *same* param_groups.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # Alias so that external code (LR schedulers, etc.) sees one set of
        # param_groups.
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    # ------------------------------------------------------------------
    # Step 1: ascend to the adversarial neighbor  θ̃ = θ + ε(θ)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Store a *clone* of the original weights so we can restore
                # them exactly in second_step (numerically safer than
                # subtracting ε back).
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # θ̃ = θ + ε

        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------
    # Step 2: revert perturbation, then do the real optimizer step using
    #         the gradient ∇L(θ̃).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # exact restore: θ̃ → θ

        self.base_optimizer.step()  # actual update with ∇L(θ̃)

        if zero_grad:
            self.zero_grad()

    # ------------------------------------------------------------------
    # Convenience: single call with a closure (à la LBFGS).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, (
            "SAM requires a closure that re-evaluates the model and returns "
            "the loss, but none was provided."
        )
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    # ------------------------------------------------------------------
    # Internal: compute ‖∇L‖₂ across all param groups.
    # ------------------------------------------------------------------
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                .norm(p=2)
                .to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    # ------------------------------------------------------------------
    # Checkpoint support: make sure base_optimizer stays in sync.
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ======================================================================
# BatchNorm running-statistics helpers
# ======================================================================
# PatchTST uses BatchNorm1d by default in its TSTEncoderLayer.  During
# SAM's second forward pass we must NOT update the running mean/var
# (they should reflect the *original* weights, not the perturbed ones).
#
# These helpers temporarily set BN momentum to 0 so that the running
# stats are frozen during the second pass.
#
# Reference: https://github.com/davda54/sam — "Training tips"
# ======================================================================

def enable_running_stats(model):
    """Restore normal BN behavior (update running stats)."""
    def _enable(module):
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.backup_momentum = module.momentum
            module.momentum = 0  # will be restored after first_step
    # First restore any previously backed-up momentum
    def _restore(module):
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if hasattr(module, 'backup_momentum'):
                module.momentum = module.backup_momentum
    model.apply(_restore)


def disable_running_stats(model):
    """Freeze BN running stats by setting momentum=0."""
    def _disable(module):
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            if not hasattr(module, 'backup_momentum'):
                module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)
