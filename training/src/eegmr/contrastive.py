"""Linear contrastive heads: InfoNCE on frozen video features, with the same EEG input as the whitened decoder (M8a).

Inputs per subject (see :func:`prepare`): the EEG window (channels x time, per-timepoint noise-whitened as in
``eegmr.decoding``) flattened and standardized with training statistics, and the frozen video features standardized
with training-video statistics. Heads (all trained with symmetric InfoNCE and a learnable temperature):

    linear      W: EEG -> video dim, cosine to the frozen video vector (the contrastive twin of ridge decoding)
    two_tower   W_e: EEG -> d and W_v: video -> d, cosine in the shared d-dim space
    warm        ``linear`` initialized at the ridge decoder (alpha given), optionally with decay toward that solution

Training is full-batch AdamW with early stopping on an inner split of the training videos, scored by retrieval
(mean percentile rank among the inner videos). Nothing here sees test data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.stats import hypergeom
from sklearn.model_selection import KFold

from eegmr import gpu
from eegmr.decoding import decoder_input, noise_whiteners


@dataclass(frozen=True)
class ContrastiveConfig:
    """One contrastive head and its training hyperparameters."""

    kind: str = "linear"  # linear | two_tower | warm
    dim: int = 128  # two_tower: shared dimension
    lr: float = 1e-3
    weight_decay: float = 1e-3
    anchor: bool = False  # warm: decay toward the ridge weights instead of toward 0
    max_steps: int = 1500
    eval_every: int = 10
    patience: int = 15  # evaluations without improvement before stopping
    inner_frac: float = 0.15
    tau_init: float = 0.07
    tau_min: float = 0.01
    tau_max: float = 1.0

    @property
    def name(self) -> str:
        extra = f"_d{self.dim}" if self.kind == "two_tower" else ("_anchor" if self.anchor else "")
        return f"{self.kind}{extra}_lr{self.lr:g}_wd{self.weight_decay:g}"


# CV-best config per variant on CLIP B-32 (M12; results/contrastive_cv_clip_b32.csv). The final model uses two_tower_d256.
SELECTED_CONFIGS = {"warm": ContrastiveConfig("warm", lr=3e-4, weight_decay=1e-3, anchor=True),
                    "linear": ContrastiveConfig("linear", lr=1e-3, weight_decay=1e-3),
                    "two_tower_d256": ContrastiveConfig("two_tower", dim=256, lr=1e-3, weight_decay=1e-2)}


@dataclass
class ContrastiveHead:
    """A trained head; :meth:`similarity` gives cosine similarities ``(n_eeg, n_videos)``."""

    W_e: torch.Tensor
    W_v: torch.Tensor | None
    log_tau: torch.Tensor
    steps: int
    best_inner_score: float

    def embed(self, X: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.nn.functional.normalize(X @ self.W_e, dim=-1)
        v = V @ self.W_v if self.W_v is not None else V
        return e, torch.nn.functional.normalize(v, dim=-1)

    @torch.no_grad()
    def similarity(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        e, v = self.embed(X, V)
        return e @ v.T


def prepare(eeg_trn: np.ndarray, eeg_new: np.ndarray, F_trn: np.ndarray, F_new: np.ndarray,
            whiteners: torch.Tensor | None, window: slice, device: str) -> tuple[torch.Tensor, ...]:
    """Whitened, standardized EEG inputs and standardized video features (training statistics), float32 on device."""
    X_trn = decoder_input(eeg_trn, window, whiteners, device)
    X_new = decoder_input(eeg_new, window, whiteners, device)
    F_trn_t, F_new_t = gpu.as_tensor(F_trn, device, torch.float64), gpu.as_tensor(F_new, device, torch.float64)
    return (gpu.standardize(X_trn, X_trn).float(), gpu.standardize(X_new, X_trn).float(),
            gpu.standardize(F_trn_t, F_trn_t).float(), gpu.standardize(F_new_t, F_trn_t).float())


def ridge_weights(X: torch.Tensor, V: torch.Tensor, alpha: float) -> torch.Tensor:
    """Ridge solution ``(D_eeg, D_video)`` on standardized inputs and targets (no intercept needed: both are centred)."""
    U, S, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    return (Vh.T * (S / (S**2 + alpha))) @ (U.T @ V.double())


def mean_percentile_rank(sim: torch.Tensor) -> float:
    """Mean (rank of the diagonal - 1) / (n - 1) over rows; 0 = perfect, 0.5 = chance."""
    ranks = (sim >= sim.diagonal()[:, None]).sum(dim=1)
    return float(((ranks - 1) / (len(sim) - 1)).double().mean())


def expected_topk(ranks: np.ndarray, n_pool: int, k: int, n_candidates: int = 102) -> float:
    """Expected top-k accuracy among ``n_candidates`` drawn from a pool of ``n_pool`` (true video included), given each
    trial's 1-based rank in the full pool (hypergeometric)."""
    better = ranks - 1  # pool items ranked above the true video
    return float(np.mean(hypergeom.cdf(k - 1, n_pool - 1, better, n_candidates - 1)))


def _init(X: torch.Tensor, V: torch.Tensor, cfg: ContrastiveConfig, gen: torch.Generator,
          W0: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    dev = X.device
    if cfg.kind == "warm":
        W_e = W0.float().clone()
    else:
        out = V.shape[1] if cfg.kind == "linear" else cfg.dim
        W_e = (torch.randn(X.shape[1], out, generator=gen) * X.shape[1] ** -0.5).to(dev)
    W_v = (torch.randn(V.shape[1], cfg.dim, generator=gen) * V.shape[1] ** -0.5).to(dev) if cfg.kind == "two_tower" else None
    return W_e.requires_grad_(True), None if W_v is None else W_v.requires_grad_(True), \
        torch.tensor(np.log(cfg.tau_init), device=dev, requires_grad=True)


def _optimize(X: torch.Tensor, V: torch.Tensor, cfg: ContrastiveConfig, gen: torch.Generator, W0: torch.Tensor | None,
              n_steps: int, X_eval: torch.Tensor | None = None, V_eval: torch.Tensor | None = None
              ) -> tuple[ContrastiveHead, int, float]:
    """Run AdamW for up to ``n_steps``. With an eval set, early-stop on its retrieval score (step 0 included) and
    return the best state; without one, return the state after ``n_steps``."""
    W_e, W_v, log_tau = _init(X, V, cfg, gen, W0)
    head = ContrastiveHead(W_e, W_v, log_tau, 0, float("inf"))
    params = [W_e] + ([W_v] if W_v is not None else [])
    opt = torch.optim.AdamW([{"params": params, "weight_decay": 0.0 if cfg.anchor else cfg.weight_decay},
                             {"params": [log_tau], "weight_decay": 0.0}], lr=cfg.lr)
    anchor = W0.float() if cfg.anchor else None
    target = torch.arange(len(X), device=X.device)

    def snapshot(step: int) -> tuple:
        return W_e.detach().clone(), None if W_v is None else W_v.detach().clone(), log_tau.detach().clone(), step

    best, best_state, bad = float("inf"), snapshot(0), 0
    if X_eval is not None:
        best = mean_percentile_rank(head.similarity(X_eval, V_eval))
    for step in range(1, n_steps + 1):
        e, v = head.embed(X, V)
        logits = (e @ v.T) / log_tau.exp()
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, target)
                      + torch.nn.functional.cross_entropy(logits.T, target))
        if anchor is not None:
            loss = loss + cfg.weight_decay * ((W_e - anchor) ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_tau.clamp_(np.log(cfg.tau_min), np.log(cfg.tau_max))
        if X_eval is not None and step % cfg.eval_every == 0:
            score = mean_percentile_rank(head.similarity(X_eval, V_eval))
            if score < best - 1e-6:
                best, best_state, bad = score, snapshot(step), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if X_eval is None:
        best_state = snapshot(n_steps)
    W_e_b, W_v_b, tau_b, step_b = best_state
    return ContrastiveHead(W_e_b, W_v_b, tau_b, step_b, best), step_b, best


def train_head(X: torch.Tensor, V: torch.Tensor, cfg: ContrastiveConfig, ridge_alpha: float, seed: int = 0
               ) -> ContrastiveHead:
    """Two-phase training. (1) Hold out ``inner_frac`` of the training videos, train on the rest (warm start: ridge
    fitted on the rest only) and find the best step count by retrieval on the held-out part (step 0 included).
    (2) Retrain on all videos (warm start: ridge on all) for that many steps."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=gen).to(X.device)
    n_inner = max(2, int(round(cfg.inner_frac * len(X))))
    inner, fit = perm[:n_inner], perm[n_inner:]
    W0_fit = ridge_weights(X[fit], V[fit], ridge_alpha) if cfg.kind == "warm" else None
    _, best_steps, inner_score = _optimize(X[fit], V[fit], cfg, gen, W0_fit, cfg.max_steps, X[inner], V[inner])
    W0 = ridge_weights(X, V, ridge_alpha) if cfg.kind == "warm" else None
    head, _, _ = _optimize(X, V, cfg, gen, W0, best_steps)
    head.steps, head.best_inner_score = best_steps, inner_score
    return head


def _row_z(sim: torch.Tensor) -> torch.Tensor:
    return (sim - sim.mean(dim=1, keepdim=True)) / sim.std(dim=1, keepdim=True)


def cv_scores(
    features: np.ndarray,
    eeg: np.ndarray,
    configs: list[ContrastiveConfig],
    enc_alpha: float,
    dec_alpha: float,
    window: slice,
    n_folds: int = 5,
    seed: int = 0,
    device: str | None = None,
    ensemble_weights: tuple[float, ...] = (),
) -> list[dict]:
    """Outer k-fold train-CV of every config: one row per config with mean percentile rank and expected top-1 / top-5
    of 102. Whiteners, standardization and the ridge warm start are re-estimated inside each outer fold. Also scores
    the ridge decoder itself (``ridge``, = M8a) on the same folds as the reference, and for each weight w in
    ``ensemble_weights`` the blend ``w * rowz(ridge) + (1 - w) * rowz(head)`` (``ens<w>_<config>``)."""
    device = device or gpu.default_device()
    pr = {c.name: [] for c in configs} | {"ridge": []} | {f"ens{w:g}_{c.name}": [] for c in configs for w in ensemble_weights}
    ranks: dict[str, list[tuple[np.ndarray, int]]] = {k: [] for k in pr}
    steps: dict[str, list[int]] = {c.name: [] for c in configs}
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        W = noise_whiteners(features[trn], eeg[trn], enc_alpha, window, device=device)
        X_trn, X_val, V_trn, V_val = prepare(eeg[trn], eeg[val], features[trn], features[val], W, window, device)
        W0 = ridge_weights(X_trn, V_trn, dec_alpha)
        sims = {"ridge": torch.nn.functional.normalize((X_val.double() @ W0).float(), dim=-1)
                @ torch.nn.functional.normalize(V_val, dim=-1).T}
        for cfg in configs:
            head = train_head(X_trn, V_trn, cfg, dec_alpha, seed=seed)
            sims[cfg.name] = head.similarity(X_val, V_val)
            steps[cfg.name].append(head.steps)
            for w in ensemble_weights:
                sims[f"ens{w:g}_{cfg.name}"] = w * _row_z(sims["ridge"]) + (1 - w) * _row_z(sims[cfg.name])
        for name, sim in sims.items():
            r = (sim >= sim.diagonal()[:, None]).sum(dim=1).cpu().numpy()
            ranks[name].append((r, len(val)))
        del X_trn, X_val, V_trn, V_val, W0, sims
    rows = []
    for name, parts in ranks.items():
        all_pr = np.concatenate([(r - 1) / (m - 1) for r, m in parts])
        cfg = next((c for c in configs if c.name == name), None)
        weights = np.array([len(r) for r, _ in parts], dtype=float)
        rows.append({"config": name, "cv_score": float(all_pr.mean()),
                     "exp_top1": float(np.average([expected_topk(r, m, 1) for r, m in parts], weights=weights)),
                     "exp_top5": float(np.average([expected_topk(r, m, 5) for r, m in parts], weights=weights)),
                     "mean_steps": float(np.mean(steps[name])) if name in steps else 0.0,
                     **({k: v for k, v in asdict(cfg).items()} if cfg else {"kind": "ridge"})})
    return rows
