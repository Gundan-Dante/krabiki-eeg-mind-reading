"""Stage 3 feasibility: trained nonlinear EEG encoders with InfoNCE against frozen video features (per subject).

Same inputs and protocol as the linear contrastive heads (``eegmr.contrastive``, M12): the EEG window is noise-whitened
per timepoint (Ledoit-Wolf on encoding OOF residuals) and standardized with training statistics; video features are
frozen and standardized with training-video statistics. Encoders (``Stage3Config.model``):

    linear    Linear(C*T -> D): the M12 "A" head re-implemented here, as the reference for the nonlinear models
    mlp       Linear(C*T -> h) -> GELU -> Dropout -> Linear(h -> D)
    resmlp    linear + mlp branches summed (the nonlinearity has to earn its keep on top of the linear map)
    eegnet    braindecode EEGNet on the (C, T) time course, output D
    shallow   braindecode ShallowFBCSPNet on the (C, T) time course, output D

Training: symmetric InfoNCE with a learnable temperature, full-batch AdamW, optional augmentation (time shift,
channel dropout, Gaussian noise). Two phases as in M12: early stopping on an inner split of the training videos
(retrieval score, step 0 included), then a refit on all training videos for the chosen number of steps.
:func:`cv_scores` runs the shared outer 5-fold train-CV with per-fold whiteners and the ridge decoder (M8a) on the
same folds as the reference; ``train_fraction`` subsamples each outer training part (learning curves).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn

from eegmr import gpu
from eegmr.contrastive import _row_z, expected_topk, mean_percentile_rank, prepare, ridge_weights
from eegmr.decoding import noise_whiteners

N_CHANS = 17


@dataclass(frozen=True)
class Stage3Config:
    """One encoder and its training hyperparameters."""

    model: str = "mlp"  # linear | mlp | resmlp | eegnet | shallow
    hidden: int = 512
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-2
    window: tuple[int, int] = (5, 60)  # samples at 100 Hz
    augment: bool = False
    shift: int = 2  # max time shift in samples (20 ms at 100 Hz)
    chan_drop: float = 0.1
    noise: float = 0.1
    max_steps: int = 2000
    eval_every: int = 10
    patience: int = 20
    inner_frac: float = 0.15
    tau_init: float = 0.07
    tau_min: float = 0.01
    tau_max: float = 1.0

    @property
    def name(self) -> str:
        arch = {"mlp": f"mlp{self.hidden}", "resmlp": f"resmlp{self.hidden}"}.get(self.model, self.model)
        drop = "" if self.model == "linear" else f"_do{self.dropout:g}"
        aug = "_aug" if self.augment else ""
        return f"{arch}{drop}_lr{self.lr:g}_wd{self.weight_decay:g}_w{self.window[0]}-{self.window[1]}{aug}"


class _Flat(nn.Module):
    """Wrap a model on flattened input so that every encoder takes ``(n, C, T)``."""

    def __init__(self, net: nn.Module, skip: nn.Module | None = None) -> None:
        super().__init__()
        self.net, self.skip = net, skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.net(x) + (self.skip(x) if self.skip is not None else 0)


class Encoder(nn.Module):
    """EEG encoder plus the learnable InfoNCE temperature."""

    def __init__(self, cfg: Stage3Config, n_times: int, out_dim: int) -> None:
        super().__init__()
        d_in = N_CHANS * n_times
        if cfg.model == "linear":
            self.net = _Flat(nn.Linear(d_in, out_dim))
        elif cfg.model in ("mlp", "resmlp"):
            mlp = nn.Sequential(nn.Linear(d_in, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
                                nn.Linear(cfg.hidden, out_dim))
            self.net = _Flat(mlp, nn.Linear(d_in, out_dim) if cfg.model == "resmlp" else None)
        elif cfg.model == "eegnet":
            from braindecode.models import EEGNet

            self.net = EEGNet(n_chans=N_CHANS, n_outputs=out_dim, n_times=n_times, kernel_length=25, F1=8, D=2,
                              F2=16, pool_mode="mean", drop_prob=cfg.dropout)
        elif cfg.model == "shallow":
            from braindecode.models import ShallowFBCSPNet

            self.net = ShallowFBCSPNet(n_chans=N_CHANS, n_outputs=out_dim, n_times=n_times, n_filters_time=20,
                                       n_filters_spat=20, filter_time_length=13, pool_time_length=15,
                                       pool_time_stride=5, final_conv_length="auto", drop_prob=cfg.dropout)
        else:
            raise ValueError(f"unknown model {cfg.model!r}")
        self.log_tau = nn.Parameter(torch.tensor(float(np.log(cfg.tau_init))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.net(x), dim=-1)

    @torch.no_grad()
    def similarity(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Cosine similarity ``(n_eeg, n_videos)`` (eval mode)."""
        was_training = self.training
        self.eval()
        sim = self(X) @ torch.nn.functional.normalize(V, dim=-1).T
        self.train(was_training)
        return sim


def augment(x: torch.Tensor, cfg: Stage3Config, gen: torch.Generator) -> torch.Tensor:
    """Random circular time shift (±``shift`` samples), channel dropout and Gaussian noise, per trial."""
    n, c, t = x.shape
    if cfg.shift:
        shifts = torch.randint(-cfg.shift, cfg.shift + 1, (n,), generator=gen).to(x.device)
        idx = (torch.arange(t, device=x.device)[None, :] - shifts[:, None]) % t
        x = torch.gather(x, 2, idx[:, None, :].expand(n, c, t))
    if cfg.chan_drop:
        keep = (torch.rand(n, c, 1, generator=gen) >= cfg.chan_drop).to(x.device, x.dtype)
        x = x * keep / (1 - cfg.chan_drop)
    if cfg.noise:
        x = x + cfg.noise * torch.randn(x.shape, generator=gen).to(x.device)
    return x


def _optimize(X: torch.Tensor, V: torch.Tensor, cfg: Stage3Config, seed: int, n_steps: int,
              X_eval: torch.Tensor | None = None, V_eval: torch.Tensor | None = None) -> tuple[Encoder, int, float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    model = Encoder(cfg, X.shape[-1], V.shape[1]).to(X.device)
    decay = [p for n, p in model.named_parameters() if n != "log_tau"]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": [model.log_tau], "weight_decay": 0.0}], lr=cfg.lr)
    v = torch.nn.functional.normalize(V, dim=-1)
    target = torch.arange(len(X), device=X.device)
    best, best_state, best_step, bad = float("inf"), None, 0, 0
    if X_eval is not None:
        best = mean_percentile_rank(model.similarity(X_eval, V_eval))
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.train()
    for step in range(1, n_steps + 1):
        x = augment(X, cfg, gen) if cfg.augment else X
        logits = (model(x) @ v.T) / model.log_tau.exp()
        loss = 0.5 * (torch.nn.functional.cross_entropy(logits, target)
                      + torch.nn.functional.cross_entropy(logits.T, target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.log_tau.clamp_(float(np.log(cfg.tau_min)), float(np.log(cfg.tau_max)))
        if X_eval is not None and step % cfg.eval_every == 0:
            score = mean_percentile_rank(model.similarity(X_eval, V_eval))
            if score < best - 1e-6:
                best, best_step, bad = score, step, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, (best_step if X_eval is not None else n_steps), best


def train_encoder(X: torch.Tensor, V: torch.Tensor, cfg: Stage3Config, seed: int = 0) -> Encoder:
    """Two-phase training: early stopping on an inner split, then refit on all videos for the chosen step count."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=gen).to(X.device)
    n_inner = max(2, int(round(cfg.inner_frac * len(X))))
    inner, fit = perm[:n_inner], perm[n_inner:]
    _, best_steps, inner_score = _optimize(X[fit], V[fit], cfg, seed, cfg.max_steps, X[inner], V[inner])
    model, _, _ = _optimize(X, V, cfg, seed, best_steps)
    model.steps, model.inner_score = best_steps, inner_score
    return model


def inputs(eeg_trn: np.ndarray, eeg_new: np.ndarray, F_trn: np.ndarray, F_new: np.ndarray, whiteners: torch.Tensor,
           window: slice, device: str) -> tuple[torch.Tensor, ...]:
    """Whitened, standardized EEG ``(n, C, T)`` and standardized features, from training statistics."""
    X_trn, X_new, V_trn, V_new = prepare(eeg_trn, eeg_new, F_trn, F_new, whiteners, window, device)
    t = window.stop - window.start
    return X_trn.view(len(X_trn), N_CHANS, t), X_new.view(len(X_new), N_CHANS, t), V_trn, V_new


def cv_scores(
    features: np.ndarray,
    eeg: np.ndarray,
    configs: list[Stage3Config],
    enc_alpha: float,
    dec_alpha: float,
    ridge_window: slice = slice(5, 60),
    n_folds: int = 5,
    seed: int = 0,
    device: str | None = None,
    ensemble_weights: tuple[float, ...] = (),
    train_fraction: float = 1.0,
) -> list[dict]:
    """Outer k-fold train-CV of every config, plus the ridge decoder (``ridge``, = M8a) on the same folds and the
    blends ``w * rowz(ridge) + (1 - w) * rowz(model)`` (``ens<w>_<config>``). With ``train_fraction`` < 1, every model
    (ridge included) is fitted on a seeded random subset of each outer training part; the held-out folds don't change."""
    device = device or gpu.default_device()
    ranks: dict[str, list[tuple[np.ndarray, int]]] = {}
    steps: dict[str, list[int]] = {}
    rng = np.random.default_rng(seed)
    for trn, val in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        if train_fraction < 1.0:
            trn = np.sort(rng.choice(trn, int(round(train_fraction * len(trn))), replace=False))
        whiteners: dict[tuple[int, int], torch.Tensor] = {}

        def fold_inputs(w: tuple[int, int]) -> tuple[torch.Tensor, ...]:
            if w not in whiteners:
                whiteners[w] = noise_whiteners(features[trn], eeg[trn], enc_alpha, slice(*w), device=device)
            return inputs(eeg[trn], eeg[val], features[trn], features[val], whiteners[w], slice(*w), device)

        X_trn, X_val, V_trn, V_val = fold_inputs((ridge_window.start, ridge_window.stop))
        W0 = ridge_weights(X_trn.flatten(1), V_trn, dec_alpha)
        sims = {"ridge": torch.nn.functional.normalize((X_val.flatten(1).double() @ W0).float(), dim=-1)
                @ torch.nn.functional.normalize(V_val, dim=-1).T}
        for cfg in configs:
            X_trn, X_val, V_trn, V_val = fold_inputs(cfg.window)
            model = train_encoder(X_trn, V_trn, cfg, seed=seed)
            sims[cfg.name] = model.similarity(X_val, V_val)
            steps.setdefault(cfg.name, []).append(model.steps)
            for w in ensemble_weights:
                sims[f"ens{w:g}_{cfg.name}"] = w * _row_z(sims["ridge"]) + (1 - w) * _row_z(sims[cfg.name])
            del model
        for name, sim in sims.items():
            ranks.setdefault(name, []).append(((sim >= sim.diagonal()[:, None]).sum(dim=1).cpu().numpy(), len(val)))
        del sims
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    rows = []
    by_name = {c.name: c for c in configs}
    for name, parts in ranks.items():
        all_pr = np.concatenate([(r - 1) / (m - 1) for r, m in parts])
        weights = np.array([len(r) for r, _ in parts], dtype=float)
        cfg = by_name.get(name)
        rows.append({"config": name, "cv_score": float(all_pr.mean()),
                     "exp_top1": float(np.average([expected_topk(r, m, 1) for r, m in parts], weights=weights)),
                     "exp_top5": float(np.average([expected_topk(r, m, 5) for r, m in parts], weights=weights)),
                     "mean_steps": float(np.mean(steps[name])) if name in steps else 0.0,
                     "train_fraction": train_fraction,
                     **({k: str(v) for k, v in asdict(cfg).items()} if cfg else {})})
    return rows
