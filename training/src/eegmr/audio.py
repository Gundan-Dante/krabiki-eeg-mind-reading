"""Audio features for the 1,102 clips (track-audio), cached in the same format as :mod:`eegmr.features`.

Model (public, released before 2026-10-16):
    clap  LAION-CLAP ``laion/clap-htsat-unfused`` (Apache-2.0, 2023), HTS-AT audio tower via transformers,
          ``get_audio_features`` (512-d, projected). Trained on LAION-Audio-630K; verify no Moments in Time on the card.

Cache: ``features/<name>_all_f1.npz`` with ``ids`` (1102,), ``mean`` (1102, D) = embedding of the whole clip,
``frames`` (1102, S, D) float16 = embeddings of S consecutive 1 s segments (for ``--pooling`` / time diagnostics), and
``no_audio`` = IDs without a usable audio track (no stream, or RMS below ``SILENT_RMS``). Those rows get the mean
embedding of the audible *training* clips, so they carry no information and don't bias any candidate.
``load_features("clap_all", 1, ids)`` works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
from tqdm import tqdm

from eegmr.data import ALL_IDS, DEFAULT_ROOT, TRAIN_IDS, video_path
from eegmr.features import DEFAULT_FEATURE_DIR, feature_path

CLAP_REPO = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
CLAP_SR = 48_000
CLIP_SECONDS = 3.0
SEGMENT_SECONDS = 1.0
SILENT_RMS = 1e-4


def load_audio(video_id: int, sr: int, root: Path = DEFAULT_ROOT) -> np.ndarray | None:
    """Mono float32 waveform at ``sr`` Hz, cut to the 3 s clip; ``None`` if the video has no audio stream."""
    # some clips carry non-UTF-8 metadata tags; they are irrelevant to the audio, so ignore decoding errors there
    with av.open(str(video_path(video_id, root)), metadata_errors="ignore") as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = [f.to_ndarray().reshape(-1) for frame in container.decode(streams[0]) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]  # flush
    y = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return y[: int(CLIP_SECONDS * sr)]


def _clap(device: str):
    from transformers import ClapFeatureExtractor, ClapModel

    model = ClapModel.from_pretrained(CLAP_REPO, revision=CLAP_REVISION).eval().to(device)
    fe = ClapFeatureExtractor.from_pretrained(CLAP_REPO, revision=CLAP_REVISION)

    @torch.no_grad()
    def run(waves: list[np.ndarray]) -> np.ndarray:
        inp = fe(waves, sampling_rate=CLAP_SR, return_tensors="pt").to(device)
        out = model.get_audio_features(**inp)
        out = out if isinstance(out, torch.Tensor) else out.pooler_output
        return out.float().cpu().numpy()

    return run


AUDIO_ENCODERS = {"clap": (_clap, CLAP_SR)}


# ---------------------------------------------------------------------------------------------------------------
# Time-resolved low-level audio-visual features (no pretrained model). Per 100 ms bin of the 3 s clip:
#   visual: luminance, contrast (pixel std), edge energy (Sobel), motion (mean |frame - previous frame|), grayscale [0, 1]
#   audio:  RMS loudness, spectral-flux onset strength and 8 log10-mel band energies (librosa), 16 kHz mono; clips
#           without audio get 0 loudness / onset and the log-power floor (-10) in the mel bands. The mel bands capture background texture (rumble / hiss / tonal) that RMS and onsets miss.
# Frames are binned by timestamp (fps varies 24-30 between clips). Caches (ids, mean = flattened bins, frames = bins):
#   lowvis_all_f1 (30 x 4), lowaud_all_f1 (30 x 10), lowav_all_f1 (30 x 14).
# ---------------------------------------------------------------------------------------------------------------

LOW_BIN_S = 0.1
LOW_SR = 16_000
LOW_N_BINS = int(round(CLIP_SECONDS / LOW_BIN_S))
LOW_N_MELS = 8


def _bin_mean(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Average rows of ``values`` into LOW_N_BINS time bins; an empty bin copies the previous one."""
    b = np.minimum((times / LOW_BIN_S).astype(int), LOW_N_BINS - 1)
    out = np.zeros((LOW_N_BINS, values.shape[1]), np.float32)
    for k in range(LOW_N_BINS):
        out[k] = values[b == k].mean(0) if (b == k).any() else out[k - 1]
    return out


def lowlevel_visual(video_id: int, root: Path = DEFAULT_ROOT) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(video_path(video_id, root)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        edges = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
        rows.append([g.mean(), g.std(), edges, 0.0 if prev is None else np.abs(g - prev).mean()])
        prev = g
    cap.release()
    out = _bin_mean(np.arange(len(rows)) / fps, np.array(rows, np.float32))
    out[0, 3] = out[1, 3]  # motion of the first frame is undefined
    return out


def lowlevel_audio(video_id: int, root: Path = DEFAULT_ROOT) -> np.ndarray:
    import librosa

    y = load_audio(video_id, LOW_SR, root)
    if y is None or len(y) == 0 or np.sqrt(np.mean(y ** 2)) < SILENT_RMS:
        out = np.zeros((LOW_N_BINS, 2 + LOW_N_MELS), np.float32)
        out[:, 2:] = -10.0  # silence = the log-power floor (log10 1e-10), not 0 dB
        return out
    hop = LOW_SR // 100  # 10 ms
    rms = librosa.feature.rms(y=y, frame_length=4 * hop, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=LOW_SR, hop_length=hop)
    # spectral texture (background rumble / hiss / tonal content): log energy in a few mel bands
    mel = np.log10(librosa.feature.melspectrogram(y=y, sr=LOW_SR, n_fft=4 * hop, hop_length=hop,
                                                  n_mels=LOW_N_MELS, fmax=LOW_SR // 2) + 1e-10)
    n = min(len(rms), len(onset), mel.shape[1])
    return _bin_mean(np.arange(n) * hop / LOW_SR, np.concatenate([rms[None, :n], onset[None, :n], mel[:, :n]]).T)


def extract_lowlevel_av(root: Path = DEFAULT_ROOT, feature_dir: Path = DEFAULT_FEATURE_DIR,
                        ids: np.ndarray = ALL_IDS) -> list[Path]:
    vis = np.stack([lowlevel_visual(int(v), root) for v in tqdm(ids, desc="low-level visual")])
    aud = np.stack([lowlevel_audio(int(v), root) for v in tqdm(ids, desc="low-level audio")])
    outs = []
    for name, x in [("lowvis", vis), ("lowaud", aud), ("lowav", np.concatenate([vis, aud], axis=2))]:
        out = feature_path(f"{name}_all", 1, feature_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, ids=ids, mean=x.reshape(len(ids), -1).astype(np.float32), frames=x.astype(np.float16))
        outs.append(out)
    return outs


def extract_audio_features(
    name: str = "clap",
    root: Path = DEFAULT_ROOT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ids: np.ndarray = ALL_IDS,
) -> Path:
    """Embed the whole clip and its 1 s segments for every video and cache ``<name>_all_f1.npz``."""
    make, sr = AUDIO_ENCODERS[name]
    encode = make(device)
    n_seg = int(round(CLIP_SECONDS / SEGMENT_SECONDS))
    seg_len = int(SEGMENT_SECONDS * sr)
    means, segs, silent = [], [], []
    for vid in tqdm(ids, desc=f"{name} audio"):
        y = load_audio(int(vid), sr, root)
        if y is None or len(y) < seg_len or np.sqrt(np.mean(y ** 2)) < SILENT_RMS:
            silent.append(int(vid))
            means.append(None)
            segs.append(None)
            continue
        parts = [y[k * seg_len:(k + 1) * seg_len] for k in range(n_seg)]
        parts = [p if len(p) == seg_len else np.pad(p, (0, seg_len - len(p))) for p in parts]
        emb = encode([y] + parts)
        means.append(emb[0])
        segs.append(emb[1:])
    # rows without audio: mean of the audible training clips (uninformative, no candidate favoured)
    audible_train = [i for i, v in enumerate(ids) if v in set(TRAIN_IDS.tolist()) and means[i] is not None]
    fill_mean = np.mean([means[i] for i in audible_train], axis=0)
    fill_seg = np.mean([segs[i] for i in audible_train], axis=0)
    means = np.stack([fill_mean if m is None else m for m in means]).astype(np.float32)
    segs = np.stack([fill_seg if s is None else s for s in segs]).astype(np.float16)
    out = feature_path(f"{name}_all", 1, feature_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ids, mean=means, frames=segs, no_audio=np.array(silent, dtype=int))
    return out
