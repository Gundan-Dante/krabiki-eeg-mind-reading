"""External (non-challenge) EEG data: THINGS-EEG2 preprocessed release (Gifford et al., 2022; OSF 3jk45 / anp5v).

The downloaded files are UNTRUSTED. The preprocessed ``.npy`` files are pickled dicts, and loading a pickle can execute code.
This module is therefore self-contained (numpy, scipy, zipfile, pickle only), so the converter can run in isolated mode,

    python -I src/eegmr/external.py convert <zip dir> <out dir> --subjects 1 2

and it unpickles with a whitelist (numpy array reconstruction, dict/list only; anything else is refused). The converter writes
plain float32 ``.npz`` files that all later code loads with ``allow_pickle=False`` (:func:`load_converted`).
"""

from __future__ import annotations

import argparse
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np

# Our 17 channels, in order, as given by the organizers (the same set as THINGS-EEG2's 17 posterior channels).
OUR_CHANNELS = "Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2".split()
THINGS_SFREQ = 100  # Hz, Gifford release
THINGS_T0 = -0.2  # s, first sample of an epoch (epochs run -200..800 ms)

_ALLOWED = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"), ("builtins", "set"),
    ("collections", "OrderedDict"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # noqa: D102
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused to unpickle {module}.{name}")


def load_pickled(path: Path | str) -> dict:
    """Load a dict stored either as an object ``.npy`` (``np.save`` of a dict) or as a raw pickle, with a class whitelist."""
    with open(path, "rb") as fh:
        head = fh.read(6)
        fh.seek(0)
        if head == b"\x93NUMPY":
            version = np.lib.format.read_magic(fh)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            reader(fh)  # shape, fortran_order, dtype: we only need the stream position
        obj = _RestrictedUnpickler(fh).load()
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    assert isinstance(obj, dict), f"expected a dict, got {type(obj)}"
    return obj


def _safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    for info in zf.infolist():
        name = info.filename
        assert not name.startswith("/") and ".." not in Path(name).parts, f"unsafe zip member {name!r}"
    return zf.infolist()


def convert_subject(zip_path: Path, out_path: Path, work_dir: Path) -> None:
    """Extract one ``sub-XX.zip``, load its two pickled dicts once with the whitelist and re-save plain float32 arrays.

    Output ``.npz`` keys: ``train`` (conditions, reps, channels, time), ``test`` (same), ``ch_names`` (unicode), ``times``.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = _safe_members(zf)
        wanted = [m for m in members if m.filename.endswith(("preprocessed_eeg_training.npy", "preprocessed_eeg_test.npy"))]
        assert len(wanted) == 2, f"unexpected zip content: {[m.filename for m in members][:10]}"
        for m in wanted:
            zf.extract(m, work_dir)
    out = {}
    for key, fname in (("train", "preprocessed_eeg_training.npy"), ("test", "preprocessed_eeg_test.npy")):
        path = next(work_dir.rglob(fname))
        d = load_pickled(path)
        out[key] = np.asarray(d["preprocessed_eeg_data"], dtype=np.float32)
        out["ch_names"] = np.array([str(c) for c in d["ch_names"]])
        out["times"] = np.asarray(d["times"], dtype=np.float64)
        path.unlink()
    np.savez(out_path, **out)


def load_converted(path: Path | str) -> dict[str, np.ndarray]:
    """Load a converted ``.npz`` (plain arrays only)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def things_to_ours(eeg: np.ndarray, ch_names: np.ndarray, window: slice = slice(25, 80), lowpass: float = 40.0,
                   sfreq: int = THINGS_SFREQ) -> np.ndarray:
    """Align THINGS-EEG2 epochs ``(..., 17, 100)`` with our recordings: channels reordered by name to :data:`OUR_CHANNELS`,
    zero-phase low-pass at ``lowpass`` Hz (our data are band-passed to 0.1-40 Hz), then the matching window cropped.

    Epoch sample ``i`` is at ``-0.2 + i / 100`` s, so samples 25:80 are 50-600 ms, as our samples 5:60 at 100 Hz."""
    from scipy.signal import butter, sosfiltfilt

    order = [list(ch_names).index(c) for c in OUR_CHANNELS]
    x = eeg[..., order, :]
    x = sosfiltfilt(butter(4, lowpass, btype="low", fs=sfreq, output="sos"), x, axis=-1)
    return np.ascontiguousarray(x[..., window].astype(np.float32))


def sha256(path: Path | str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert downloaded sub-XX.zip files into plain float32 .npz (run with python -I)")
    c.add_argument("zip_dir", type=Path)
    c.add_argument("out_dir", type=Path)
    c.add_argument("--subjects", type=int, nargs="+", required=True)
    m = sub.add_parser("meta", help="convert image_metadata.npy into a plain .npz of unicode arrays")
    m.add_argument("meta_file", type=Path)
    m.add_argument("out_file", type=Path)
    args = p.parse_args()
    if args.cmd == "convert":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for s in args.subjects:
            work = args.out_dir / f"_work_sub-{s:02d}"
            work.mkdir(exist_ok=True)
            convert_subject(args.zip_dir / f"sub-{s:02d}.zip", args.out_dir / f"sub-{s:02d}.npz", work)
            for f in sorted(work.rglob("*"), reverse=True):
                f.rmdir() if f.is_dir() else f.unlink()
            work.rmdir()
            z = load_converted(args.out_dir / f"sub-{s:02d}.npz")
            print(f"sub-{s:02d}: train {z['train'].shape} {z['train'].dtype}, test {z['test'].shape}, "
                  f"channels {list(z['ch_names'])}, times {z['times'][0]:.2f}..{z['times'][-1]:.2f} ({len(z['times'])})")
    else:
        d = load_pickled(args.meta_file)
        print("metadata keys:", {k: (type(v).__name__, getattr(v, "shape", len(v))) for k, v in d.items()})
        np.savez(args.out_file, **{k: np.array([str(x) for x in v]) for k, v in d.items()})


if __name__ == "__main__":
    main()
