"""
Wrapper around IRCAM's `ircamdescriptor` CLI tool (native binary, no MATLAB dependency),
evaluated as a drop-in replacement for TimbreToolbox (MATLAB, ~20-40s startup per instance --
impractical for an IMGEP loop that may run thousands of evaluations).

Pipeline: audio array -> temp wav -> ircamdescriptor -> SDIF -> sdiftotext -> plain text ->
parsed into {descriptor_code: [values over short-term frames]} -> per-descriptor median/IQR,
mirroring TimbreToolbox's own naming convention (`_med`, `_IQR`) so features stay comparable.

Binaries are only on PATH in an interactive shell on IRCAM machines (bonsho/gottan) -- hence
the hardcoded absolute paths below rather than relying on shutil.which().
"""
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

IRCAMDESCRIPTOR_BIN = "/u/formes/share/bin/x86_64-Linux-db10/ircamdescriptor"
SDIFTOTEXT_BIN = "/u/formes/share/bin/x86_64-Linux-db10/sdiftotext"
DEFAULT_CONFIG = Path(__file__).parent / "ircamdescriptor_config_light.xml"

_FRAME_HEADER_RE = re.compile(r"^1DSC\s+\S+\s+\S+\s+([\d.eE+-]+)\s*$")
_MATRIX_HEADER_RE = re.compile(r"^\s*(\w{4})\s+0x[0-9a-fA-F]+\s+(\d+)\s+(\d+)\s*$")


def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nstderr:\n{result.stderr}")


def _parse_sdif_text(text_path: Path) -> Dict[str, List[List[float]]]:
    """ Returns {descriptor_code: [frame_0_values, frame_1_values, ...]}, one entry per
    short-term frame, in chronological order. Frame boundaries are detected via `1DSC` header
    lines; matrices are read until the next header (the `matrixCount` field in the `1DSC`
    header does not reliably match the number of matrices printed, so it is not used). """
    series: Dict[str, List[List[float]]] = {}
    with open(text_path) as f:
        lines = f.readlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if not line or not _FRAME_HEADER_RE.match(lines[i]):
            i += 1
            continue
        i += 1
        # Read matrices belonging to this frame until the next frame header (or EOF)
        while i < n:
            if _FRAME_HEADER_RE.match(lines[i]):
                break
            mheader = _MATRIX_HEADER_RE.match(lines[i])
            if not mheader:
                i += 1
                continue
            code, rows, cols = mheader.group(1), int(mheader.group(2)), int(mheader.group(3))
            i += 1
            values: List[float] = []
            for _ in range(rows):
                if i >= n:
                    break
                values.extend(float(x) for x in lines[i].split())
                i += 1
            series.setdefault(code, []).append(values)
    return series


def _aggregate(series: Dict[str, List[List[float]]]) -> Dict[str, float]:
    """ Median + IQR per descriptor, per scalar dimension (flattening multi-dim matrices into
    dim-indexed sub-features), mirroring TimbreToolbox's `_med`/`_IQR` suffix convention. """
    feats: Dict[str, float] = {}
    for code, frames in series.items():
        dims = max(len(f) for f in frames)
        for d in range(dims):
            track = np.array([f[d] for f in frames if len(f) > d], dtype=float)
            track = track[np.isfinite(track)]
            if track.size == 0:
                continue
            suffix = f"_{d}" if dims > 1 else ""
            feats[f"{code}{suffix}_med"] = float(np.median(track))
            q75, q25 = np.percentile(track, [75, 25])
            feats[f"{code}{suffix}_IQR"] = float(q75 - q25)
    return feats


def ircam_descriptor_extractor(audio: np.ndarray, fs: int, config_path: Path = DEFAULT_CONFIG) -> Dict[str, float]:
    """ Computes IRCAM descriptors on an in-memory audio array, aggregated (median/IQR) over
    the whole signal. Mirrors utils.timbral_models.Extractor.timbral_extractor's array+fs
    interface, so it can be swapped in wherever that is used. """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav_path = tmp / "audio.wav"
        sdif_path = tmp / "out.sdif"
        text_path = tmp / "out.txt"

        sf.write(wav_path, audio, fs)
        _run([IRCAMDESCRIPTOR_BIN, str(wav_path), str(config_path), f"-o{sdif_path}"])
        _run([SDIFTOTEXT_BIN, str(sdif_path), str(text_path)])

        series = _parse_sdif_text(text_path)
        return _aggregate(series)
