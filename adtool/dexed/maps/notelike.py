"""The "note-like" criterion: does a discovery behave like a musical note?

Motivation. Both diversity measures used in this project count a discovery as diverse when
its descriptor vector is far from the others, whether or not anyone could use the sound. A
blind listening test showed how far that goes: on 40 rated excerpts, the previous validity
mask separated nothing at all (mean rating 0.58 for accepted against 0.56 for rejected),
while every human preset was rated 2/2.

What this criterion constrains, and what it deliberately does not. Only the TEMPORAL
envelope is constrained -- attack, decay, effective duration, temporal centroid, amplitude
modulation. The spectral descriptors are left completely free, because a discovery tool is
supposed to find timbres nobody has heard; what it is not supposed to produce is something
that fails to behave like a note at all. In the same listening test, excerpts rated 2 sat at
a median Mahalanobis distance of 5.74 from the human envelope, against 11.55 for those rated
0 and 17.51 for those rated 1.

Why Mahalanobis rather than per-descriptor ranges. Requiring nine marginal percentile
intervals simultaneously is far stricter than each interval suggests -- the human corpus
itself only passed at 81.9%, purely as an artefact of intersecting nine constraints. A
Mahalanobis distance on the human covariance accounts for the correlations (a long attack
goes with a long decay in human presets) and, more importantly, makes the threshold an
explicit parameter: it is set as a percentile of the human distances, so the fraction of
human presets that pass is chosen rather than suffered.
"""
import os
from functools import lru_cache

import numpy as np

ENVELOPE_FEATURES = ('tt_Att', 'tt_LAT', 'tt_Dec', 'tt_DecSlope', 'tt_EffDur',
                     'tt_TempCent', 'tt_RMSEnv_med', 'tt_RMSEnv_IQR', 'tt_AmpMod')
HUMAN_PERCENTILE = 95.0     # fraction of human presets the criterion accepts, by construction
HUMAN_FEATURES_PATH = os.environ.get(
    'DEXED_HUMAN_FEATURES',
    '/data2/anasynth_nonbp/longeot/spinvae2_datasets/Dexed/raw_timbre_features_ttb.df.pickle')


@lru_cache(maxsize=1)
def _fit():
    """Median/IQR scaling, inverse covariance and threshold, all fitted on the human corpus."""
    import pandas as pd
    from maps.DexedStatistics import Z_FEATURE_NAMES

    names = list(Z_FEATURE_NAMES)
    idx = np.array([names.index(f) for f in ENVELOPE_FEATURES])
    df = pd.read_pickle(HUMAN_FEATURES_PATH)
    df = df[df['variation'] == 0]
    H = df[names].to_numpy(dtype=float)
    H = H[~np.isnan(H).any(axis=1)][:, idx]

    med = np.median(H, axis=0)
    q1, q3 = np.percentile(H, [25, 75], axis=0)
    iqr = np.where(q3 - q1 > 0, q3 - q1, 1.0)
    S = (H - med) / iqr
    # Ridge term: several envelope descriptors are near-collinear on human presets, which
    # makes the raw covariance ill-conditioned.
    cov_inv = np.linalg.inv(np.cov(S.T) + 1e-6 * np.eye(len(idx)))
    dists = np.sqrt(np.einsum('ij,jk,ik->i', S, cov_inv, S))
    return idx, med, iqr, cov_inv, float(np.percentile(dists, HUMAN_PERCENTILE))


def distance(Z):
    """Mahalanobis distance of each row of Z (38-D descriptors) to the human envelope."""
    idx, med, iqr, cov_inv, _ = _fit()
    S = (np.atleast_2d(np.asarray(Z, dtype=float))[:, idx] - med) / iqr
    if not np.isfinite(S).all():
        S = np.nan_to_num(S, nan=1e6, posinf=1e6, neginf=-1e6)
    return np.sqrt(np.einsum('ij,jk,ik->i', S, cov_inv, S))


def threshold():
    return _fit()[4]


def is_note_like(Z):
    """True for rows whose temporal envelope falls inside the human region."""
    return distance(Z) <= threshold()


def is_note_like_one(z):
    return bool(is_note_like(np.atleast_2d(z))[0])
