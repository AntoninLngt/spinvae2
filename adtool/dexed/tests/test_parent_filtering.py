"""Regression tests for parent filtering in NicheBiasedIMGEPExplorer.

Why these exist. `filter_non_notelike_parents` was correctly threaded from the config all the
way to an instance attribute, and never read: `_vector_search_for_goal` returned early unless
`filter_degenerate_parents` was set, so the note-like filter sat behind a guard that could not
open. Nothing raised, nothing logged, and two of the four conditions of an experiment ran as
exact copies of the control before anyone noticed.

A test asserting only that the attribute is set would have passed. So the discriminating test
here (`test_non_notelike_nearest_is_skipped`) drives the actual selection path with a nearest
candidate that is NOT note-like and a runner-up that is, and asserts which payload comes back
-- the one thing that differs between "filter runs" and "filter is dead code".

Run:  python -m pytest adtool/dexed/tests/test_parent_filtering.py -v
      (from ~/projects/spinvae2, with adtool/dexed on sys.path)
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

DEXED_ROOT = Path(__file__).resolve().parent.parent
for p in (str(DEXED_ROOT), str(DEXED_ROOT / 'scripts')):
    if p not in sys.path:
        sys.path.insert(0, p)

from explorers.NicheBiasedIMGEPExplorer import (  # noqa: E402
    NicheBiasedIMGEPExplorerInstance as Instance,
)


def _human_corpus_available():
    from maps import notelike
    return os.path.exists(notelike.HUMAN_FEATURES_PATH)


needs_corpus = unittest.skipUnless(
    _human_corpus_available(),
    'human feature corpus not reachable from this host (/data2 is host-local)')


class _Match:
    """Minimal stand-in for what history.nearest() returns."""

    def __init__(self, feature, payload):
        self.feature = feature
        self.payload = payload


class _History:
    def __init__(self, matches):
        self.matches = matches
        self.nearest_calls = 0

    def nearest(self, goal, k=1, history_lookback_length=None):
        self.nearest_calls += 1
        return self.matches[:k]

    def random(self, history_lookback_length=None):
        return None


class _ParameterMap:
    def __init__(self):
        self.sample_calls = 0

    def sample(self):
        self.sample_calls += 1
        return {'payload': 'FRESH_DRAW'}


def _make(filter_degenerate=False, filter_notelike=False, matches=()):
    """An instance with only the attributes the selection path touches."""
    o = Instance.__new__(Instance)
    o.filter_degenerate_parents = filter_degenerate
    o.filter_non_notelike_parents = filter_notelike
    o.niche_curiosity_bias = False          # -> niche_tracker is None
    o._niche_tracker = None
    o.niche_curiosity_k = 5
    o.history = _History(list(matches))
    o.parameter_map = _ParameterMap()
    o._degenerate_escape_count = 0
    return o


class TestParentFilteringGuard(unittest.TestCase):
    """The guard must open for EITHER flag: both gate the same code path."""

    def test_guard_truth_table(self):
        for degenerate in (False, True):
            for notelike in (False, True):
                with self.subTest(degenerate=degenerate, notelike=notelike):
                    o = _make(degenerate, notelike)
                    self.assertEqual(o._parent_filtering_enabled, degenerate or notelike)


class TestRejectedParent(unittest.TestCase):

    def test_all_zeros_sentinel_is_rejected_without_any_flag(self):
        """The silence fallback must never be mutated, filters on or off."""
        o = _make()
        self.assertTrue(o._is_rejected_parent(np.zeros(38)))

    def test_out_of_band_marker_is_rejected(self):
        o = _make()
        self.assertTrue(o._is_rejected_parent(np.full(38, 1e6)))

    @needs_corpus
    def test_notelike_flag_changes_the_verdict(self):
        z_ok, z_bad = _fixtures()
        off, on = _make(filter_notelike=False), _make(filter_notelike=True)
        self.assertFalse(off._is_rejected_parent(z_bad),
                         'without the flag, a non-note-like parent must still be accepted')
        self.assertTrue(on._is_rejected_parent(z_bad))
        self.assertFalse(on._is_rejected_parent(z_ok))


@needs_corpus
class TestSelectionPath(unittest.TestCase):
    """Drives _vector_search_for_goal itself -- the test the original bug would have failed."""

    def setUp(self):
        self.z_ok, self.z_bad = _fixtures()
        self.goal = np.zeros(38)

    def _matches(self):
        # Deliberately ordered so the NEAREST candidate is the unusable one.
        return [_Match(self.z_bad, {'payload': 'BAD'}),
                _Match(self.z_ok, {'payload': 'GOOD'})]

    def test_non_notelike_nearest_is_skipped(self):
        o = _make(filter_notelike=True, matches=self._matches())
        got = o._vector_search_for_goal(self.goal, history_lookback_length=None)
        self.assertEqual(got, {'payload': 'GOOD'},
                         'the note-like filter did not run: the nearest candidate was returned '
                         'even though its envelope is outside the human region')

    def test_without_the_flag_the_nearest_is_returned(self):
        """Guards against the opposite error: filtering when nobody asked for it."""
        o = _make(matches=self._matches())
        got = o._vector_search_for_goal(self.goal, history_lookback_length=None)
        self.assertEqual(got, {'payload': 'BAD'})

    def test_fresh_draw_when_no_candidate_is_viable(self):
        """With every candidate rejected, the loop must escape rather than mutate one anyway."""
        matches = [_Match(self.z_bad, {'payload': 'BAD1'}),
                   _Match(self.z_bad, {'payload': 'BAD2'})]
        o = _make(filter_notelike=True, matches=matches)
        got = o._vector_search_for_goal(self.goal, history_lookback_length=None)
        self.assertEqual(got, {'payload': 'FRESH_DRAW'})
        self.assertEqual(o.parameter_map.sample_calls, 1)
        self.assertEqual(o._degenerate_escape_count, 1)


def _fixtures():
    """One clearly note-like and one clearly non-note-like 38-D descriptor vector.

    Built from the fitted human statistics rather than from run data, so the tests do not
    depend on any particular experiment being present on disk. Both are checked against the
    criterion itself, so a fixture can never silently drift to the wrong side of the threshold.
    """
    from maps import notelike
    idx, med, iqr, _, thr = notelike._fit()

    z_ok = np.full(38, 1e-3)
    z_ok[idx] = med                       # the human median: distance ~0
    z_bad = np.full(38, 1e-3)
    z_bad[idx] = med + 40.0 * iqr         # far out, but below the 1e4 degeneracy marker

    d_ok, d_bad = notelike.distance(z_ok[None])[0], notelike.distance(z_bad[None])[0]
    assert d_ok <= thr < d_bad, f'fixtures invalides : d_ok={d_ok:.2f} thr={thr:.2f} d_bad={d_bad:.2f}'
    assert np.abs(z_bad).max() < 1e4, 'z_bad tomberait dans _is_degenerate, ce qui testerait autre chose'
    return z_ok, z_bad


if __name__ == '__main__':
    unittest.main()
