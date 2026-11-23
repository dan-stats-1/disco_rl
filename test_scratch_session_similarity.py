"""Focused tests for application weights and reference-to-candidate ranking."""

import math
import unittest

import pandas as pd

from scratch_session_similarity import (
    aggregate_sessions,
    fit_application_weights,
    rank_candidates,
)


class ApplicationWeightsTest(unittest.TestCase):

  def setUp(self):
    self.events = pd.DataFrame({
        'SessionId': ['a', 'a', 'b', 'c', 'd'],
        'ApplicationName': ['mail', 'mail', 'mail', 'storage', 'storage'],
        'browser': ['chrome', 'edge', 'chrome', 'edge', 'edge'],
        'enabled': [True, True, False, True, True],
        'latency': [0.0, 10.0, 10.0, None, 20.0],
    })

  def fit(self, events):
    fingerprints = aggregate_sessions(
        events,
        categorical_features=['browser', 'enabled'],
        numeric_features=['latency'],
    )
    return fit_application_weights(
        fingerprints,
        categorical_features=['browser', 'enabled'],
        numeric_scales={'latency': 5.0},
    )

  def test_application_specific_distribution_weights(self):
    weights = self.fit(self.events)
    self.assertEqual(weights.index.names, ['ApplicationName', 'feature'])
    self.assertEqual(len(weights), 6)
    self.assertEqual(weights.loc[('mail', 'browser'), 'weight'], 0.5)
    self.assertEqual(weights.loc[('mail', 'browser'), 'variability'], 0.5)
    self.assertEqual(weights.loc[('mail', 'enabled'), 'weight'], 0.0)
    self.assertEqual(weights.loc[('storage', 'browser'), 'weight'], 1.0)
    self.assertAlmostEqual(
        weights.loc[('mail', 'latency'), 'weight'], math.exp(-1)
    )

  def test_event_volume_does_not_change_weights(self):
    longer_session = pd.concat(
        [self.events, self.events[self.events['SessionId'] == 'a']]
    )
    pd.testing.assert_frame_equal(
        self.fit(self.events), self.fit(longer_session)
    )

  def test_identical_mixed_distributions_are_stable(self):
    events = pd.DataFrame({
        'SessionId': ['a', 'a', 'b', 'b', 'c', 'c'],
        'ApplicationName': ['mail'] * 6,
        'browser': ['edge', 'chrome'] * 3,
    })
    fingerprints = aggregate_sessions(
        events, categorical_features=['browser'], numeric_features=[]
    )
    weights = fit_application_weights(
        fingerprints, categorical_features=['browser'], numeric_scales={}
    )
    self.assertEqual(weights.loc[('mail', 'browser'), 'weight'], 1.0)
    self.assertEqual(weights.loc[('mail', 'browser'), 'pair_count'], 3)

  def test_distinct_pairs_only_and_missing_evidence(self):
    events = self.events.copy()
    events.loc[events['SessionId'] == 'a', 'latency'] = None
    weights = self.fit(events)
    for application in ['mail', 'storage']:
      row = weights.loc[(application, 'latency')]
      self.assertEqual(row['session_count'], 1)
      self.assertEqual(row['pair_count'], 0)
      self.assertTrue(math.isnan(row['weight']))
      self.assertTrue(math.isnan(row['variability']))
    singleton = self.fit(events[events['SessionId'] == 'a'])
    self.assertTrue(singleton['weight'].isna().all())
    self.assertTrue((singleton['pair_count'] == 0).all())
    self.assertEqual(singleton.loc[('mail', 'latency'), 'session_count'], 0)

  def test_missing_categories_are_evidence(self):
    events = self.events.assign(browser=None)
    weights = self.fit(events)
    self.assertTrue((weights.xs('browser', level='feature')['weight'] == 1).all())

  def test_empty_history_and_invalid_configuration(self):
    fingerprints = aggregate_sessions(
        self.events,
        categorical_features=['browser'],
        numeric_features=['latency'],
    )
    empty = fit_application_weights(
        fingerprints.iloc[:0],
        categorical_features=['browser'],
        numeric_scales={},
    )
    self.assertTrue(empty.empty)
    for scale in [0, -1, math.nan, math.inf]:
      with self.subTest(scale=scale), self.assertRaises(ValueError):
        fit_application_weights(
            fingerprints,
            categorical_features=['browser'],
            numeric_scales={'latency': scale},
        )
    for frame, categorical, numeric in [
        (fingerprints, ['browser'], {'browser': 1.0}),
        (fingerprints, ['missing'], {}),
        (fingerprints, ['event_count'], {}),
        (fingerprints.reset_index(), ['browser'], {}),
        (pd.concat([fingerprints, fingerprints]), ['browser'], {}),
    ]:
      with self.subTest(features=categorical), self.assertRaises(ValueError):
        fit_application_weights(
            frame, categorical_features=categorical, numeric_scales=numeric
        )


class CandidateRankingTest(unittest.TestCase):

  def setUp(self):
    events = pd.DataFrame({
        'SessionId': ['bad', 'different', 'match', 'missing', 'unknown'],
        'ApplicationName': ['mail', 'storage', 'storage', 'mail', 'new_app'],
        'browser': ['chrome', 'edge', 'chrome', 'chrome', 'chrome'],
        'client': ['desktop'] * 5,
        'latency': [100.0, 110.0, 100.0, None, 100.0],
    })
    self.fingerprints = aggregate_sessions(
        events,
        categorical_features=['browser', 'client'],
        numeric_features=['latency'],
    )
    index = pd.MultiIndex.from_product(
        [['mail', 'storage'], ['browser', 'client', 'latency']],
        names=['ApplicationName', 'feature'],
    )
    self.weights = pd.DataFrame(
        {'weight': [1.0, 0.8, 0.6, 0.5, 0.2, 0.4]}, index=index
    )

  def rank(self, fingerprints=None, weights=None):
    return rank_candidates(
        self.fingerprints if fingerprints is None else fingerprints,
        ('bad', 'mail'),
        self.weights if weights is None else weights,
        categorical_features=['browser', 'client'],
        numeric_scales={'latency': 10.0},
    )

  def test_minimum_weights_ranking_and_explanations(self):
    ranking, details = self.rank()
    self.assertEqual(list(ranking.index), [
        ('match', 'storage'), ('missing', 'mail'),
        ('different', 'storage'), ('unknown', 'new_app'),
    ])
    row = ranking.loc[('different', 'storage')]
    self.assertAlmostEqual(row['similarity'], (0.2 + 0.4 * math.exp(-1)) / 1.1)
    self.assertAlmostEqual(row['weight_sum'], 1.1)
    self.assertEqual(row['coverage'], 1.0)
    self.assertEqual(row['compared_features'], 3)
    browser = details.loc[('different', 'storage', 'browser')]
    self.assertEqual(browser['reference_weight'], 1.0)
    self.assertEqual(browser['candidate_weight'], 0.5)
    self.assertEqual(browser['used_weight'], 0.5)
    self.assertEqual(browser['contribution'], 0.0)
    self.assertEqual(browser['status'], 'compared')

  def test_missing_observations_and_unavailable_baselines(self):
    ranking, details = self.rank()
    row = ranking.loc[('missing', 'mail')]
    self.assertEqual(row['similarity'], 1.0)
    self.assertAlmostEqual(row['coverage'], 2 / 3)
    self.assertAlmostEqual(row['weight_sum'], 1.8)
    self.assertEqual(
        details.loc[('missing', 'mail', 'latency'), 'status'],
        'missing_observations',
    )
    unknown = ranking.loc[('unknown', 'new_app')]
    self.assertTrue(math.isnan(unknown['similarity']))
    self.assertEqual(unknown['coverage'], 0.0)
    self.assertTrue(
        (details.loc[('unknown', 'new_app'), 'status'] == 'unavailable_weight').all()
    )
    weights = self.weights.copy()
    weights.loc[('mail', 'browser'), 'weight'] = math.nan
    _, details = self.rank(weights=weights)
    self.assertEqual(
        details.loc[('match', 'storage', 'browser'), 'status'],
        'unavailable_weight',
    )

  def test_zero_weights_and_reference_only_input(self):
    ranking, details = self.rank(weights=self.weights.assign(weight=0.0))
    self.assertTrue(ranking['similarity'].isna().all())
    self.assertTrue((ranking['weight_sum'] == 0).all())
    self.assertTrue((ranking['coverage'] == 0).all())
    self.assertTrue((details.loc[('match', 'storage'), 'status'] == 'zero_weight').all())
    ranking, details = self.rank(
        fingerprints=self.fingerprints.loc[[('bad', 'mail')]]
    )
    self.assertTrue(ranking.empty)
    self.assertTrue(details.empty)
    self.assertEqual(ranking.index.names, ['SessionId', 'ApplicationName'])
    self.assertEqual(
        details.index.names, ['SessionId', 'ApplicationName', 'feature']
    )

  def test_fitted_weights_work_without_refitting_candidates(self):
    history = pd.DataFrame({
        'SessionId': ['h1', 'h2', 'h3', 'h4'],
        'ApplicationName': ['mail', 'mail', 'storage', 'storage'],
        'browser': ['chrome'] * 4,
        'client': ['desktop'] * 4,
        'latency': [100.0] * 4,
    })
    history = aggregate_sessions(
        history,
        categorical_features=['browser', 'client'],
        numeric_features=['latency'],
    )
    weights = fit_application_weights(
        history,
        categorical_features=['browser', 'client'],
        numeric_scales={'latency': 10.0},
    )
    original = weights.copy(deep=True)
    ranking, _ = self.rank(weights=weights)
    self.assertEqual(ranking.loc[('match', 'storage'), 'similarity'], 1.0)
    pd.testing.assert_frame_equal(weights, original)

  def test_invalid_weights_and_absent_reference_raise(self):
    for invalid in [-0.1, 1.1, math.inf, -math.inf, 'invalid']:
      with self.subTest(weight=invalid), self.assertRaises((ValueError, TypeError)):
        self.rank(weights=self.weights.assign(weight=invalid))
    for weights in [
        self.weights.reset_index(),
        pd.concat([self.weights, self.weights]),
        self.weights.rename(columns={'weight': 'wrong'}),
    ]:
      with self.assertRaises(ValueError):
        self.rank(weights=weights)
    with self.assertRaises(ValueError):
      self.rank(fingerprints=self.fingerprints.drop(index=('bad', 'mail')))


if __name__ == '__main__':
  unittest.main()
