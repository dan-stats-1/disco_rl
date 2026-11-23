"""Aggregate session features, fit stability weights, and rank reference matches."""

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
import math
from itertools import combinations
from typing import TypeAlias

import pandas as pd
from scipy.stats import wasserstein_distance


Category: TypeAlias = str | int | float | bool | None


def categorical_proportions(
    observations: Iterable[Category],
) -> dict[Category, float]:
  """Return relative frequencies, combining scalar missing values under None.

  Supply categorical or boolean scalar values, not event rows.
  """
  counts = Counter(
      None if pd.isna(value) else value for value in observations
  )
  total = counts.total()
  if total == 0:
    raise ValueError('Cannot compare a feature with no observations.')
  return {value: count / total for value, count in counts.items()}


def categorical_similarity(
    proportions_a: Mapping[Category, float],
    proportions_b: Mapping[Category, float],
) -> float:
  """Return histogram overlap: 0 means disjoint, 1 means identical mixes.

  Inputs are normalized frequency dictionaries, as produced by
  aggregate_sessions, not raw observations.
  """
  for proportions in (proportions_a, proportions_b):
    if (
        not proportions
        or any(not math.isfinite(p) or p < 0 for p in proportions.values())
        or not math.isclose(math.fsum(proportions.values()), 1.0)
    ):
      raise ValueError('Category proportions must be nonnegative and sum to 1.')
  return math.fsum(
      min(proportion, proportions_b.get(value, 0.0))
      for value, proportion in proportions_a.items()
  )


def numeric_similarity(
    observations_a: Sequence[float],
    observations_b: Sequence[float],
    *,
    scale: float,
) -> float | None:
  """Return exp(-Wasserstein distance / scale), or None for missing evidence.

  Inputs are numeric observation lists from aggregate_sessions. Each list is
  treated as an empirical distribution, regardless of its observation count.
  scale is a positive, finite tolerance in the feature's units; a distance
  equal to scale scores exp(-1), approximately 0.37. Reuse the same scale for
  comparisons of a feature rather than deriving it from each pair.
  """
  if not math.isfinite(scale) or scale <= 0:
    raise ValueError('Numeric scale must be positive and finite.')
  for observations in (observations_a, observations_b):
    if any(not math.isfinite(value) for value in observations):
      raise ValueError('Numeric observations must be finite.')
  if len(observations_a) == 0 or len(observations_b) == 0:
    return None
  distance = wasserstein_distance(observations_a, observations_b)
  return math.exp(-distance / scale)


def _numeric_observations(observations: pd.Series) -> list[float]:
  values = pd.to_numeric(observations, errors='raise').dropna().astype(float)
  if not all(math.isfinite(value) for value in values):
    raise ValueError(f'Numeric feature {observations.name!r} contains infinity.')
  return values.tolist()


def aggregate_sessions(
    events: pd.DataFrame,
    *,
    categorical_features: Sequence[str],
    numeric_features: Sequence[str],
) -> pd.DataFrame:
  """Return one row per (SessionId, ApplicationName), indexed by those keys.

  Categorical/boolean cells contain normalized frequency dictionaries; all
  missing representations become the None category. Numeric cells contain
  lists of nonmissing observations; an empty list means no numeric evidence.
  event_count records the number of input rows, including missing observations.

  Feature types are explicit: numeric-looking identifiers may be categorical.
  Missing grouping keys, overlapping feature lists and invalid numbers raise.
  """
  group_columns = ['SessionId', 'ApplicationName']

  features = [*categorical_features, *numeric_features]
  if not events.columns.is_unique:
    raise ValueError('Input column names must be unique.')
  if not features or len(set(features)) != len(features):
    raise ValueError('Provide nonempty, nonoverlapping feature lists.')
  if set(features) & {*group_columns, 'event_count'}:
    raise ValueError('Features cannot use grouping keys or event_count.')
  
  missing_columns = set(group_columns + features) - set(events.columns)
  if missing_columns:
    raise ValueError(f'Missing columns: {sorted(missing_columns)}')
  if events[group_columns].isna().any().any():
    raise ValueError('SessionId and ApplicationName cannot be missing.')

  aggregations = {
      feature: categorical_proportions for feature in categorical_features
  }
  aggregations.update({
      feature: _numeric_observations for feature in numeric_features
  })
  
  groups = events.groupby(group_columns, sort=False, observed=True)
  fingerprints = groups.agg(aggregations)
  fingerprints['event_count'] = groups.size()
  return fingerprints


def _validate_fingerprints(
    fingerprints: pd.DataFrame,
    categorical_features: Sequence[str],
    numeric_scales: Mapping[str, float],
) -> list[str]:
  features = [*categorical_features, *numeric_scales]
  if not features or len(set(features)) != len(features):
    raise ValueError('Provide nonempty, nonoverlapping feature lists.')
  if (
      not isinstance(fingerprints.index, pd.MultiIndex)
      or list(fingerprints.index.names) != ['SessionId', 'ApplicationName']
      or not fingerprints.index.is_unique
      or fingerprints.index.to_frame(index=False).isna().any().any()
  ):
    raise ValueError(
        'Expected unique, nonmissing (SessionId, ApplicationName) index keys.'
    )
  if not fingerprints.columns.is_unique:
    raise ValueError('Fingerprint column names must be unique.')
  missing_columns = set(features) - set(fingerprints.columns)
  if missing_columns:
    raise ValueError(f'Missing columns: {sorted(missing_columns)}')
  if 'event_count' in features:
    raise ValueError('event_count is metadata, not a feature distribution.')
  if any(not math.isfinite(scale) or scale <= 0
         for scale in numeric_scales.values()):
    raise ValueError('Numeric scales must be positive and finite.')
  return features


def fit_application_weights(
    fingerprints: pd.DataFrame,
    *,
    categorical_features: Sequence[str],
    numeric_scales: Mapping[str, float],
) -> pd.DataFrame:
  """Estimate feature stability from historical, aggregated session rows.

  Returns a DataFrame indexed by (ApplicationName, feature), with columns:
    variability: Mean distance (1 - similarity) between distinct sessions.
    weight: Mean similarity, bounded between 0 and 1; not normalized.
    session_count: Number of sessions with evidence for this feature.
    pair_count: Number of session pairs contributing to the estimate.

  Sessions contribute equally, regardless of event_count. Empty numeric lists
  are excluded; categorical missing values remain a category. With fewer than
  two usable sessions, variability and weight are NaN, not estimated defaults.
  numeric_scales maps numeric feature names to positive, finite scales reused
  when scoring candidates. Fit on reference history, not incoming candidates.

  Exact pairwise comparison is quadratic in sessions per application.
  """
  features = _validate_fingerprints(
      fingerprints, categorical_features, numeric_scales
  )
  records = []
  for application, sessions in fingerprints.groupby(
      level='ApplicationName', sort=False, observed=True
  ):
    for feature in features:
      observations = sessions[feature].tolist()
      if feature in numeric_scales:
        observations = [values for values in observations if len(values) > 0]
      similarity_sum = 0.0
      pair_count = 0
      for left, right in combinations(observations, 2):
        if feature in numeric_scales:
          score = numeric_similarity(left, right, scale=numeric_scales[feature])
        else:
          score = categorical_similarity(left, right)
        if score is not None:
          similarity_sum += score
          pair_count += 1
      weight = similarity_sum / pair_count if pair_count else math.nan
      records.append({
          'ApplicationName': application,
          'feature': feature,
          'variability': 1.0 - weight,
          'weight': weight,
          'session_count': len(observations),
          'pair_count': pair_count,
      })
  return pd.DataFrame.from_records(
      records,
      columns=[
          'ApplicationName', 'feature', 'variability', 'weight',
          'session_count', 'pair_count',
      ],
  ).set_index(['ApplicationName', 'feature'])


def rank_candidates(
    fingerprints: pd.DataFrame,
    reference_key: tuple[Hashable, Hashable],
    application_weights: pd.DataFrame,
    *,
    categorical_features: Sequence[str],
    numeric_scales: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Compare each row against reference_key; return (ranking, feature_details).

  fingerprints includes the reference and candidate rows. Only the exact
  reference key is excluded; filter candidates by date before calling.
  application_weights comes from fit_application_weights on reference history.
  Use the same feature definitions and numeric scales for fitting and scoring.

  Each feature uses min(reference app weight, candidate app weight). Missing
  baselines, NaN weights, zero weights and empty numeric observations do not
  contribute. Similarity is the weighted mean of contributing feature scores,
  or NaN when none contribute. Coverage is the contributing feature count
  divided by the total configured feature count (including excluded features).

  Ranking is indexed by (SessionId, ApplicationName), descending similarity,
  with NaN last and ties in input order. Details add a feature index level,
  exposing scores, both application weights, used weights and exclusion reasons.
  Scores are similarities, not probabilities of malicious activity.
  """
  features = _validate_fingerprints(
      fingerprints, categorical_features, numeric_scales
  )
  if len(reference_key) != 2 or reference_key not in fingerprints.index:
    raise ValueError('reference_key must identify an existing session/app row.')
  if (
      not isinstance(application_weights.index, pd.MultiIndex)
      or list(application_weights.index.names) != ['ApplicationName', 'feature']
      or not application_weights.index.is_unique
      or application_weights.index.to_frame(index=False).isna().any().any()
      or not application_weights.columns.is_unique
      or 'weight' not in application_weights.columns
  ):
    raise ValueError(
        'Expected unique (ApplicationName, feature) weight rows and a weight column.'
    )
  weights = pd.to_numeric(application_weights['weight'], errors='raise')
  if any(not math.isfinite(w) or not 0 <= w <= 1 for w in weights.dropna()):
    raise ValueError('Weights must be between 0 and 1, or NaN for unavailable.')

  reference = fingerprints.loc[reference_key]
  summaries = []
  details = []
  for key, candidate in fingerprints.iterrows():
    if key == reference_key:
      continue
    used_weights = []
    contributions = []
    for feature in features:
      if feature in numeric_scales:
        score = numeric_similarity(
            reference[feature], candidate[feature], scale=numeric_scales[feature]
        )
      else:
        score = categorical_similarity(reference[feature], candidate[feature])
      reference_weight = weights.get((reference_key[1], feature), math.nan)
      candidate_weight = weights.get((key[1], feature), math.nan)
      weight = (
          math.nan if pd.isna(reference_weight) or pd.isna(candidate_weight)
          else min(reference_weight, candidate_weight)
      )
      used_weight = 0.0
      contribution = 0.0
      if pd.isna(weight):
        status = 'unavailable_weight'
      elif weight == 0:
        status = 'zero_weight'
      elif score is None:
        status = 'missing_observations'
      else:
        status = 'compared'
        used_weight = weight
        contribution = weight * score
        used_weights.append(used_weight)
        contributions.append(contribution)
      details.append({
          'SessionId': key[0], 'ApplicationName': key[1], 'feature': feature,
          'similarity': score if score is not None else math.nan,
          'reference_weight': reference_weight,
          'candidate_weight': candidate_weight,
          'weight': weight, 'used_weight': used_weight,
          'contribution': contribution, 'status': status,
      })
    weight_sum = math.fsum(used_weights)
    summaries.append({
        'SessionId': key[0], 'ApplicationName': key[1],
        'similarity': (
            math.fsum(contributions) / weight_sum if weight_sum else math.nan
        ),
        'coverage': len(used_weights) / len(features),
        'compared_features': len(used_weights), 'weight_sum': weight_sum,
    })
  ranking = pd.DataFrame.from_records(
      summaries,
      columns=[
          'SessionId', 'ApplicationName', 'similarity', 'coverage',
          'compared_features', 'weight_sum',
      ],
  ).set_index(['SessionId', 'ApplicationName'])
  feature_details = pd.DataFrame.from_records(
      details,
      columns=[
          'SessionId', 'ApplicationName', 'feature', 'similarity',
          'reference_weight', 'candidate_weight', 'weight', 'used_weight',
          'contribution', 'status',
      ],
  ).set_index(['SessionId', 'ApplicationName', 'feature'])
  return (
      ranking.sort_values(
          'similarity', ascending=False, kind='stable', na_position='last'
      ),
      feature_details,
  )


if __name__ == '__main__':
  events = pd.DataFrame({
      'SessionId': ['known_bad'] * 22 + ['new_session'] * 20,
      'ApplicationName': ['mail'] * 22 + ['storage'] * 20,
      'browser': ['edge'] * 2 + ['chrome'] * 40,
      'latency': [100.0] * 22 + [110.0] * 20,
  })
  fingerprints = aggregate_sessions(
      events, categorical_features=['browser'], numeric_features=['latency']
  )
  known_bad = fingerprints.loc[('known_bad', 'mail')]
  candidate = fingerprints.loc[('new_session', 'storage')]
  browser_score = categorical_similarity(known_bad['browser'], candidate['browser'])
  latency_score = numeric_similarity(
      known_bad['latency'], candidate['latency'], scale=10.0
  )
  print(f'Browser similarity: {browser_score:.3f}')
  print(f'Latency similarity: {latency_score}')
