"""Aggregate session/application features and compare feature distributions."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import math
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
