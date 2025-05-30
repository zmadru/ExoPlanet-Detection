# Copyright 2018 The Exoplanet ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility function for smoothing data by binning and aggregating."""

import numpy as np


def bin_and_aggregate(x,
                      y,
                      num_bins,
                      bin_width=None,
                      x_min=None,
                      x_max=None,
                      aggr_fn=None):
  """Aggregates y-values in uniform intervals (bins) along the x-axis.

  The interval [x_min, x_max) is divided into num_bins uniformly spaced
  intervals of width bin_width. The value computed for each bin is the
  aggregation of all y-values whose corresponding x-value is in the interval.
  The default aggregation function is np.median.

  NOTE: x must be sorted in ascending order or the results will be incorrect.

  Args:
    x: 1D NumPy array of x-coordinates sorted in ascending order. Must have at
      least 2 elements, and all elements cannot be the same value.
    y: N-dimensional NumPy array with the same length as x.
    num_bins: The number of intervals to divide the x-axis into. Must be at
      least 2.
    bin_width: The width of each bin on the x-axis. Must be positive, and less
      than x_max - x_min. Defaults to (x_max - x_min) / num_bins.
    x_min: The inclusive leftmost value to consider on the x-axis. Must be less
      than or equal to the largest value of x. Defaults to min(x).
    x_max: The exclusive rightmost value to consider on the x-axis. Must be
      greater than x_min. Defaults to max(x).
    aggr_fn: A function that will be called with signature aggr_fn(y, axis=0) to
      aggregate values within each bin. Default is np.median.

  Returns:
    result: NumPy array of length num_bins containing the aggregated y-values of
      uniformly spaced bins on the x-axis.
    bin_counts: 1D NumPy array of length num_bins indicating the number of
      points in each bin.
  """
  if num_bins < 2:
    raise ValueError("num_bins must be at least 2. Got: {}".format(num_bins))

  # Validate the lengths of x and y.
  x_len = len(x)
  if x_len < 2:
    raise ValueError("len(x) must be at least 2. Got: {}".format(x_len))
  if x_len != len(y):
    raise ValueError("len(x) (got: {}) must equal len(y) (got: {})".format(
        x_len, len(y)))

  # Validate x_min and x_max.
  x_min = x_min if x_min is not None else x[0]
  x_max = x_max if x_max is not None else x[-1]
  if x_min >= x_max:
    raise ValueError("x_min (got: {}) must be less than x_max (got: {})".format(
        x_min, x_max))
  if x_min > x[-1]:
    raise ValueError(
        "x_min (got: {}) must be less than or equal to the largest value of x "
        "(got: {})".format(x_min, x[-1]))

  # Validate bin_width.
  bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
  if bin_width <= 0:
    raise ValueError("bin_width must be positive. Got: {}".format(bin_width))
  if bin_width >= x_max - x_min:
    raise ValueError(
        "bin_width (got: {}) must be less than x_max - x_min (got: {})".format(
            bin_width, x_max - x_min))

  bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

  if aggr_fn is None:
    aggr_fn = np.median

  # Initialize output arrays.
  # Ensure default_value is a floating point type, otherwise the aggregated
  # values may unexpectedly be cast to a non-floating type.
  default_dtype = y.dtype if isinstance(y.dtype, np.floating) else np.float32
  default_value = np.zeros_like(y[0], dtype=default_dtype)
  result = np.repeat([default_value], num_bins, axis=0)
  bin_counts = np.zeros(num_bins, dtype=np.int32)

  # Find the first element of x >= x_min. This loop is guaranteed to produce
  # a valid index because we know that x_min <= x[-1].
  x_start = 0
  while x[x_start] < x_min:
    x_start += 1

  # The bin at index i is the aggregation of all elements y[j] such that
  # bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of
  # bin i.
  bin_min = x_min  # Left endpoint of the current bin.
  bin_max = x_min + bin_width  # Right endpoint of the current bin.
  j_start = x_start  # Inclusive left index of the current bin.
  j_end = x_start  # Exclusive end index of the current bin.

  for i in range(num_bins):
    # Move j_start to the first index of x >= bin_min.
    while j_start < x_len and x[j_start] < bin_min:
      j_start += 1

    # Move j_end to the first index of x >= bin_max (exclusive end index).
    while j_end < x_len and x[j_end] < bin_max:
      j_end += 1

    if j_end > j_start:
      # Compute and insert the aggregate y value in the bin.
      result[i] = aggr_fn(y[j_start:j_end], axis=0)
      bin_counts[i] = j_end - j_start

    # Advance the bin.
    bin_min += bin_spacing
    bin_max += bin_spacing

  return result, bin_counts

def generate_view(time,
                  values,
                  num_bins,
                  bin_width,
                  t_min,
                  t_max,
                  normalize=True):
  """Generates a view of a phase-folded and binned light curve.

  Args:
    time: 1D NumPy array of time values, sorted in ascending order.
    values: N-dimensional NumPy array with the same length as time.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.

  Returns:
    NumPy array of length num_bins containing the aggregated values in uniformly
    spaced bins on the phase-folded time axis.
  """
  view, bin_counts = bin_and_aggregate(time, values, num_bins,
                                               bin_width, t_min, t_max)
  # Empty bins fall back to the global median.
  view = np.where(bin_counts > 0, view, np.median(values))

  if normalize:
    view -= np.median(view, axis=0)
    view /= np.abs(np.min(view, axis=0))

  return view



def global_view(time, values, period, num_bins=2001, bin_width_factor=1 / 2001, normalize=True):
  """Generates a 'global view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D NumPy array of time values, sorted in ascending order.
    values: N-dimensional NumPy array with the same length as time.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    NumPy array of length num_bins containing the aggregated values in uniformly
    spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      values,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period / 2,
      t_max=period / 2,
      normalize=normalize)


def local_view(time,
               values,
               period,
               duration,
               num_bins=201,
               bin_width_factor=0.16,
               num_durations=4,
               normalize=True):
  """Generates a 'local view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D NumPy array of time values, sorted in ascending order.
    values: N-dimensional NumPy array with the same length as time.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
      event is assumed to be centered at 0).

  Returns:
    NumPy array of length num_bins containing the aggregated values in uniformly
    spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      values,
      num_bins=num_bins,
      bin_width=duration * bin_width_factor,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations),
      # t_min=-duration * num_durations,
      # t_max=duration * num_durations,
      normalize=normalize)

