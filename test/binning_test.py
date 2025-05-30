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

"""Tests for binning.py."""

from absl.testing import absltest
import numpy as np
from context import bin_and_aggregate



class BinningTest(absltest.TestCase):
    def testErrors(self):
        # x size less than 2.
        x = [1]
        y = [2]
        with self.assertRaises(ValueError):
            bin_and_aggregate(x, y, num_bins=2, bin_width=1, x_min=0, x_max=2)

        # x and y not the same size.
        x = [1, 2]
        y = [4, 5, 6]
        with self.assertRaises(ValueError):
            bin_and_aggregate(x, y, num_bins=2, bin_width=1, x_min=0, x_max=2)

        # x_min not less than x_max.
        x = [1, 2, 3]
        with self.assertRaises(ValueError):
            bin_and_aggregate(x, y, num_bins=2, bin_width=1, x_min=-1, x_max=-1)

        # x_min greater than the last element of x.
        with self.assertRaises(ValueError):
            bin_and_aggregate(x, y, num_bins=2, bin_width=0.25, x_min=3.5, x_max=4)

        # bin_width nonpositive.
        with self.assertRaises(ValueError):
            bin_and_aggregate(x, y, num_bins=2, bin_width=0, x_min=1, x_max=3)

        # bin_width greater than or equal to x_max - x_min.
        with self.assertRaises(ValueError):
            bin_and_aggregate(x, y, num_bins=2, bin_width=1, x_min=1.5, x_max=2.5)

        # num_bins less than 2.
        x = [1, 2, 3]
        with self.assertRaises(ValueError):
            bin_and_aggregate(x, y, num_bins=1, bin_width=1, x_min=0, x_max=2)

    def testBucketBoundaries(self):
        x = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5)
        np.testing.assert_array_equal([2.5, 4.5, 6.5, 8.5, 10.5], result)
        np.testing.assert_array_equal([2, 2, 2, 2, 2], bin_counts)

    def testMultiSizeBins(self):
        # Construct bins with size 0, 1, 2, 3, 4, 5, 10, respectively.
        x = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
        y = np.array([0, -1, 1, 4, 5, 6, 2, 2, 4, 4, 1, 1, 1, 1, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=7, bin_width=1, x_min=0, x_max=7)
        np.testing.assert_array_equal([0, 0, 0, 5, 3, 1, 5.5], result)
        np.testing.assert_array_equal([0, 1, 2, 3, 4, 5, 10], bin_counts)

    def testMedianAggr(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array([0, -1, 1, 4, 5, 6, 1, 3, 3, 9, 1, 1, 1, 1, -2])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5)
        np.testing.assert_array_equal([0, 0, 5, 3, 1], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)

    def testMeanAggr(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array([0, -1, 1, 4, 5, 6, 1, 3, 3, 9, 1, 1, 1, 1, -2])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5, aggr_fn=np.mean)
        np.testing.assert_array_almost_equal([0, 0, 5, 4, 0.4], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)

    def testMinAggr(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array([0, -1, 1, 4, 5, 6, 1, 3, 3, 9, 1, 1, 1, 1, -2])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5, aggr_fn=np.min)
        np.testing.assert_array_equal([0, -1, 4, 1, -2], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)

    def testMaxAggr(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array([0, -1, 1, 4, 5, 6, 1, 3, 3, 9, 1, 1, 1, 1, -2])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5, aggr_fn=np.max)
        np.testing.assert_array_equal([0, 1, 6, 9, 1], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)

    def testWideBins(self):
        x = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=6, x_min=-7, x_max=7)
        np.testing.assert_array_equal([3, 4.5, 6.5, 8.5, 10.5], result)
        np.testing.assert_array_equal([5, 6, 6, 6, 6], bin_counts)

    def testNarrowBins(self):
        x = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=1, x_min=-4.5, x_max=4.5)
        np.testing.assert_array_equal([3, 5, 7, 9, 11], result)
        np.testing.assert_array_equal([1, 1, 1, 1, 1], bin_counts)

    def testDefaultXMinMax(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array([7, -1, 3, 4, 5, 6, 2, 2, 4, 4, 1, 1, 1, 1, -1])
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5)
        np.testing.assert_array_equal([7, 1, 5, 0, 3], result)
        np.testing.assert_array_equal([1, 2, 3, 0, 4], bin_counts)

    def test2D(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array(
            [
                [0, 123],
                [-1, 0],
                [1, 2],
                [4, -4],
                [5, 4],
                [6, 6],
                [1, 50],
                [3, 100],
                [3, 100],
                [9, 110],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [-2, 5],
            ]
        )
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5)
        np.testing.assert_array_equal([[0, 123], [0, 1], [5, 4], [3, 100], [1, 0]], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)

    def test2DMean(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array(
            [
                [0, 123],
                [-1, 0],
                [1, 2],
                [4, -4],
                [5, 4],
                [6, 6],
                [1, 50],
                [3, 100],
                [3, 100],
                [9, 110],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [-2, 5],
            ]
        )
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5, aggr_fn=np.mean)
        np.testing.assert_array_almost_equal([[0, 123], [0, 1], [5, 2], [4, 90], [0.4, 1]], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)

    def test2DMin(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array(
            [
                [0, 123],
                [-1, 0],
                [1, 2],
                [4, -4],
                [5, 4],
                [6, 6],
                [1, 50],
                [3, 100],
                [3, 100],
                [9, 110],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [-2, 5],
            ]
        )
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5, aggr_fn=np.min)
        np.testing.assert_array_almost_equal([[0, 123], [-1, 0], [4, -4], [1, 50], [-2, 0]], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)

    def test2DMax(self):
        x = np.array([-4, -2, -2, 0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        y = np.array(
            [
                [0, 123],
                [-1, 0],
                [1, 2],
                [4, -4],
                [5, 4],
                [6, 6],
                [1, 50],
                [3, 100],
                [3, 100],
                [9, 110],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [-2, 5],
            ]
        )
        result, bin_counts = bin_and_aggregate(x, y, num_bins=5, bin_width=2, x_min=-5, x_max=5, aggr_fn=np.max)
        np.testing.assert_array_almost_equal([[0, 123], [1, 2], [6, 6], [9, 110], [1, 5]], result)
        np.testing.assert_array_equal([1, 2, 3, 4, 5], bin_counts)


if __name__ == "__main__":
    absltest.main()
