"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_daily_mean_shape():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input).shape[0], test_input.shape[1])

def test_daily_max_ints():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_max

    test_input = np.array([[10, 2, 3, 4],
                           [5, 65, 7, 8],
                           [9, 10, 11, 1]])
    test_result = np.array([10, 65, 11, 8])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_min_ints():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_min

    test_input = np.array([[10, 2, 3, 4],
                           [5, 65, 7, 8],
                           [9, 10, 11, 1]])
    test_result = np.array([5, 2, 3, 1])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)
    
def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])