"""Tests for the VSN class."""
from src.vsn import VSN, F
import pytest
import numpy as np


def test_pytest():
    """Test pytest can do simple maths."""
    assert 2 + 2 == 4


class TestVSN:
    """Tests for the VSN class."""

    @pytest.fixture
    def data(self):
        """Create example data for testing."""
        return np.arange(15).reshape(3, 5)

    def test_init(self):
        """Test the class inits correctly."""
        vsn = VSN()
        assert vsn.f is np.exp
        assert vsn.df is np.exp
        assert vsn.a is None
        assert vsn.b is None
        assert vsn.x is None

    def test_Y(self, data):
        """Test the Y function."""
        vsn = VSN()
        vsn.x = data
        vsn.a = np.ones(data.shape[0])
        vsn.b = np.ones(data.shape[0])

        expected = np.exp(np.ones(data.shape)) * data + np.ones(data.shape)

        assert np.allclose(vsn._Y(), expected)

    def test_h(self, data):
        """Test the h function."""
        vsn = VSN()
        vsn.x = data
        vsn.a = np.ones(data.shape[0])
        vsn.b = np.ones(data.shape[0])

        assert np.allclose(
            vsn._h(),
            np.arcsinh(np.exp(np.ones(data.shape)) * data + np.ones(data.shape)),
        )

    def test_mu(self, data):
        """Test the mu function."""
        vsn = VSN()
        vsn.x = data
        assert np.allclose(vsn._mu(), np.array([5, 6, 7, 8, 9]))

    def test_sigma(self, data):
        """Test the sigma function."""
        vsn = VSN()
        vsn.x = data
        assert np.allclose(vsn._sigma(), 4.320493798938574)

    def test_r(self, data):
        """Test the r function."""
        vsn = VSN()
        vsn.x = data
        vsn.a = np.ones(data.shape[0])
        vsn.b = np.ones(data.shape[0])
        assert np.allclose(
            vsn._r(),
            np.array(
                [
                    [
                        [
                            -4.11862641,
                            -3.9759804,
                            -4.43887755,
                            -5.08959988,
                            -5.83080647,
                        ],
                        [
                            -1.62524773,
                            -2.45475297,
                            -3.30910021,
                            -4.18196948,
                            -5.06918075,
                        ],
                        [
                            -0.9678257,
                            -1.87579932,
                            -2.79152896,
                            -3.71380877,
                            -4.64169384,
                        ],
                    ]
                ]
            ),
        )

    def test_A(self, data):
        """Test the A function."""
        vsn = VSN()
        vsn.x = data
        vsn.a = np.ones(data.shape[0])
        vsn.b = np.ones(data.shape[0])
        assert np.allclose(
            vsn._A(),
            np.array(
                [
                    [0.70710678, 0.25971293, 0.15352065, 0.10858589, 0.08392666],
                    [0.0683731, 0.05767494, 0.04986804, 0.04392086, 0.03924006],
                    [0.0354603, 0.03234438, 0.0297316, 0.02750924, 0.02559591],
                ]
            ),
        )

    def test_big_sigma(self, data):
        """Test the big sigma function."""
        vsn = VSN()
        vsn.x = data
        vsn.a = np.ones(data.shape[0])
        vsn.b = np.ones(data.shape[0])
        assert np.allclose(
            vsn._big_sigma(include_x=False),
            np.array([-0.21790353, 0.07345058, 0.05822896]),
        )

    def test_delta_a(self, data):
        """Test the delta a function."""
        vsn = VSN()
        vsn.x = data
        vsn.a = np.ones(data.shape[0])
        vsn.b = np.ones(data.shape[0])
        assert np.allclose(
            vsn._delta_a(), np.array([-0.21790353, 0.07345058, 0.05822896])
        )

    def test_delta_b(self, data):
        """Test the delta b function."""
        vsn = VSN()
        vsn.x = data
        vsn.a = np.ones(data.shape[0])
        vsn.b = np.ones(data.shape[0])
        assert np.allclose(
            vsn._delta_b(), np.array([-5.48577879, -3.93426055, -3.29970523])
        )
