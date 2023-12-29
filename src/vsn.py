"""Variance Stabilizing Normalization class for microarray data."""
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from enum import Enum
import typing as tp


class F(Enum):
    """Enum for the function to be used in the VSN."""

    EXP = 1
    ID = 2


class VSN(BaseEstimator, TransformerMixin):
    """Variance Stabilizing Normalization class for microarray data."""

    def __init__(self, f: F = F.EXP):
        """Initialize the VSN.

        Args:
            f (F, optional): Function used in VSN on the gradients (b) before applied
            to x. A choice of exponential makes calculations easier.
            Defaults to F.EXP.
        """
        self.f = np.exp if f is F.EXP else lambda x: x
        self.df = np.exp if f is F.EXP else lambda _: 1

    def fit(self, x: npt.NDArray, tol: float = 1e-6, max_iter=1000) -> None:
        """Fit the VSN to the data.

        Args:
            x (npt.ArrayLike): Input data to be fit.
            tol (float, optional): Tolerance for gradient descent to stop. Defaults to 1e-6.
            max_iter (int, optional): Max number of iterations before gradient descent stops. Defaults to 1000.
        """
        self.x = x
        self.a = np.ones(x.shape[0])
        self.b = np.ones(x.shape[0])

        self._fit(tol, max_iter)

    def transform(self, x: npt.NDArray) -> npt.NDArray:
        """Transform the data using the fitted VSN.

        Args:
            x (npt.ArrayLike): Input data to be transformed.

        Returns:
            npt.NDArray: Transformed data.
        """
        return self._h(x)

    def fit_transform(self, x: npt.NDArray) -> npt.NDArray:
        """Both fit and transform the data.

        Args:
            x (npt.ArrayLike): Input data to be fit and transformed.

        Returns:
            npt.NDArray: Transformed data.
        """
        self.fit(x)
        return self.transform(x)

    def _h(self, x: tp.Optional[npt.NDArray] = None) -> npt.NDArray:
        """Calculate the h function.

        Args:
            x (npt.ArrayLike, optional): Data to be input into the function.
            For transformation, the data is passed directly, but for fitting,
            the class attribute is used. Defaults to None.

        Returns:
            npt.NDArray: Output of the h function.
        """
        if x is None:
            x = self.x
        return np.arcsinh(self._Y())

    def _Y(self) -> npt.NDArray:
        """Calculate the Y function (the inner function of h).

        Returns:
            npt.NDArray: Output of Y function
        """
        return np.broadcast_to(
            self.f(self.b[:, None]), self.x.shape
        ) * self.x + np.broadcast_to(self.a[:, None], self.x.shape)

    def _mu(self) -> npt.NDArray:
        """Calculate the mean of the data along the columns.

        Returns:
            npt.NDArray: Mean of the data along the columns.
        """
        return np.mean(self.x, axis=0)

    def _sigma(self) -> np.float64:
        """Calculate the standard deviation of all the data.

        Returns:
            npt.NDArray: Standard deviation of all the data.
        """
        return np.var(self.x)

    def _r(self) -> npt.NDArray:
        """Calculate the intermediate r function.

        Returns:
            npt.NDArray: Output of the r function.
        """
        return self._h() - self._mu()

    def _A(self) -> npt.NDArray:
        """Calculate the intermediate A function.

        Returns:
            npt.NDArray: Output of the A function
        """
        return 1 / (np.sqrt(1 + self._Y() ** 2))

    def _big_sigma(self, include_x: bool) -> npt.NDArray:
        """Calculate the sum of the intermediate sum in the gradient calculations.

        Args:
            include_x (bool): Include multiplying the result by the input data
            for the gradient of b.

        Returns:
            npt.NDArray: Output of the sum of the intermediate sum in the
            gradient calculations.
        """
        to_be_summed = (self._r() / self._sigma() + self._A() * self._Y()) * self._A()

        if include_x:
            to_be_summed = to_be_summed * self.x

        return np.sum(to_be_summed, axis=1)

    def _delta_a(self) -> npt.NDArray:
        """Calculate the gradient of a.

        Returns:
            npt.NDArray: Gradient of a.
        """
        return self._big_sigma(include_x=False)

    def _delta_b(self) -> npt.NDArray:
        """Calculate the gradient of b.

        Returns:
            npt.NDArray: Gradient of b.
        """
        n = self.x.shape[1]

        return -n * (self.df(self.b) / self.f(self.b)) + self.df(
            self.b
        ) * self._big_sigma(include_x=True)

    def _update(self) -> None:
        """Update the parameters a and b."""
        self.a += self._delta_a()
        self.b += self._delta_b()

    def _converged(self, tol: float) -> np.bool_:
        """Decide if the gradient descent has converged.

        Args:
            tol (float): Tolerance for the gradient descent convergence.

        Returns:
            bool: True if converged, False otherwise.
        """
        return np.all(np.abs(self._delta_a()) < tol) and np.all(
            np.abs(self._delta_b()) < tol
        )

    def _fit(self, tol: float, max_iter: int) -> None:
        """Perform the gradient descent.

        Args:
            tol (float): Tolerance for the gradient descent convergence.
            max_iter (int): Max number of iterations before gradient descent stops.
        """
        for _ in range(max_iter):
            self._update()
            if self._converged(tol):
                break
