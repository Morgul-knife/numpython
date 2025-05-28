from __future__ import annotations

import random
from typing import Any, Sequence
import numpy as np

class Pymatrix:
    """
    A simple matrix class with basic operations.
    """
    def __init__(self, data: list[list[float]]):
        """
        Initialize a Pymatrix from a 2D list.

        Parameters
        ----------
        data : list[list[float]]
            A non-empty list of lists with equal-length rows.
        """
        if not data or not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same length and be non-empty.")
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        self.shape = self.rows, self.cols

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Pymatrix:
        """
        Create a matrix filled with zeros.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.

        Returns
        -------
        Pymatrix
            A matrix of zeros.
        """
        return cls([[0.0] * cols for _ in range(rows)])

    @classmethod
    def random(cls, rows: int, cols: int, min_val=0.0, max_val=1.0) -> Pymatrix:
        """
        Create a matrix filled with random float values.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        min_val : float
            Minimum random value.
        max_val : float
            Maximum random value.

        Returns
        -------
        Pymatrix
            A matrix with random float values.
        """
        return cls(
            [[random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)]
            )

    def __getitem__(self, idx: int) -> list[float]:
        """
        Get a row by index.

        Parameters
        ----------
        idx : int
            Index of the row.

        Returns
        -------
        list[float]
            The requested row.
        """
        return self.data[idx]

    def __setitem__(self, idx: int, value: list[float]) -> None:
        """
        Set a row by index.

        Parameters
        ----------
        idx : int
            Row index.
        value : list[float]
            New row values.

        Raises
        ------
        ValueError
            If the length of the row does not match number of columns.
        """
        if len(value) != self.cols:
            raise ValueError("Invalid row size")
        self.data[idx] = value

    def __repr__(self) -> str:
        """
        Return the official string representation of the matrix.

        Returns
        -------
        str
            Representation of the matrix with dimensions.
        """
        return f"Pymatrix({self.rows}, {self.cols})"

    def __str__(self) -> str:
        """
        Return a nicely formatted string of the matrix content.

        Returns
        -------
        str
            Formatted matrix as a string.
        """
        return "\n".join(" ".join(f"{val:8.3f}" for val in row) for row in self.data)

    def transpose(self) -> Pymatrix:
        """
        Return the transpose of the matrix.

        Returns
        -------
        Pymatrix
            Transposed matrix.
        """
        transposed = list(zip(*self.data))
        return Pymatrix([list(row) for row in transposed])

    def copy(self) -> Pymatrix:
        """
        Return a deep copy of the matrix.

        Returns
        -------
        Pymatrix
            A new matrix with copied data.
        """
        return Pymatrix([row[:] for row in self.data])

    def __eq__(self, other: Any) -> bool:
        """
        Check equality with another matrix.

        Parameters
        ----------
        other : Any
            Another object to compare with.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        if not isinstance(other, Pymatrix):
            return False
        return self.data == other.data