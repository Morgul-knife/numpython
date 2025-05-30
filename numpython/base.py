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
    def from_numpy(cls, ndarray: np.ndarray) -> Pymatrix:
        """
        Create a matrix from np.ndarray.

        Parameters
        ----------
        ndarray : np.ndarray
            The matrix in numpy.ndarray format.

        Returns
        -------
        Pymatrix
            The matrix is identical to ndarray.
        """
        return cls([ndarray[row, :].tolist() for row in range(len(ndarray))])

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

    def __getitem__(self, idx: int | tuple[int, int]) -> float | list[float] | Pymatrix:
        """
        Get a matrix element, row, column, or submatrix using indexing or slicing.

        Parameters
        ----------
        idx : int | tuple[int | slice, int | slice]
 
        Returns
        -------
        : float | list[float] | Pymatrix
            The requested element, row/column, or submatrix.
        """
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            
            # Returns a number (float).
            if isinstance(row_idx, int) and isinstance(col_idx, int):
                return self.data[row_idx][col_idx]
            
            # Returns a slice (Pymatrix).
            rows = self.data[row_idx]
            if isinstance(rows[0], list):
                sliced = [row[col_idx] for row in rows]
            else:
                sliced = rows[col_idx]
            return Pymatrix(sliced)
        
        # Returns a row (list).
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

        """
        rows_str = []
        for row in self.data:
            formatted_row = ", ".join(f"{val:3g}" for val in row)
            rows_str.append(f"[{formatted_row}]")
        whitespace = len("Pymatrix([") * " "

        return f"Pymatrix([" + str("," + "\n" + whitespace).join(rows_str) + "])"

    def __str__(self) -> str:
        """
        Return a nicely formatted string of the matrix content.
        
        """
        rows_str = []
        for row in self.data:
            formatted_row = ", ".join(f"{val:3g}" for val in row)
            rows_str.append(f"[{formatted_row}]")
        whitespace = len("[") * " "

        return f"[" + str("\n" + whitespace).join(rows_str) + "]"

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