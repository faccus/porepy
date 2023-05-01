from __future__ import annotations

import porepy as pp
import numpy as np

from . import domains, fracture_sets
from typing import Union


class SquareDomainOrthogonalFractures:
    """Create a mixed-dimensional grid for a square domain with up to two
    orthogonal fractures.

    To be used as a mixin taking precedence over
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    params: dict
    """Parameters for the model geometry. Entries relevant for this mixin are:
        - domain_size: The side length of the square domain.
        - fracture_indices: List of indices of fractures to be included in the grid.

    """
    units: pp.Units
    """Units for the model geometry."""

    @property
    def domain_size(self) -> pp.number:
        """Return the side length of the square domain."""
        # Scale by length unit.
        return self.params.get("domain_size", 1) / self.units.m

    def set_fractures(self) -> None:
        """Assigns 0 to 2 fractures."""
        fracture_indices = self.params.get("fracture_indices", [0])
        all_fractures = fracture_sets.orthogonal_fractures_2d(self.domain_size)
        self._fractures = [all_fractures[i] for i in fracture_indices]

    def set_domain(self) -> None:
        """Set the square domain."""
        self._domain = domains.nd_cube_domain(2, self.domain_size)


class CubeDomainOrthogonalFractures:
    """Create a mixed-dimensional grid for a cube domain with up to three
    orthogonal fractures.

    To be used as a mixin taking precedence over
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    params: dict
    """Parameters for the model geometry. Entries relevant for this mixin are:
        - domain_size: The side length of the cube domain.
        - fracture_indices: List of indices of fractures to be included in the grid.

    """
    units: pp.Units
    """Units for the model geometry."""

    @property
    def domain_size(self) -> pp.number:
        """Return the side length of the cube domain."""
        # Scale by length unit.
        return self.params.get("domain_size", 1) / self.units.m

    def set_fractures(self) -> None:
        """Assigns 0 to 3 fractures."""
        fracture_indices = self.params.get("fracture_indices", [0])
        all_fractures = fracture_sets.orthogonal_fractures_3d(self.domain_size)
        self._fractures = [all_fractures[i] for i in fracture_indices]

    def set_domain(self) -> None:
        """Set the cube domain."""
        self._domain = domains.nd_cube_domain(3, self.domain_size)


class BenchmarkFlow2dGeigerGeometry:
    """Mixin class containing the geometry for the benchmark 4.1 from [1].

    To be used as a mixin taking precedence over
    :class:`~porepy.models.geometry.ModelGeometry`.

    References:

        -[1] Flemisch, Bernd, et al. "Benchmarks for single-phase flow in fractured
        porous media." Advances in Water Resources 111 (2018): 239-258.

    """
    def set_fractures(self) -> None:
        """The fracture set consists of six orthogonal fractures."""
        f1 = pp.LineFracture(np.array([[0.000, 1.000], [0.500, 0.500]]))
        f2 = pp.LineFracture(np.array([[0.500, 0.500], [0.000, 1.000]]))
        f3 = pp.LineFracture(np.array([[0.500, 1.000], [0.750, 0.750]]))
        f4 = pp.LineFracture(np.array([[0.750, 0.750], [0.500, 1.000]]))
        f5 = pp.LineFracture(np.array([[0.500, 0.750], [0.625, 0.625]]))
        f6 = pp.LineFracture(np.array([[0.625, 0.625], [0.500, 0.750]]))
        self._fractures: list[pp.LineFracture] = [f1, f2, f3, f4, f5, f6]
