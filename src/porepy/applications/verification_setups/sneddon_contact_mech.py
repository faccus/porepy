"""
This module contains the implementation of Sneddon's crack pressurization problem.

The problem consists of a one-dimensional crack of length :math:`2h` immersed in an
unbounded elastic solid, forming an angle :math:`\\theta` with the horizontal axis.

A pressure :math:`p_0` is exerted on both sides of the interior faces of the crack,
causing a relative normal displacement :math:`[\\mathbf{u}]_n`. Since there are no
shear forces, the relative tangential displacement is zero.

Sneddon [1] found an exact solution for the normal relative displacement:

.. math::

    [\\mathbf{u}]_n (\\eta) = 2h p_0 (1 - \\nu) G^{-1}
        \\left[1 - \\left(\\frac{\\eta}{h}\\right)^2 \\right]^{1/2},

where :math:`\\eta` is the distance from a point in the crack to its center,
:math:`\\nu` is the Poisson's coefficient, and :math:`G` is the shear modulus.

Using Sneddon's exact solution, Crouch and Starfield [2] proposed a semi-analytical
procedure based on the Boundary Element Method (BEM) to replace the infinite elastic
solid by a finite, two-dimensional elastic solid of length :math:`a` and height
:math:`b`,

In this implementation, we follow the BEM procedure from [2] to obtain the displacement
at the exterior boundary sides of the solid. Moreover, since the traction force on the
fracture is known, e.g., :math:`p_0` in the normal direction and zero in the
tangential direction, we do not need to solve the fracture equations. This means
that instead of solving the full contact mechanics problem, we solve a reduced
system, where the unknowns become the displacement field in the matrix and
the displacement field on the interface.

References:

    - [1] Sneddon, I.N.: Fourier Transforms. McGraw Hill Book Co, Inc., New York (1951).

    - [2] Crouch, S.L., Starfield, A.: Boundary Element Methods in Solid Mechanics:
      With Applications in Rock Mechanics and Geological Engineering. Allen & Unwin,
      London (1982).

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.colors as mcolors  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spsla

import porepy as pp
from porepy.applications.docstring_glossary import Glossary
from porepy.applications.verification_setups.verification_utils import (
    VerificationDataSaving,
    VerificationUtils,
)

# PorePy typings
number = pp.number
grid = pp.GridLike

# Physical parameters for the verification setup
sneddon_solid_constants: dict[str, number] = {
    "shear_modulus": 1e9,  # [Pa]
    "lame_lambda": 1e9,  # [Pa]
}


# -----> Data-saving
@dataclass
class SneddonSaveData:
    """Data class to save relevant results from the verification setup."""

    approx_force: np.ndarray
    """Numerical elastic force."""

    approx_interface_displacement: np.ndarray
    """Numerical displacement."""

    approx_matrix_displacement: np.ndarray
    """Numerical displacement in the matrix."""

    approx_normal_displacement_jump: np.ndarray
    """Numerical normal displacement jump obtained with MPSA."""

    crack_angle: number
    """Crack angle with respect to the horizontal axis in degrees."""

    bem_segments: number
    """Number of segments used to obtain the BEM solution."""

    exact_normal_displacement_jump: np.ndarray
    """Exact normal displacement jump given by Sneddon's solution."""


class SneddonDataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    exact_sol: SneddonExactSolution
    """Exact solution object."""

    displacement: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Displacement variable. Normally defined in a mixin instance of
    :class:`~porepy.models.momentum_balance.VariablesMomentumBalance`.

    """
    displacement_jump: Callable

    def collect_data(self):
        """Collect data for the current simulation time.

        Returns:
            SneedonSaveData object containing the results of the verification.

        """

        sd = self.mdg.subdomains()[0]


# -----> Exact solution
class SneddonExactSolution:
    """Class containing exact and BEM solutions to Sneddon's problem."""

    def __init__(self, setup):
        """Constructor of the class."""

        self.setup = setup
        """Sneddon's verification setup class."""

        self._num_bem_segments = self.setup.params("num_bem_segments", 1000)
        """Number of BEM segments used for obtaining exact solution. Default is 1000."""

    def get_crack_center(self) -> np.ndarray:
        """Get global coordinates of the crack center.

        We assume that the crack center coincides with the center of the domain.

        Returns:
            Array of ``shape=(3, )`` with the global coordinates of the crack center.

        """
        a, b = self.setup.domain_size()  # scaled [m]
        return np.array([a / 2, b / 2, 0.0])

    def get_bem_length(self) -> number:
        """Length of BEM segments in scaled [m] for a uniformly partitioned crack.

        Returns:
            Length of the BEM segment in scaled [m].

        """
        return self.setup.crack_length() / self._num_bem_segments

    def get_bem_centers(self) -> np.ndarray:
        """Compute the centers (in scaled [m]) of the BEM segments.

        Returns:
            Array of ``shape=(3, num_segments)`` with the global coordinates of the
            BEM centers.

        """
        n = self._num_bem_segments
        crack_length = self.setup.get_crack_length()  # scaled [m]
        crack_center = self.get_crack_center()  # scaled [m]
        bem_length = self.get_bem_length()  # scaled [m]
        theta = self.setup.crack_angle()  # [radians]

        x0 = crack_center[0] - 0.5 * (crack_length - bem_length) * np.cos(theta)
        y0 = crack_center[1] - 0.5 * (crack_length - bem_length) * np.sin(theta)

        xc = x0 + np.arange(n) * bem_length * np.cos(theta)
        yc = y0 + np.arange(n) * bem_length * np.sin(theta)
        zc = np.zeros(n)

        return np.array([xc, yc, zc])

    def transform_coordinates(
        self,
        bem_center: np.ndarray,
        points_in_global_coo: np.ndarray,
    ) -> np.ndarray:
        """Transform from global to local coordinates (relative to the BEM segment).

        Parameters:
            bem_center: Global coordinates of the BEM segment center in scaled [m] with
                ``shape=(3, )``.
            points_in_global_coo: global coordinates of the set of points in scaled [m]
                with ``shape=(3, num_points)``.

        Returns:
            Array of ``shape=(3, num_points)`` containing the local coordinates in
            scaled [m] for the given set of points.

        """
        theta = self.setup.crack_angle()  # [radians]

        # fmt: off
        x_bar = np.zeros_like(points_in_global_coo)
        x_bar[0] = (
                (points_in_global_coo[0] - bem_center[0]) * np.cos(theta)
                + (points_in_global_coo[1] - bem_center[1]) * np.sin(theta)
        )
        x_bar[1] = (
                - (points_in_global_coo[0] - bem_center[0]) * np.sin(theta)
                + (points_in_global_coo[1] - bem_center[1]) * np.cos(theta)
        )
        # fmt: on

        return x_bar

    def bem_contribution_to_displacement(
        self,
        normal_relative_displacement: number,
        points_in_local_coo: np.ndarray,
    ) -> np.ndarray:
        """Compute BEM segment contribution to the displacement outside the crack.

        Note that the expressions for the displacements outside the crack are only valid
        for the case when a BEM segment undergoes a *constant* relative displacement.
        In addition, for the pressurized crack problem, we further assume zero
        tangential relative displacement.

        Parameters:
            normal_relative_displacement: constant normal relative displacement in
                scaled [m] that the BEM segment undergoes. In our setting, this will
                typically correspond to the exact normal relative displacement obtained
                via Sneddon's solution.
            points_in_local_coo: points in local coordinates (relative to the BEM
                segment) in scaled [m] at which the displacement solution will be
                evaluated.

        Returns:
            Array of ``shape=(2 * num_points, )`` containing the BEM contribution to
            the displacement at the given points. Note that the array is returned as
            in flattened vector format.

        Notes:
            The expressions are given for :math:`u_x` and :math:`u_y` for an
            arbitrarily oriented BEM segment of length :math:`2l` (see Eq. 5.5.4 from
            [2]).

            Since for the pressurized crack problem, the displacement discontinuity
            in the tangential direction is zero, the expressions reduced to:

            .. math::

                u_x = D_n
                \\left[
                - (1 - 2 \\nu)  \\cos\\theta \\bar{F}_2
                - 2 (1 - \\nu) \\sin\\theta \\bar{F}_3
                - \\bar{y}
                    \\left( \\cos\\theta \\bar{F}_4 + \\sin\\theta \\bar{F}_5 \\right)
                \\right]

                u_y = D_n
                \\left[
                - (1 - 2 \\nu) \\sin \\theta \\bar{F}_2
                + 2 (1 - \\nu) \\cos\\theta \\var{F}_3
                - \\bar{y}
                    \\left( \\sin\\theta \\bar{F}_4 - \\cos\\beta \\bar{F}_5 \\right)
                \\right]

            Here, :math:`D_n` is the _constant_ relative normal displacement,
            :math:`\\bar{y}` is the local vertical coordinate, and
            :math:`\\bar{F}_2`, :math:`\\bar{F}_3`, :math:`\\bar{F}_4`, and
            :math:`\\bar{F}_5` are derivatives of the :math:`f(x,y)` function (see
            Section 5.5 of [2]).

        """
        D_n = normal_relative_displacement  # scaled [m]
        theta = self.setup.params.get("crack_angle", 0)  # [radians]
        nu = self.setup.poisson_coefficient()  # [-]
        bem_length = self.get_bem_length()  # scaled [m]
        hl = bem_length / 2  # half-length of the bem segment in scaled [m]
        coo = points_in_local_coo  # scaled [m]

        # Common term multiplying the different expressions
        c0 = 1 / (4 * np.pi * (1 - nu))

        # F2(xbar, ybar) = f_{xbar}
        F2_bar = c0 * (
            np.log(np.sqrt((coo[0] - hl) ** 2 + coo[1] ** 2))
            - np.log(np.sqrt((coo[0] + hl) ** 2 + coo[1] ** 2))
        )

        # F3(x_bar, y_bar) = f_{y_bar}
        # Note that we have to use arctan2 and not arctan. Not entirely sure  why,
        # but if arctan is employed, we get wrong results.
        F3_bar = -c0 * (
            np.arctan2(coo[1], (coo[0] - hl)) - np.arctan2(coo[1], (coo[0] + hl))
        )

        # F4(x_bar, y_bar) = f_{x_bar, y_bar}
        F4_bar = c0 * (
            coo[1] / ((coo[0] - hl) ** 2 + coo[1] ** 2)
            - coo[1] / ((coo[0] + hl) ** 2 + coo[1] ** 2)
        )

        # F5(x_bar, y_bar) = f_{x_bar, x_bar} = - f_{y_bar, y_bar}
        F5_bar = c0 * (
            (coo[0] - hl) / ((coo[0] - hl) ** 2 + coo[1] ** 2)
            - (coo[0] + hl) / ((coo[0] + hl) ** 2 + coo[1] ** 2)
        )

        # Compute components of the displacement vector
        u_x = D_n * (
            -(1 - 2 * nu) * np.cos(theta) * F2_bar
            - 2 * (1 - nu) * np.sin(theta) * F3_bar
            - coo[1] * (np.cos(theta) * F4_bar + np.sin(theta) * F5_bar)
        )

        u_y = D_n * (
            -(1 - 2 * nu) * np.sin(theta) * F2_bar
            + 2 * (1 - nu) * np.cos(theta) * F3_bar
            - coo[1] * (np.sin(theta) * F4_bar - np.cos(theta) * F5_bar)
        )

        return np.ravel(np.array([u_x, u_y]), "F")


# -----> Utilities
class SneddonUtilities(VerificationUtils):
    """Mixin class that provides useful utility methods for the verification setup."""

    solid: pp.SolidConstants
    """Solid constants object."""

    # -----> Derived physical constants
    def poisson_coefficient(self) -> number:
        """Obtain poisson coefficient from Lamé parameters."""
        lmbda = self.solid.lame_lambda()  # scaled [Pa]
        mu = self.solid.shear_modulus()  # scaled [Pa]
        return lmbda / (2 * (lmbda + mu))

    # -----> Plotting methods
    def plot_results(self):
        """Plot results."""
        ...


# -----> Boundary conditions
class SneddonBoundaryConditions(pp.momentum_balance.BoundaryConditionsMomentumBalance):
    """Mixin class that sets the boundary conditions for Sneddon's setup."""

    params: dict
    """Setup dictionary parameter.
     
     Accessed parameters: `crack_pressure``.
     
     """

    mdg: pp.MixedDimensionalGrid

    solid: pp.SolidConstants
    """Solid constants object."""

    units: pp.Units
    """Units object, containing the scaling the base magnitudes."""

    def crack_pressure(self) -> pp.number:
        """Set the crack pressure in scaled [Pa]."""
        p0 = self.params.get("crack_pressure", 1e5)  # [Pa]
        return self.solid.convert_units(p0, "Pa")

    def bc_values_mechanics(self, subdomains: list[pp.Grid]) -> pp.ad.Array:
        """Boundary values for the momentum balance.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            bc_values: Array of boundary condition values, containing the exact
            displacement obtained with the BEM procedure.

        """
        sd = self.mdg.subdomains()[0]


# -----> Solution strategy
class SneddonSolutionStrategy(pp.SolutionStrategy):
    """Class containing the solution strategy to solve Sneddon's problem."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    def __init__(self, params):
        """Class constructor."""

        super().__init__(params=params)

    def initial_condition(self) -> None:
        """Set initial conditions for the problem.

        Even though this is not a time-dependent problem, we need to prescribe the
        values of the initial traction in the crack.

        """
        super().initial_condition()  # this sets zero displacement in the matrix

        sd = self.mdg.subdomains()


# -----> Geometry
class SneddonGeometry(pp.ModelGeometry):
    """Mixin class that generates the geometry used in Sneddon's setup."""

    params: dict
    """Simulation model parameters."""

    fracture_network: pp.FractureNetwork2d
    """Fracture network. Empty in this case."""

    def crack_length(self) -> number:
        """Set crack length in scaled [m]."""
        ls = 1 / self.units.m  # length scaling
        return self.params.get("crack_length", 20) * ls

    def crack_angle(self) -> number:
        """Set crack angle in [radians]."""
        return self.params.get("crack_angle", 0)

    def domain_size(self) -> tuple[number, number]:
        """Set domain size in scaled [m]"""
        ls = 1 / self.units.m  # length scaling
        a, b = self.params.get("domain_size", (50, 50))  # [m]
        return a * ls, b * ls

    def set_fracture_network(self) -> None:
        """Set fracture network. Unit square with no fractures."""

        # Create domain
        a, b = self.domain_size()  # scaled [m]
        self.box = {"xmin": 0, "xmax": a, "ymin": 0, "ymax": b}
        # self.domain = pp.Domain({"xmin": 0, "xmax": a, "ymin": 0, "ymax": b})

        # Create fracture network
        hcl = self.crack_length() / 2  # half crack length in scaled [m]
        theta = self.crack_angle()  # [rad]

        x0 = (a / 2) - hcl * np.cos(theta)  # x-coord of initial tip in scaled [m]
        x1 = (a / 2) + hcl * np.cos(theta)  # x-coord of final tip in scaled [m]
        y0 = (b / 2) - hcl * np.sin(theta)  # y-coord of initial tip in scaled [m]
        y1 = (b / 2) + hcl * np.sin(theta)  # y-coord of final tip in scaled [m]

        points = np.array([[x0, x1], [y0, y1]])
        connections = np.array([[0], [1]])
        self.fracture_network = pp.FractureNetwork2d(points, connections, self.box)

    def mesh_arguments(self) -> dict:
        """Set mesh arguments.

        We assume that the mesh arguments passed via the ``params`` dictionary are
        given in scaled [m].
        """
        ls = 1 / self.units.m  # length scale
        default_mesh_arguments = {
            "mesh_size_frac": 2 * ls,
            "mesh_size_bound": 2 * ls,
        }
        return self.params.get("mesh_arguments", default_mesh_arguments)

    def set_md_grid(self) -> None:
        """Set mixed-dimensional grid."""
        self.mdg = self.fracture_network.mesh(self.mesh_arguments())
        domain = self.fracture_network.domain
        # TODO: Update after #809 is merged
        if isinstance(domain, np.ndarray):
            assert domain.shape == (2, 2)
            self.domain_bounds: dict[str, float] = {
                "xmin": domain[0, 0],
                "xmax": domain[1, 0],
                "ymin": domain[0, 1],
                "ymax": domain[1, 1],
            }
        else:
            assert isinstance(domain, dict)
            self.domain_bounds = domain


# -----> Mixer
class SneddonSetup(pp.momentum_balance.MomentumBalance):  # type: ignore[misc]
    """Mixer class for Sneddon's problem of crack pressurization.

    Model parameters of special relevance for this class:

        - crack_angle (pp.number): Angle of the crack in radians with respect to the
          horizontal axis. Default is 0 [radians].
        - crack_length (pp.number): Length of the crack. Default is 20 [m].
        - crack_pressure (pp.number): Pressure in [Pa] to be applied on the internal
          faces of the crack. Default is 1e5 [Pa].
        - domain_size (tuple[pp.number, pp.number]): Size of the domain. Default
          is 50 [m] x 50 [m].
        - num_bem_segments (int): Number of BEM segments used to obtain the exact
          solution. Default is 1000.
        - plot_results (bool): Whether to plot the results.
        - units (pp.Units): Object containing scaling of base magnitudes. No scaling
          applied by default.

    Accessed material constants:

        - solid:
            - lame_lambda
            - shear_modulus

    """
