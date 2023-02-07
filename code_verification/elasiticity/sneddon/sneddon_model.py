"""
This module contains an implementation of Sneddon's problem.

-> Insert explanation...
"""

from __future__ import annotations


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import porepy as pp
import os
import scipy.optimize as opt
import sympy as sym
import scipy.sparse.linalg as spsla
import math


from typing import Optional, Union
from dataclasses import dataclass


def relative_l2_error(
        grid: pp.Grid,
        true_array: np.ndarray,
        approx_array: np.ndarray,
        is_scalar: bool,
        is_cc: bool,
) -> float:
    """Compute discrete relative L2-error.
    Parameters:
        grid: Either a subdomain grid or a mortar grid.
        true_array: Array containing the true values of a given variable.
        approx_array: Array containing the approximate values of a given variable.
        is_scalar: Whether the variable is a scalar quantity. Use ``False`` for
            vector quantities. For example, ``is_scalar=True`` for pressure, whereas
            ``is_scalar=False`` for displacement.
        is_cc: Whether the variable is associated to cell centers. Use ``False``
            for variables associated to face centers. For example, ``is_cc=True``
            for pressures, whereas ``is_scalar=False`` for subdomain fluxes.
    Returns:
        Discrete relative L2-error between the true and approximated arrays.
    Raises:
        ValueError if a mortar grid is given and ``is_cc=False``.
    """
    # Sanity check
    if isinstance(grid, pp.MortarGrid) and not is_cc:
        raise ValueError("Mortar variables can only be cell-centered.")

    # Obtain proper measure
    if is_cc:
        if is_scalar:
            meas = grid.cell_volumes
        else:
            meas = grid.cell_volumes.repeat(grid.dim)
    else:
        assert isinstance(grid, pp.Grid)
        if is_scalar:
            meas = grid.face_areas
        else:
            meas = grid.face_areas.repeat(grid.dim)

    # Compute error
    numerator = np.sqrt(np.sum(meas * np.abs(true_array - approx_array) ** 2))
    denominator = np.sqrt(np.sum(meas * np.abs(true_array) ** 2))

    return numerator / denominator


def get_bc_values_michele(
        sd_rock: pp.Grid,
        G: float,
        poi: float,
        p0: float,
        a: float,
        n: int,
        domain_size: tuple[float, float],
        theta: float
):

    def compute_eta(pointset_centers, point):
        """
        Compute the distance of bem segments centers to the
        fracture centre.

        Parameter
        ---------
        pointset_centers: array containing centers of bem segments
        point: fracture centre, middle point of the square domain

        """
        return pp.geometry.distances.point_pointset(pointset_centers, point)

    def get_bem_centers(a, h, n, theta, center):
        """
        Compute coordinates of the centers of the bem segments

        Parameter
        ---------
        a: half fracture length
        h: bem segment length
        n: number of bem segments
        theta: orientation of the fracture
        center: center of the fracture
        """
        bem_centers = np.zeros((3, n))
        x_0 = center[0] - (a - 0.5 * h) * np.sin(theta)
        y_0 = center[1] - (a - 0.5 * h) * np.cos(theta)
        for i in range(0, n):
            bem_centers[0, i] = x_0 + i * h * np.sin(theta)
            bem_centers[1, i] = y_0 + i * h * np.cos(theta)

        return bem_centers

    def analytical_displacements(a, eta, p0, mu, nu):
        """
        Compute Sneddon's analytical solution for the pressurized crack
        problem in question.

        Parameter
        ---------
        a: half fracture length
        eta: distance from fracture centre
        p0: pressure
        mu: shear modulus
        nu: poisson ratio
        """
        cons = (1 - nu) / mu * p0 * a * 2
        return cons * np.sqrt(1 - np.power(eta / a, 2))

    def transform(xc, x, alpha):
        """
        Coordinate transofrmation for the BEM method

        Parameter
        ---------
        xc: coordinates of BEM segment centre
        x: coordinates of boundary faces
        alpha: fracture orientation
        """
        x_bar = np.zeros_like(x)
        x_bar[0, :] = (x[0, :] - xc[0]) * np.cos(alpha) + (x[1, :] - xc[1]) * np.sin(
            alpha)
        x_bar[1, :] = - (x[0, :] - xc[0]) * np.sin(alpha) + (x[1, :] - xc[1]) * np.cos(
            alpha)
        return x_bar

    def get_bc_val(g, bound_faces, xf, h, poi, alpha, du):
        """
        Compute analytical displacement using the BEM method for the pressurized crack
        problem in question.

        Parameter
        ---------
        g: grid bucket
        bound_faces: boundary faces
        xf: coordinates of boundary faces
        h: bem segment length
        poi: Poisson ratio
        alpha: fracture orientation
        du: Sneddon's analytical relative normal displacement
        """
        f2 = np.zeros(bound_faces.size)
        f3 = np.zeros(bound_faces.size)
        f4 = np.zeros(bound_faces.size)
        f5 = np.zeros(bound_faces.size)

        u = np.zeros((g.dim, g.num_faces))

        m = 1 / (4 * np.pi * (1 - poi))

        f2[:] = m * (np.log(np.sqrt((xf[0, :] - h) ** 2 + xf[1] ** 2))
                     - np.log(np.sqrt((xf[0, :] + h) ** 2 + xf[1] ** 2)))

        f3[:] = - m * (np.arctan2(xf[1, :], (xf[0, :] - h))
                       - np.arctan2(xf[1, :], (xf[0, :] + h)))

        f4[:] = m * (xf[1, :] / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
                     - xf[1, :] / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2))

        f5[:] = m * ((xf[0, :] - h) / ((xf[0, :] - h) ** 2 + xf[1, :] ** 2)
                     - (xf[0, :] + h) / ((xf[0, :] + h) ** 2 + xf[1, :] ** 2))

        u[0, bound_faces] = du * (-(1 - 2 * poi) * np.cos(alpha) * f2[:]
                                  - 2 * (1 - poi) * np.sin(alpha) * f3[:]
                                  - xf[1, :] * (np.cos(alpha) * f4[:] + np.sin(
                    alpha) * f5[:]))
        u[1, bound_faces] = du * (-(1 - 2 * poi) * np.sin(alpha) * f2[:]
                                  + 2 * (1 - poi) * np.cos(alpha) * f3[:]
                                  - xf[1, :] * (np.sin(alpha) * f4[:] - np.cos(
                    alpha) * f5[:]))

        return u

    def assign_bem(g, h, bound_faces, theta, bem_centers, u_a, poi):

        """
        Compute analytical displacement using the BEM method for the pressurized crack
        problem in question.

        Parameter
        ---------
        g: grid bucket
        h: bem segment length
        bound_faces: boundary faces
        theta: fracture orientation
        bem_centers: bem segments centers
        u_a: Sneddon's analytical relative normal displacement
        poi: Poisson ratio
        """

        bc_val = np.zeros((g.dim, g.num_faces))

        alpha = np.pi / 2 - theta

        bound_face_centers = g.face_centers[:, bound_faces]

        for i in range(0, u_a.size):
            new_bound_face_centers = transform(bem_centers[:, i],
                                               bound_face_centers, alpha)

            u_bound = get_bc_val(g, bound_faces, new_bound_face_centers,
                                 h, poi, alpha, u_a[i])

            bc_val += u_bound

        return bc_val

    # Define boundary regions
    bound_faces = sd_rock.get_all_boundary_faces()
    box_faces = sd_rock.get_boundary_faces()
    length, height = domain_size

    h = 2 * a / n
    center = np.array([length / 2, height / 2, 0])
    bem_centers = get_bem_centers(a, h, n, theta, center)
    eta = compute_eta(bem_centers, center)
    u_a = analytical_displacements(a, eta, p0, G, poi)
    u_bc = assign_bem(sd_rock, h / 2, box_faces, theta, bem_centers, u_a, poi)

    return u_bc.ravel("F")


class BEM:
    """Parent class for BEM solution"""

    def __init__(self, params: dict) -> None:
        """Constructor of the BEM class.

        Parameters:
            params: SneddonSetup parameters.

        """
        self.params = params

    def get_crack_center(self) -> np.ndarray:
        """Get global coordinates of the crack center.

        We assume that the crack center coincides with the center of the domain.

        Returns:
            ndarray of ``shape (3, )`` with the global coordinates of the crack center.

        """
        lx, ly = self.params["domain_size"]  # [m]
        return np.array([lx/2, ly/2, 0.0])

    def get_bem_length(self, num_bem_segments: int | None = None) -> float:
        """Compute length of BEM segments for a uniformly partitioned crack.

        Parameters:
            num_bem_segments: Number of BEM segments. If not specified, the number of
                BEM segments given in ``self.params["num_bem_segments"]`` will be used.

        Returns:
            Length of the BEM segment.

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        return self.params["crack_length"] / n

    def get_bem_centers(self, num_bem_segments: int | None = None) -> np.ndarray:
        """Compute the centers of the BEM segments.

        Parameters:
            num_bem_segments: Number of BEM segments. If not specified, the number of
                BEM segments given in ``self.params["num_bem_segments"]`` will be used.

        Returns:
            ndarray of ``shape=(3, num_segments)`` with the global coordinates of the
            BEM centers.

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        b = self.params["crack_length"]  # [m]
        cc = self.get_crack_center()  # [m]
        dl = self.get_bem_length(n)  # [m]
        beta = self.params["crack_angle"]  # [radians]

        x0 = cc[0] - 0.5 * (b - dl) * np.cos(beta)
        y0 = cc[1] - 0.5 * (b - dl) * np.sin(beta)

        xc = x0 + np.arange(n) * dl * np.cos(beta)
        yc = y0 + np.arange(n) * dl * np.sin(beta)
        zc = np.zeros(n)

        return np.array([xc, yc, zc])

    def get_distance_from_crack_center(self, points: np.ndarray) -> np.ndarray:
        """Compute distance from a set of points to the fracture center.

        Parameters:
            points (shape=(3, num_points)): global coordinates of the set of points.

        Returns:
            ndarray of ``shape=(num_points, )`` containing the distance from the set of
            points to the crack center.

        """
        crack_center = self.get_crack_center()

        return pp.distances.point_pointset(points, crack_center)

    def transform_coordinates(
        self,
        bem_center: np.ndarray,
        points_in_global_coo: np.ndarray,
    ) -> np.ndarray:
        """Transform from global to local coordinates (relative to the BEM segment).

        Parameters:
            bem_center (shape=(3, )): global coordinates of the BEM segment center.
            points_in_global_coo (shape=(3, num_points)): global coordinates of the
                set of points.

        Returns:
            ndarray of ``shape=(3, num_points)`` containing the local coordinates
            for the given set of points.

        """
        beta = self.params["crack_angle"]  # [radians]
        coo = points_in_global_coo  # [m]

        # fmt: off
        x_bar = np.zeros_like(coo)
        x_bar[0] = (
                (coo[0] - bem_center[0]) * np.cos(beta)
                + (coo[1] - bem_center[1]) * np.sin(beta)
        )
        x_bar[1] = (
                -(coo[0] - bem_center[0]) * np.sin(beta)
                + (coo[1] - bem_center[1]) * np.cos(beta)
        )
        # fmt: on

        return x_bar

    def bem_contribution_to_displacement(
        self,
        normal_relative_displacement: float,
        points_in_local_coo: np.ndarray,
        num_bem_segments: int | None = None,
    ) -> np.ndarray:
        """Compute BEM segment contribution to the displacement outside the crack.

        Note that the expressions for the displacements outside the crack are
        only valid for the case when a BEM segment undergoes a *constant* relative
        displacement. In addition, for the pressurized crack problem, we further assume
        zero tangential relative displacement.

        Parameters:
            normal_relative_displacement: constant normal relative displacement
                (in meters) that the BEM segment undergoes. In our setting, this will
                typically correspond to the exact normal relative displacement obtained
                via Sneddon's solution [1].
            points_in_local_coo (shape=(3, num_points)): points given in local
                coordinates (relative to the BEM segment) at which the displacement
                solution will be evaluated.
            num_bem_segments: Number of BEM segments. If not specified, the number of
                BEM segments given in ``self.params["num_bem_segments"]`` will be used.

        Returns:
            Array of ``shape=(2 * num_points, )`` containing the BEM contribution to
            the displacement at the given points. Note that the array is returned as
            a flattened vector.

        Notes:
            The expressions are given for `u_x` and `u_y` for an arbitrarily oriented
            BEM segment of length `2a` (see Eq. 5.5.4 from [2]).

            Recall that for the pressurized crack problem, the displacement
            discontinuity in the tangential direction is zero.

            The expressions can therefore be written as:

            u_x = D_n * [
                - (1 - 2*nu) * cos(beta) * F2_bar
                - 2 * (1 - nu) * sin(beta) * F3_bar
                - y_bar * (cos(beta) * F4_bar + sin(beta) * F5_bar)
            ]

            u_y = D_n * [
                - (1 - 2 * nu) * sin(beta) * F2_bar
                + 2 * (1 - nu) * cos(beta) * F3_bar
                - y_bar * (sin(beta) * F4_bar - cos(beta) * F5_bar)
            ]

            Here, D_n is the constant relative normal displacement, y_bar is the
            local vertical coordinate, beta is the angle (in radians) measured with
            respect to the horizontal axis, and F2_bar, F3_bar, F4_bar, and F5_bar
            are derivatives of the `f(x,y)` function (see Section 5.5 of [2]).

        References:
            [1] Sneddon, I.N.: Fourier Transforms. McGraw Hill Book Co, Inc.,
              New York (1951).

            [2] Crouch, S.L., Starfield, A.: Boundary Element Methods in Solid
              Mechanics: With Applications in Rock Mechanics and Geological
              Engineering. Allen & Unwin, London (1982).

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        D_n = normal_relative_displacement  # [m]
        beta = self.params["crack_angle"]  # [radians]
        nu = self.params["poisson_coefficient"]  # [-]
        dl = self.get_bem_length(num_bem_segments=n)  # [m]
        a = dl/2  # half-length of the bem segment
        coo = points_in_local_coo

        # Constant term multiplying the different expressions
        c0 = 1 / (4 * np.pi * (1 - nu))

        # F2(xbar, ybar) = f_{xbar}
        F2_bar = c0 * (
            np.log(np.sqrt((coo[0] - a) ** 2 + coo[1] ** 2))
            - np.log(np.sqrt((coo[0] + a) ** 2 + coo[1] ** 2))
        )

        # F3(x_bar, y_bar) = f_{y_bar}
        # Note that we have to use arctan2 and not arctan for computing here.
        # Not entirely why, but if arctan is employed, we get wrong results.
        F3_bar = -c0 * (
            np.arctan2(coo[1],  (coo[0] - a))
            - np.arctan2(coo[1], (coo[0] + a))
        )

        # F4(x_bar, y_bar) = f_{x_bar, y_bar}
        F4_bar = c0 * (
                coo[1] / ((coo[0] - a) ** 2 + coo[1] ** 2)
                - coo[1] / ((coo[0] + a) ** 2 + coo[1] ** 2)
        )

        # F5(x_bar, y_bar) = f_{x_bar, x_bar} = - f_{y_bar, y_bar}
        F5_bar = c0 * (
                (coo[0] - a) / ((coo[0] - a) ** 2 + coo[1] ** 2)
                - (coo[0] + a) / ((coo[0] + a) ** 2 + coo[1] ** 2)
        )

        # Compute components of the displacement vector
        u_x = D_n * (
                -(1 - 2 * nu) * np.cos(beta) * F2_bar
                - 2 * (1 - nu) * np.sin(beta) * F3_bar
                - coo[1] * (np.cos(beta) * F4_bar + np.sin(beta) * F5_bar)
        )

        u_y = D_n * (
                -(1 - 2 * nu) * np.sin(beta) * F2_bar
                + 2 * (1 - nu) * np.cos(beta) * F3_bar
                - coo[1] * (np.sin(beta) * F4_bar - np.cos(beta) * F5_bar)
        )

        u = np.ravel(np.array([u_x, u_y]), "F")

        return u

    def bem_contribution_to_stress(
        self,
        normal_relative_displacement: float,
        points_in_local_coo: np.ndarray,
        num_bem_segments: Optional[int]=None,
    ) -> list[list[np.ndarray]]:
        """Compute BEM segment contribution to symmetric stress outside the crack"""

        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        D_n = normal_relative_displacement  # [m]
        beta = self.params["crack_angle"]  # [radians]
        mu = self.params["mu_lame"]  # [Pa]
        nu = self.params["poisson_coefficient"]  # [-]
        dl = self.get_bem_length(num_bem_segments=n)  # [m]
        a = dl/2  # half-length of the bem segment
        coo = points_in_local_coo

        # Constant term multiplying the different expressions
        c0 = 1 / (4 * np.pi * (1 - nu))

        # F5(x_bar, y_bar) = f_{x_bar, x_bar} = - f_{y_bar, y_bar}
        F5_bar = c0 * (
                (coo[0] - a) / ((coo[0] - a) ** 2 + coo[1] ** 2)
                - (coo[0] + a) / ((coo[0] + a) ** 2 + coo[1] ** 2)
        )

        # F6{xbar, ybar} = f_{x_bar, y_bar, y_bar}
        F6_bar = c0 * (
            ((coo[0] - a)**2 - coo[1]**2) / ((coo[0] - a)**2 + coo[1]**2)**2
            - ((coo[0] + a)**2 - coo[1]**2) / ((coo[0] + a)**2 + coo[1]**2)**2
        )

        # F7{xbar, ybar} = f_{y_bar, y_bar, y_bar}
        F7_bar = 2 * c0 * coo[1] * (
            (coo[0] - a) / ((coo[0] - a)**2 + coo[1]**2)**2
            - (coo[0] + a) / ((coo[0] + a)**2 + coo[1]**2)**2
        )

        # Compute components of the stress tensor
        sigma_xx = 2 * mu * D_n * (
            - F5_bar
            + coo[1] * (np.sin(2 * beta * F6_bar) + np.cos(2 * beta * F7_bar))
        )

        sigma_yy = 2 * mu * D_n * (
            - F5_bar
            + coo[1] * (np.sin(2 * beta * F6_bar) - np.cos(2 * beta * F7_bar))
        )

        sigma_xy = 2 * mu * D_n * (
            -coo[1] * (np.cos(2 * beta * F6_bar) - np.sin(2 * beta * F7_bar))
        )

        sigma_yx = sigma_xy

        sigma = [[sigma_xx, sigma_xy], [sigma_yx, sigma_yy]]

        return sigma

    def compute_displacement(
        self,
        points_in_global_coo: np.ndarray,
        num_bem_segments: Optional[int]=None,
    ) -> np.ndarray:
        """Compute displacements (outside of crack) at the given points using BEM.

        Parameters:
            points_in_global_coo (shape=(3, num_points )): Global coordinates of the
                points at which displacements will be computed.
            num_bem_segments: Number of BEM segments. If not specified, the number of
                BEM segments given in ``self.params["num_bem_segments"]`` will be used.

        Returns:
            ndarray (shape=(2 * num_points, )). Computed displacement at the given
            ``points``. The array is returned in flattened vector format.

        """
        # Get number of bem segments used to discretize the crack
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        # Get bem centers in global coordinates
        bem_centers = self.get_bem_centers(n)

        # Compute distance from bem centers to the crack center
        eta = self.get_distance_from_crack_center(bem_centers)

        # Get exact relative normal displacement for each bem segment
        u_n = self.exact_relative_normal_displacement(eta)

        # BEM loop
        num_points = points_in_global_coo.shape[1]
        u = np.zeros(2 * num_points)
        for bem in range(n):

            # Transform coordinates relative to each bem center
            points_in_local_coo = self.transform_coordinates(
                bem_center=bem_centers[:, bem],
                points_in_global_coo=points_in_global_coo
            )

            # Get bem contribution to the displacement at the given set of points
            u_bem = self.bem_contribution_to_displacement(
                normal_relative_displacement=u_n[bem],
                points_in_local_coo=points_in_local_coo,
                num_bem_segments=n
            )

            # Add contribution
            u += u_bem

        return u

    def compute_stress(
        self,
        points_in_global_coo: np.ndarray,
        num_bem_segments: Optional[int]=None,
    ):
        # Get number of bem segments used to discretize the crack
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        # Get bem centers in global coordinates
        bem_centers = self.get_bem_centers(n)

        # Compute distance from bem centers to the crack center
        eta = self.get_distance_from_crack_center(bem_centers)

        # Get exact relative normal displacement for each bem segment
        u_n = self.exact_relative_normal_displacement(eta)

        # BEM loop
        num_p = points_in_global_coo.shape[1]
        zeros = np.zeros(num_p)
        sigma = [[zeros, zeros], [zeros, zeros]]
        for bem in range(n):
            # Transform coordinates relative to each bem center
            points_in_local_coo = self.transform_coordinates(
                bem_center=bem_centers[:, bem],
                points_in_global_coo=points_in_global_coo
            )

            # Get bem contribution to the displacement at the given set of points
            sigma_bem = self.bem_contribution_to_stress(
                normal_relative_displacement=u_n[bem],
                points_in_local_coo=points_in_local_coo,
                num_bem_segments=n
            )

            # Add contribution
            sigma[0][0] += sigma_bem[0][0]
            sigma[0][1] += sigma_bem[0][1]
            sigma[1][0] += sigma_bem[1][0]
            sigma[1][1] += sigma_bem[1][1]

        return sigma

    def exact_relative_normal_displacement(self, eta: np.ndarray) -> np.ndarray:
        """Compute exact relative normal displacement [1].

        Parameters:
            eta (shape=(num_points, )): array containing the distances from a set of
                points on the line crack relative to the crack center.

        Returns:
            Array of ``shape=(num_points, )`` containing the exact relative normal
            displacement for each ``eta``.

        References:
            [1] Sneddon, I.N.: Fourier Transforms. McGraw Hill Book Co, Inc.,
              New York (1951).

       """
        nu_s = self.params["poisson_coefficient"]  # [-]
        mu_s = self.params["mu_lame"]  # [Pa]
        p0 = self.params["crack_pressure"]  # [Pa]
        frac_length = self.params["crack_length"]  # [m]
        a = frac_length / 2

        c0 = (1 - nu_s) / mu_s * p0 * frac_length
        u_n = c0 * np.sqrt(1 - np.power(eta / a, 2))

        return u_n

    def bem_relative_normal_displacement(
        self, num_bem_segments: int | None = None
    ) -> np.ndarray:
        """Numerical approximation of the relative normal displacement using BEM.

        We follow the procedure given in Section 5.3 from [1].

        Parameters:
            num_bem_segments: Number of BEM segments used to generate the approximate
                solution. If not specified, "self.bem_num" is employed.

        Returns:
            Approximate relative normal displacement for the pressurized crack
            problem using the Boundary Element Method. Shape is (num_bem_segments, ).

        References:
            [1] Crouch, S.L., Starfield, A.: Boundary Element Methods in Solid
              Mechanics: With Applications in Rock Mechanics and Geological
              Engineering. Allen & Unwin, London (1982).

        """
        if num_bem_segments is None:
            n = self.params["num_bem_segments"]
        else:
            n = num_bem_segments

        p = self.params["crack_pressure"]  # [Pa]
        mu_s = self.params["mu_lame"]  # [Pa]
        nu_s = self.params["poisson_coefficient"]  # [-]
        dl = self.get_bem_length(n)  # [m]
        c = self.get_bem_centers(n)  # [m]

        # Transform coordinates
        # FIXME: Do we need to transform the coordinates?
        # c_bar = self.coordinate_transform(c)

        # Compute matrix of influence coefficients A_{i,j}
        a0 = -mu_s / (np.pi * (1 - nu_s))
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = a0 * ((dl / 2) / ((c[0][i] - c[0][j]) ** 2 - (dl / 2) ** 2))

        # Vector of constants
        b = p * np.ones(n)

        # Solve linear system
        x = spsla.spsolve(A, b)

        return x


class SneddonSetup(pp.ContactMechanics):
    """Parent class for Sneddon's problem."""

    def __init__(self, params):
        """Constructor for Sneddon's class."""

        def set_default_params(keyword: str, value: object) -> None:
            """
            Set default parameters if a keyword is absent in the `params` dictionary.

            Parameters:
                keyword: Parameter keyword, e.g., "mesh_size".
                value: Value of `keyword`, e.g., 1.0.

            """
            if keyword not in params.keys():
                params[keyword] = value

        # Define default parameters
        default_params: list[tuple] = [
            ("domain_size", (50.0, 50.0)),  # [m]
            ("crack_angle", 0),  # [radians]
            ("crack_length", 20.0),  # [m]
            ("crack_pressure", 1e-4),  # [GPa]
            ("poisson_coefficient", 0.25),  # [-]
            ("mesh_size", 2.0),  # [m]
            ("mu_lame", 1.0),  # [GPa]
            ("num_bem_segments", 1000),
            ("plot_results", True),
            ("use_ad", True),  # only `use_ad = True` is supported
        ]

        # Set default values
        for key, val in default_params:
            set_default_params(key, val)
        super().__init__(params)

        # ad sanity check
        if not self.params["use_ad"]:
            raise ValueError("Model only valid when ad is used.")

        # Store other useful parameters

        # Crack center
        lx, ly = self.params["domain_size"]
        self.params["crack_center"] = np.array([lx / 2, ly / 2, 0.0])

        # TODO: Store lambda lame parameter

        # Create a BEM dictionary to store BEM-related quantities
        self.bem = BEM(self.params)

    def create_grid(self) -> None:
        """Create mixed-dimensional grid.

        The fracture will be placed at the center of the domain at an angle `theta`
        measured with respect to the horizontal axis.

        """
        # Retrieve data
        lx, ly = self.params["domain_size"]  # [m]
        h = self.params["mesh_size"]  # [m]
        beta = self.params["crack_angle"]  # [radians]
        crack_length = self.params["crack_length"]  # [m]
        a = crack_length / 2  # [m]

        # Create bounding box
        self.box = {"xmin": 0.0, "xmax": lx, "ymin": 0.0, "ymax": ly}

        # Create fracture network
        x_0 = (lx / 2) - a * np.cos(beta)  # initial tip coo in x
        x_1 = (lx / 2) + a * np.cos(beta)  # final tip coo in x
        y_0 = (ly / 2) - a * np.sin(beta)  # initial tip coo in y
        y_1 = (ly / 2) + a * np.sin(beta)  # final tip coo in y
        frac_pts = np.array([[x_0, x_1],
                             [y_0, y_1]])
        frac_edges = np.array([[0],
                               [1]])
        network_2d = pp.FractureNetwork2d(frac_pts, frac_edges, self.box)

        # Create mixed-dimensional grid
        mesh_args = {"mesh_size_bound": h, "mesh_size_frac": h}
        self.mdg = network_2d.mesh(mesh_args)

        # Set projections
        pp.contact_conditions.set_projections(self.mdg)

    def _initial_condition(self):
        super()._initial_condition()

        for sd, data in self.mdg.subdomains(return_data=True):
            if sd.dim == 1:
                p0 = self.params["crack_pressure"] * sd.cell_volumes
                initial_tangential_force = np.zeros(sd.num_cells)
                initial_normal_force = p0
                initial_traction = np.array(
                    [initial_tangential_force, initial_normal_force]
                ).ravel("F")
                data[pp.STATE][self.contact_traction_variable] = initial_traction
                data[pp.STATE][pp.ITERATE][self.contact_traction_variable] = initial_traction

    def _friction_coefficient(self, sd: pp.Grid) -> np.ndarray:
        """Set friction coefficient."""
        return 0.5 * np.ones(sd.num_cells)

    def _bc_type(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        super()._bc_type(sd)
        if sd.dim == 2:
            faces = sd.get_all_boundary_faces()
            return pp.BoundaryConditionVectorial(sd, faces, "dir")
    def _bc_values(self, sd: pp.Grid) -> np.ndarray:
        """Set boundary condition values"""
        super()._bc_values(sd)
        if sd.dim == 2:
            return setup.get_boundary_conditions()
        # if sd.dim == 2:
        #     return get_bc_values_michele(
        #         sd_rock=sd,
        #         G=self.params["mu_lame"],
        #         poi=self.params["poisson_coefficient"],
        #         p0=self.params["crack_pressure"],
        #         a=self.params["crack_length"]/2,
        #         n=self.params["num_bem_segments"],
        #         domain_size=self.params["domain_size"],
        #         theta=math.radians(90-self.params["crack_angle"])
        #     )

    def _stiffness_tensor(self, sd: pp.Grid) -> pp.FourthOrderTensor:
        """Set stiffness tensor"""
        mu_s = self.params["mu_lame"]  # [Pa]
        nu_s = self.params["poisson_coefficient"]  # [-]
        lmbda_s = (2 * mu_s * nu_s) / (1 - 2 * nu_s)  # [Pa]

        lam = lmbda_s * np.ones(sd.num_cells)
        mu = mu_s * np.ones(sd.num_cells)

        return pp.FourthOrderTensor(mu, lam)

    def after_simulation(self) -> None:
        """Method to be called once the simulation has finished."""
        if self.params["plot_results"]:
            self.plot_results()

    # -----> Methods relate to BEM

    def get_boundary_conditions(self) -> np.ndarray:

        sd_rock = self.mdg.subdomains()[0]
        sides = self._domain_boundary_sides(sd_rock)
        # bc_faces = sides.all_bf
        bc_faces = sd_rock.get_boundary_faces()
        bc_coo = sd_rock.face_centers[:, bc_faces]

        u_bc = self.bem.compute_displacement(bc_coo)
        bc_vals = np.zeros(sd_rock.dim * sd_rock.num_faces)
        bc_vals[::sd_rock.dim][bc_faces] = u_bc[::sd_rock.dim]
        bc_vals[1::sd_rock.dim][bc_faces] = u_bc[1::sd_rock.dim]

        return bc_vals

    # -----> Methods related to the analytical solution
    def exact_normal_displacement_jump(self, eta: np.ndarray) -> np.ndarray:
        """Compute exact relative normal displacement jump.

        Sneddon, I. (1952). Fourier transforms. Bull. Amer. Math. Soc, 58, 512-513.

        Parameters:
            eta: array containing the distances from a set of points `p` to the
                crack center. Shape is (num_points, ).

        Returns:
            Exact normal relative displacement jump for each ``eta``.

        """
        nu_s = self.params["poisson_coefficient"]  # [-]
        mu_s = self.params["mu_lame"]  # [Pa]
        p0 = self.params["crack_pressure"]  # [Pa]
        frac_length = self.params["crack_length"]  # [m]
        half_length = frac_length / 2

        c0 = (1 - nu_s) / mu_s * p0 * frac_length
        u_jump = c0 * (1 - (eta / half_length) ** 2) ** 0.5

        return u_jump

    # ------> Helper methods
    def distance_from_crack_center(self, point_set: np.ndarray) -> np.ndarray:
        """Compute distance from a set of points to the fracture center.

        Args:
            point_set: coordinates of the set of points. Shape is (3, num_points).

        Returns:
            Distance from the set of point to the fracture center. Shape is
                (num_points, ).

        """
        length, height = self.params["domain_size"]  # [m]

        frac_center = np.array([[length / 2], [height / 2], [0.0]])

        return pp.distances.point_pointset(frac_center, point_set)

    # -----> Plotting methods
    def plot_results(self) -> None:
        """Plot results."""

        # Relative displacement
        self._plot_relative_displacement()

    def _plot_relative_displacement(self):
        """Plot relative displacement as a function of distance from crack center"""

        # Generate exact points
        # TODO: Do we need to rotate the coordinates?
        sd_frac = self.mdg.subdomains()[1]
        crack_length = self.params["crack_length"]
        x_min = np.min(sd_frac.nodes[0])
        x_max = np.max(sd_frac.nodes[0])
        y_min = np.min(sd_frac.nodes[1])
        y_max = np.max(sd_frac.nodes[1])
        x_ex = np.linspace(x_min, x_max, 100)
        y_ex = np.linspace(y_min, y_max, 100)
        z_ex = np.zeros(100)
        points_ex = np.array([x_ex, y_ex, z_ex])
        eta_ex = self.distance_from_crack_center(points_ex)
        u_jump_ex = self.exact_normal_displacement_jump(eta_ex)

        fig, ax = plt.subplots(figsize=(9, 8))

        # Plot exact solution
        ax.plot(
            x_ex / crack_length,
            u_jump_ex / crack_length,
            linewidth=3,
            color="blue",
            alpha=0.4,
        )

        # Plot BEM solution
        bem_elements = 20
        bem_centers = self.bem.get_bem_centers(bem_elements)
        bem_sol = self.bem.bem_relative_normal_displacement(bem_elements)
        ax.plot(
            bem_centers[0] / crack_length,
            bem_sol / crack_length,
            marker="o",
            markersize=6,
            linewidth=0,
            color="orange",
        )
        # plt.step(
        #     bem_centers[0] / crack_length,
        #     bem_sol / crack_length,
        #     where="mid",
        #     color="orange",
        # )
        # plt.step(
        #     np.array([2.0, bem_centers[0][0] / crack_length]),
        #     np.array([0, bem_sol[0] / crack_length]),
        #     where="pre",
        #     color="orange",
        # )
        # plt.step(
        #     np.array([bem_centers[0][-1] / crack_length, 3.0]),
        #     np.array([bem_sol[-1] / crack_length, 0]),
        #     where="post",
        #     color="orange",
        #     alpha=0.7,
        # )

        # Label plot
        plt.plot(
            [],
            [],
            linewidth=4,
            color="blue",
            alpha=0.4,
            label="Exact, Sneddon (1951).",
        )
        plt.plot(
            [],
            [],
            linewidth=4,
            color="orange",
            marker="o",
            markersize=8,
            label=f"BEM ({bem_sol.size} elements), Crouch & Starfield (1983).",
        )

        # Set labels and legends
        ax.set_xlabel(r"Non-dimensional horizontal distance, " r"$x~/~b$", fontsize=13)
        ax.set_ylabel(
            r"Relative normal displacement, " r"$\hat{u}_n(x)~/~b$", fontsize=13
        )
        ax.legend(fontsize=13)

        if not os.path.exists("out"):
            os.makedirs("out")
        plt.savefig("out/" + "sneddon" + ".pdf", bbox_inches="tight")
        plt.gcf().clear()


#%% Runner
params = {
    "crack_angle": 0,
    "plot_results": False,
    "num_bem_segments": 1000,
    "domain_size": (50, 50),
    "mesh_size": 2,
}
setup = SneddonSetup(params=params)
setup.prepare_simulation()

mdg = setup.mdg

sd_rock = mdg.subdomains()[0]
sd_frac = mdg.subdomains()[1]
intf = mdg.interfaces()[0]

data_rock = setup.mdg.subdomain_data(sd_rock)
data_frac = setup.mdg.subdomain_data(sd_frac)
data_intf = setup.mdg.interface_data(intf)

#%% Extract subsystem
eq_reduced = setup._eq_manager.subsystem_equation_manager(
    eq_names=["momentum", "force_balance"],
    variables=[setup._ad.displacement, setup._ad.interface_displacement]
)
A_reduced, b_reduced = eq_reduced.assemble()

x = spsla.spsolve(A_reduced, b_reduced)
u_rock = x[: 2 * sd_rock.num_cells]
u_intf = x[2*sd_rock.num_cells :]

u_intf_t = u_intf[::2]
u_intf_n = u_intf[1::2]

jump_u = (
            intf.mortar_to_secondary_avg(nd=2)
            * intf.sign_of_mortar_sides(nd=2)
            * u_intf
    )

jump_mpsa = np.abs(jump_u[1::2]) / setup.params["crack_length"]

# u_bem = setup.bem.compute_displacement(sd_rock.cell_centers)
# pp.plot_grid(sd_rock, u_bem[::2], plot_2d=True, title="u_bem_x", linewidth=0)
# pp.plot_grid(sd_rock, u_bem[1::2], plot_2d=True, title="u_bem_y", linewidth=0)

# u_mpsa = data_rock[pp.STATE][setup.displacement_variable]
# pp.plot_grid(sd_rock, u_mpsa[::2], plot_2d=True, title="u_mpsa_x", linewidth=0)
# pp.plot_grid(sd_rock, u_mpsa[1::2], plot_2d=True, title="u_mpsa_y", linewidth=0)


# %%
bc_vals = setup.get_boundary_conditions()
sides = setup._domain_boundary_sides(sd_rock)
bc_faces = sides.all_bf
bc_vals_x = bc_vals[::2]
bc_vals_y = bc_vals[1::2]

bc_jv_x = bc_vals_x[bc_faces]
bc_jv_y = bc_vals_y[bc_faces]

# %% Michele's solution
import math
bc_michele = get_bc_values_michele(
    sd_rock = sd_rock,
    G = setup.params["mu_lame"],
    poi = setup.params["poisson_coefficient"],
    p0 = setup.params["crack_pressure"],
    a = setup.params["crack_length"] / 2,
    n = setup.params["num_bem_segments"],
    domain_size = setup.params["domain_size"],
    theta = math.radians(90 - setup.params["crack_angle"])
)

bc_ms_x = bc_michele[0::2][bc_faces]
bc_ms_y = bc_michele[1::2][bc_faces]

#%% Plot boundary values
# plt.figure()
# plt.plot(bc_ms_x, "r", label="ms_x")
# plt.plot(bc_jv_x, "b.", label="jv_x")
# plt.plot(bc_ms_y, "g", label="ms_y")
# plt.plot(bc_jv_y, "m.", label="jv_y")
# plt.show()

# %%
# a = setup.params["crack_length"] / 2
# length, height = setup.params["domain_size"]
# theta = 0
# theta = math.radians(90-theta)
#
# y_0 = height / 2 - a * np.cos(theta)
# x_0 = length / 2 - a * np.sin(theta)
# y_1 = height / 2 + a * np.cos(theta)
# x_1 = length / 2 + a * np.sin(theta)
#
# frac_pts = np.array([[x_0, y_0], [x_1, y_1]]).T
# frac_edges = np.array([[0,1]]).T

# %% Plot sparsity of the linear system
# plt.spy(setup.linear_system[0], markersize=1)
# plt.show()

#%% Extract blocks from the systems of equations

# for i in range(1):
#
#     # Retrieve linear system
#     # setup.prepare_simulation()
#     setup.assemble_linear_system()
#     A_full, b_full = setup.linear_system
#
#     # Extract degrees of freedom from the full system
#     dof_rock = setup.dof_manager.grid_and_variable_to_dofs(sd_rock, "u")
#     dof_frac = setup.dof_manager.grid_and_variable_to_dofs(sd_frac, "contact_traction")
#     dof_intf = setup.dof_manager.grid_and_variable_to_dofs(intf, "mortar_u")
#
#     # Obtain new indices for reduced systems
#     reduce_ind = np.hstack([dof_rock, dof_intf])
#     rock_ind_reduced = np.arange(dof_rock.size)
#     mortar_ind_reduced = np.arange(dof_rock.size, reduce_ind.size)
#
#     # Create vector of known traction
#     normal_force_frac = setup.params["crack_pressure"] * sd_frac.cell_volumes
#     shear_force_frac = np.zeros(sd_frac.num_cells)
#     force_frac = np.array([shear_force_frac, normal_force_frac]).ravel("F")
#
#     # Create reduced linear system
#     A_rm = A_full[reduce_ind][:, reduce_ind]  # A_{rock, mortar}
#     b_rm = b_full[reduce_ind]
#
#     # Retrieve blocks from fracture contribution
#     A_rm_f = A_full[reduce_ind][:, dof_frac]  # A_{rock, mortar; fracture}
#     b_corrected = b_rm - A_rm_f * force_frac
#
#     # Displacement in the rock and on the interface
#     u_rm = spsla.spsolve(A_rm, b_corrected)
#
#     # Distribute variables
#     u_rock = u_rm[rock_ind_reduced]
#     u_intf = u_rm[mortar_ind_reduced]
#
#     jump_u = (
#             intf.mortar_to_secondary_avg(nd=2)
#             * intf.sign_of_mortar_sides(nd=2)
#             * u_intf
#     )
#
#     jump_mpsa = np.abs(jump_u[1::2]) / setup.params["crack_length"]
#
# #jump_u = np.reshape(np.absolute(jump_u), (sd_frac.num_cells, 2))
# #jump_u = jump_u[:, 0] * np.cos(beta) + jump_u[:, 1] * np.sin(beta)
#
# # pp.plot_grid(sd_rock, u_rock[::2], plot_2d=True, title="u_x")
# # pp.plot_grid(sd_rock, u_rock[1::2], plot_2d=True, title="u_y")
#
# %%
"""Plot relative displacement as a function of distance from crack center"""

# Generate exact points
# TODO: Do we need to rotate the coordinates?
sd_frac = setup.mdg.subdomains()[1]
crack_length = setup.params["crack_length"]
x_min = np.min(sd_frac.nodes[0])
x_max = np.max(sd_frac.nodes[0])
y_min = np.min(sd_frac.nodes[1])
y_max = np.max(sd_frac.nodes[1])
x_ex = np.linspace(x_min, x_max, 100)
y_ex = np.linspace(y_min, y_max, 100)
z_ex = np.zeros(100)
points_ex = np.array([x_ex, y_ex, z_ex])
eta_ex = setup.distance_from_crack_center(points_ex)
u_jump_ex = setup.exact_normal_displacement_jump(eta_ex)

fig, ax = plt.subplots(figsize=(9, 8))

# Plot exact solution
ax.plot(
    x_ex / crack_length,
    u_jump_ex / crack_length,
    linewidth=3,
    color="blue",
    alpha=0.6,
)

# Plot BEM solution
bem_elements = 20
bem_centers = setup.bem.get_bem_centers(bem_elements)
bem_sol = setup.bem.bem_relative_normal_displacement(bem_elements)
ax.plot(
    bem_centers[0] / crack_length,
    bem_sol / crack_length,
    marker="o",
    markersize=6,
    linewidth=0,
    color="orange",
)

# Plot MPSA solution
ax.plot(
    sd_frac.cell_centers[0] / crack_length,
    jump_mpsa,
    marker="s",
    markersize=6,
    linewidth=0,
    color="red",
)

# plt.step(
#     bem_centers[0] / crack_length,
#     bem_sol / crack_length,
#     where="mid",
#     color="orange",
# )
# plt.step(
#     np.array([2.0, bem_centers[0][0] / crack_length]),
#     np.array([0, bem_sol[0] / crack_length]),
#     where="pre",
#     color="orange",
# )
# plt.step(
#     np.array([bem_centers[0][-1] / crack_length, 3.0]),
#     np.array([bem_sol[-1] / crack_length, 0]),
#     where="post",
#     color="orange",
#     alpha=0.7,
# )

# Label plot
plt.plot(
    [],
    [],
    linewidth=4,
    color="blue",
    alpha=0.6,
    label="Exact solution",
)

plt.plot(
    [],
    [],
    linewidth=4,
    color="red",
    marker="s",
    markersize=8,
    label=f"MPSA ({sd_frac.num_cells} fracture cells)",
)

plt.plot(
    [],
    [],
    linewidth=4,
    color="orange",
    marker="o",
    markersize=8,
    label=f"BEM ({bem_sol.size} fracture elements)",
)



# Set labels and legends
ax.set_xlabel(r"Non-dimensional horizontal distance, " r"$x~/~b$", fontsize=13)
ax.set_ylabel(
    r"Relative normal displacement, " r"$\hat{u}_n(x)~/~b$", fontsize=13
)
ax.legend(fontsize=13)

if not os.path.exists("out"):
    os.makedirs("out")
plt.savefig("out/" + "sneddon" + ".pdf", bbox_inches="tight")
plt.gcf().clear()

#%%
u_rock_x = u_rock[::2]
u_rock_y = u_rock[1::2]
# pp.plot_grid(sd_rock, u_rock_x, plot_2d=True, linewidth=0, title="u_x (MPSA)")
# pp.plot_grid(sd_rock, u_rock_y, plot_2d=True, linewidth=0, title="u_y (MPSA)")
#
#
#%% Plot BEM stress solution
# cc = sd_rock.cell_centers
# sigma = setup.bem.compute_stress(
#     points_in_global_coo=cc,
#     num_bem_segments=1000
# )

# pp.plot_grid(sd_rock, sigma[0][0], plot_2d=True, title="sigma_xx", linewidth=0)
# pp.plot_grid(sd_rock, sigma[1][1], plot_2d=True, title="sigma_yy", linewidth=0)
# pp.plot_grid(sd_rock, sigma[0][1], plot_2d=True, title="sigma_xy", linewidth=0)

#%% Convergence analysis

# Elastic force

# Displacement
cc = sd_rock.cell_centers
u_bem = setup.bem.compute_displacement(cc)
# pp.plot_grid(sd_rock, u_rock[::2], plot_2d=True, linewidth=0, title="u_x (MPSA)")
# pp.plot_grid(sd_rock, u_rock[1::2], plot_2d=True, linewidth=0, title="u_y (MPSA)")
# pp.plot_grid(sd_rock, u_bem[::2], plot_2d=True, linewidth=0, title="u_x (BEM)")
# pp.plot_grid(sd_rock, u_bem[1::2], plot_2d=True, linewidth=0, title="u_y (BEM)")

# Relative normal displacement



#%%
fc = sd_rock.face_centers
nf = sd_rock.face_normals

# BEM FORCE
sigma_bem = setup.bem.compute_stress(points_in_global_coo=fc, num_bem_segments=1000)
force_bem_x = sigma_bem[0][0] * nf[0] + sigma_bem[0][1] * nf[1]
force_bem_y = sigma_bem[1][0] * nf[0] + sigma_bem[1][1] * nf[1]
force_bem = np.concatenate((force_bem_x, force_bem_y)).ravel("F")

# MPSA FORCE
setup.reconstruct_stress()
force_mpsa = data_rock[pp.STATE]["stress"].copy()

error_force = relative_l2_error(
    grid=sd_rock,
    true_array=force_bem,
    approx_array=force_mpsa,
    is_scalar=False,
    is_cc=False
)

error_displacement = relative_l2_error(
    grid=sd_rock,
    true_array=u_bem,
    approx_array=u_rock,
    is_scalar=False,
    is_cc=True
)

print("Force error", error_force)
print("Displacement error", error_displacement)

#%%
mesh_size = np.array([2, 1, 0.5, 0.25])
errors = np.array([0.010883, 0.00449455, 0.001592, 0.0006610])

plt.figure()

rate = 1
x1 = np.log2(1 / 2)
x2 = np.log2(1 / 0.25)
y1 = -6
y2 = y1 - rate * (x2 - x1)
plt.plot([x1, x2], [y1, y2], "k-", linewidth=3, label="First order")

rate = 2
x1 = np.log2(1 / 2)
x2 = np.log2(1 / 0.25)
y1 = -7
y2 = y1 - rate * (x2 - x1)
plt.plot([x1, x2], [y1, y2], "k--", linewidth=3, label="Second order")

plt.plot(np.log2(1/mesh_size), np.log2(errors), "o-r", linewidth=3)
plt.xlabel("log2( 1/h )")
plt.ylabel("log2(error displacement)")
plt.grid()
plt.legend()
plt.show()
