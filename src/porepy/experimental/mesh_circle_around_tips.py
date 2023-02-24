"""
Module with a function that creates meshes with circular regions around tips.

The function assumes a single line fracture of length `2h` fully embedded in the
middle of a rectangular domain `(0, 0), (a, b)` forming an agle `theta` with the
horizontal axis. The circular region around the two tips of the fracture is
controlled by the circle radii `r`.

We require `r >= h`, where `h` is the target mesh size. If the radii is not given,
the mesh is created without the circular regions around tips, e.g., in the usual way.

"""
import porepy as pp
import numpy as np
from porepy.fracs.fracture_importer import dfm_from_gmsh
import gmsh
from typing import Optional


def create_mdg_with_circles_around_tips(
        domain_size: tuple[pp.number, pp.number],
        fracture_length: pp.number,
        fracture_angle: pp.number,
        mesh_size: pp.number,
        circle_radii: Optional[pp.number] = None,
) -> pp.MixedDimensionalGrid:
    """Create a mixed-dimensional grid with circular constraints around fracture tips.

    Parameters:
        domain_size: Size of the domain in scaled [m].
        fracture_length: Length of the fracture in scaled [m].
        fracture_angle: Angle of the fracture w.r.t. the horizontal axis in [rad].
        mesh_size: Target mesh size in scaled [m].
        circle_radii: Radii of the circular regions around the fracture tips in which
            the mesh should conform in scaled [m].
    Returns:
        Mixed-dimensional grid, with the constrained circular regions around the tips.

    Raises:
        - ``ValueError`` if ``circle_radii`` is less than ``mesh_size``.

    """
    a, b = domain_size
    fl = fracture_length
    theta = fracture_angle
    h = mesh_size
    if circle_radii is not None:
        r = circle_radii
        is_constrain = True
        # Sanity check
        msg = "Radii cannot be smaller than target mesh size."
        if r < h:
            raise ValueError(msg)
    else:
        is_constrain = False

    # Compute fracture tips
    # Tip 0
    x0 = (a / 2) - (fl / 2) * np.cos(theta)
    y0 = (b / 2) - (fl / 2) * np.sin(theta)
    # Tip 1
    x1 = (a / 2) + (fl / 2) * np.cos(theta)
    y1 = (b / 2) + (fl / 2) * np.sin(theta)
    # Points
    frac_pts = np.array([[x0, x1], [y0, y1]])

    if is_constrain:
        # Compute circle points
        # Circle of Tip 0, Endpoint 0
        x0_e0 = x0 - r * np.cos(theta)
        y0_e0 = y0 - r * np.sin(theta)
        # Circle of Tip 0, Endpoint 1
        x0_e1 = x0 + r * np.cos(theta)
        y0_e1 = y0 + r * np.sin(theta)
        # Circle of Tip 1, Endpoint 0
        x1_e0 = x1 - r * np.cos(theta)
        y1_e0 = y1 - r * np.sin(theta)
        # Circle of Tip 1, Endpoint 1
        x1_e1 = x1 + r * np.cos(theta)
        y1_e1 = y1 + r * np.sin(theta)
        # Points
        arc_pts = np.array(
            [[x0_e0, x0_e1, x1_e0, x1_e1], [y0_e0, y0_e1, y1_e0, y1_e1]]
        )

    if not is_constrain:
        def _create_mdg(frac_pts, domain_size, mesh_size):
            ...
    else:
        def _create_constrained_mdg(frac_pts, arc_pts, domain_size, mesh_size):
            ...


def _create_mdg(
        frac_pts: np.ndarray,
        domain_size: tuple[pp.number, pp.number],
        mesh_size: pp.number,
) -> pp.MixedDimensionalGrid:
    """Create mixed-dimensional grid for without constrains.

    Parameters:
        frac_pts: Coordinates of the fracture tips. Shape is (2, 2).
        domain_size: Size of the domain.
        mesh_size: Target mesh size.

    Retunrs:
        Mixed-dimensional grid.

    """
    line = pp.LineFracture(points=frac_pts)
    fractures = [line]
    a, b = domain_size
    domain = pp.Domain({"xmin": 0, "xmax": a, "ymin": 0, "ymax": b})