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
        file_name: str = "gmsh_frac_file",
        export_geo: bool = True,
) -> pp.MixedDimensionalGrid:
    """Create a mixed-dimensional grid with circular constraints around fracture tips.

    Parameters:
        domain_size: Size of the domain in scaled [m].
        fracture_length: Length of the fracture in scaled [m].
        fracture_angle: Angle of the fracture w.r.t. the horizontal axis in [rad].
        mesh_size: Target mesh size in scaled [m].
        circle_radii: Radii of the circular regions around the fracture tips in which
            the mesh should conform in scaled [m].
        file_name: Name of the file to be exported. Default is ``gmsh_frac_file``.
        export_geo: Whether to export in ``geo_unrolled`` format. Default is True.

    Returns:
        Mixed-dimensional grid, with the constrained circular regions around the tips.

    Raises:
        - ``ValueError`` if ``circle_radii`` is less than ``mesh_size``.

    Note:
        We assume that the radii is sufficiently small such that the circular region
        around a tip does not intersect the other circle, or a boundary line.

    """
    if circle_radii is not None:
        is_constrained = True
        # Sanity check
        if circle_radii < mesh_size:
            raise ValueError("Radii cannot be smaller than target mesh size.")
    else:
        is_constrained = False

    # Compute fracture tips
    # Tip 0
    x0 = (domain_size[0] / 2) - (fracture_length / 2) * np.cos(fracture_angle)
    y0 = (domain_size[1] / 2) - (fracture_length / 2) * np.sin(fracture_angle)
    # Tip 1
    x1 = (domain_size[0] / 2) + (fracture_length / 2) * np.cos(fracture_angle)
    y1 = (domain_size[1] / 2) + (fracture_length / 2) * np.sin(fracture_angle)
    # Points
    frac_pts = np.array([[x0, x1], [y0, y1]])

    if not is_constrained:
        # Without constrains. Create mdg in the usual way.
        mdg = _create_mdg(
            frac_pts=frac_pts,
            domain_size=domain_size,
            mesh_size=mesh_size,
            file_name=file_name,
            export_geo=export_geo,
        )
    else:
        # With constraints. First, compute the circle endpoints
        # Circle of Tip 0, Endpoint 0
        x0_e0 = x0 - circle_radii * np.cos(fracture_angle)
        y0_e0 = y0 - circle_radii * np.sin(fracture_angle)
        # Circle of Tip 0, Endpoint 1
        x0_e1 = x0 + circle_radii * np.cos(fracture_angle)
        y0_e1 = y0 + circle_radii * np.sin(fracture_angle)
        # Circle of Tip 1, Endpoint 0
        x1_e0 = x1 - circle_radii * np.cos(fracture_angle)
        y1_e0 = y1 - circle_radii * np.sin(fracture_angle)
        # Circle of Tip 1, Endpoint 1
        x1_e1 = x1 + circle_radii * np.cos(fracture_angle)
        y1_e1 = y1 + circle_radii * np.sin(fracture_angle)
        # Points
        arc_pts = np.array(
            [[x0_e0, x0_e1, x1_e0, x1_e1], [y0_e0, y0_e1, y1_e0, y1_e1]]
        )
        mdg = _create_constrained_mdg(
            frac_pts=frac_pts,
            arc_pts=arc_pts,
            domain_size=domain_size,
            mesh_size=mesh_size,
            file_name=file_name,
            export_geo=export_geo,
        )

    return mdg


def _create_mdg(
        frac_pts: np.ndarray,
        domain_size: tuple[pp.number, pp.number],
        mesh_size: pp.number,
        file_name: str,
        export_geo: bool,
) -> pp.MixedDimensionalGrid:
    """Create mixed-dimensional grid for with constrains.

    Parameters:
        frac_pts: Coordinates of the fracture tips. Shape is (2, 2).
        domain_size: Size of the domain.
        mesh_size: Target mesh size.
        file_name: Name of the file.
        export_geo: Whether to export the geo file.

    Returns:
        Mixed-dimensional grid.

    """
    # Create list of fractures
    fractures: list[pp.LineFracture] = [pp.LineFracture(points=frac_pts)]
    # Retrieve length and width of the domain
    a, b = domain_size
    # Construct domain object
    domain = pp.Domain({"xmin": 0, "xmax": a, "ymin": 0, "ymax": b})
    # Construct fracture network
    fn = pp.create_fracture_network(fractures, domain)
    # Mesh arguments
    mesh_args = {"mesh_size_frac": mesh_size, "mesh_size_bound": mesh_size}
    # Construct mdg
    mdg = fn.mesh(mesh_args=mesh_args, file_name=file_name, write_geo=export_geo)
    return mdg


def _create_constrained_mdg(
        frac_pts: np.ndarray,
        arc_pts: np.ndarray,
        domain_size: tuple[pp.number, pp.number],
        mesh_size: pp.number,
        file_name: str,
        export_geo: bool,
) -> pp.MixedDimensionalGrid:
    """Create mixed-dimensional grid with curved constrains.

    Parameters:
        frac_pts: Coordinates of the fracture tips. Shape is (2, 2).
        arc_pts: Coordinates of the endpoints of the arcs that will form the
            constrained circular regions around the tips. Shape is (2, 4). The first
            two columns correspond to the coordinates of the first fracture tip,
            and the last two columns to the coordinates fo the second fracture tip.
        domain_size: Size of the domain.
        mesh_size: Target mesh size.
        file_name: Name of the file.
        export_geo: Whether to export the geo file.

    Returns:
        Mixed-dimensional grid.

    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.model.add(file_name)

    """
    Definition of geometrical objects.

        - 0d: Fracture tips, arcs endpoints around tips, boundary points.
        - 1d: Fracture lines, arcs around tips, boundary lines, boundary loop.
        - 2d: Domain surface.

    """
    # Extract fracture points
    x0 = frac_pts[0][0]
    x1 = frac_pts[0][1]
    y0 = frac_pts[1][0]
    y1 = frac_pts[1][1]

    # Extract arc points
    x0_e0 = arc_pts[0][0]
    x0_e1 = arc_pts[0][1]
    x1_e0 = arc_pts[0][2]
    x1_e1 = arc_pts[0][3]
    y0_e0 = arc_pts[1][0]
    y0_e1 = arc_pts[1][1]
    y1_e0 = arc_pts[1][2]
    y1_e1 = arc_pts[1][3]

    # Unpack domain_size
    a, b = domain_size

    # ----> 0-dimensional objects

    # Fracture tips
    pt_tip_1 = gmsh.model.geo.addPoint(x0, y0, 0, meshSize=mesh_size)
    pt_tip_2 = gmsh.model.geo.addPoint(x1, y1, 0, meshSize=mesh_size)

    # Arcs endpoints
    pt_arc_1 = gmsh.model.geo.addPoint(x0_e0, y0_e0, 0, meshSize=mesh_size)
    pt_arc_2 = gmsh.model.geo.addPoint(x0_e1, y0_e1, 0, meshSize=mesh_size)
    pt_arc_3 = gmsh.model.geo.addPoint(x1_e0, y1_e0, 0, meshSize=mesh_size)
    pt_arc_4 = gmsh.model.geo.addPoint(x1_e1, y1_e1, 0, meshSize=mesh_size)

    # Boundary points
    pt_sw = gmsh.model.geo.addPoint(0, 0, 0, meshSize=mesh_size)
    pt_se = gmsh.model.geo.addPoint(a, 0, 0, meshSize=mesh_size)
    pt_nw = gmsh.model.geo.addPoint(a, b, 0, meshSize=mesh_size)
    pt_ne = gmsh.model.geo.addPoint(0, b, 0, meshSize=mesh_size)

    gmsh.model.geo.synchronize()

    # ----> 1-dimensional objects

    # Fracture lines
    ln_frac_1 = gmsh.model.geo.addLine(pt_tip_1, pt_arc_2)
    ln_frac_2 = gmsh.model.geo.addLine(pt_arc_2, pt_arc_3)
    ln_frac_3 = gmsh.model.geo.addLine(pt_arc_3, pt_tip_2)

    # Arcs around tips
    ln_arc_1 = gmsh.model.geo.addCircleArc(pt_arc_1, pt_tip_1, pt_arc_2)
    ln_arc_2 = gmsh.model.geo.addCircleArc(pt_arc_2, pt_tip_1, pt_arc_1)
    ln_arc_3 = gmsh.model.geo.addCircleArc(pt_arc_3, pt_tip_2, pt_arc_4)
    ln_arc_4 = gmsh.model.geo.addCircleArc(pt_arc_4, pt_tip_2, pt_arc_3)

    # Boundary lines
    ln_w = gmsh.model.geo.addLine(pt_sw, pt_se)
    ln_s = gmsh.model.geo.addLine(pt_se, pt_nw)
    ln_e = gmsh.model.geo.addLine(pt_nw, pt_ne)
    ln_n = gmsh.model.geo.addLine(pt_sw, pt_ne)

    # Boundary loop
    bound_loop = gmsh.model.geo.addCurveLoop([ln_w, ln_s, ln_e, -ln_n], reorient=True)

    gmsh.model.geo.synchronize()

    # -----> 2-dimensional objects

    # Domain surface
    surface = gmsh.model.geo.addPlaneSurface([bound_loop])

    gmsh.model.geo.synchronize()

    """
    Definition of physical objects.

        - Domain boundary points.
        - Domain boundary lines.
        - Fracture (line).
        - Auxiliary lines (arcs).
        - Domain.

    """

    # Boundary points
    for idx, tag in enumerate([pt_sw, pt_se, pt_nw, pt_ne]):
        pg = gmsh.model.addPhysicalGroup(dim=0, tags=[tag])
        gmsh.model.setPhysicalName(dim=0, tag=pg, name=f"DOMAIN_BOUNDARY_POINT_{idx}")

    # Boundary lines
    for idx, tag in enumerate([ln_w, ln_s, ln_e, ln_n]):
        pg = gmsh.model.addPhysicalGroup(dim=1, tags=[tag])
        gmsh.model.setPhysicalName(dim=1, tag=pg, name=f"DOMAIN_BOUNDARY_LINE_{idx}")

    # Fracture
    pg = gmsh.model.addPhysicalGroup(dim=1, tags=[ln_frac_1, ln_frac_2, ln_frac_3])
    gmsh.model.setPhysicalName(dim=1, tag=pg, name="FRACTURE_0")

    # Auxiliary lines
    for idx, tag in enumerate([[ln_arc_1, ln_arc_2], [ln_arc_3, ln_arc_4]]):
        pg = gmsh.model.addPhysicalGroup(dim=1, tags=tag)
        gmsh.model.setPhysicalName(dim=1, tag=pg, name=f"AUXILIARY_LINE_{idx}")

    # Domain
    pg = gmsh.model.addPhysicalGroup(dim=2, tags=[surface])
    gmsh.model.setPhysicalName(dim=2, tag=pg, name="DOMAIN")

    """
    Mesh-related actions:

        - Embed fracture and arcs into the surface.
        - Create mesh.

    """

    # Embed lower-dimensional objects
    # We need to embedd three fracture lines and four circle arcs
    gmsh.model.mesh.embed(
        dim=1,
        tags=[ln_frac_1, ln_frac_2, ln_frac_3, ln_arc_1, ln_arc_2, ln_arc_3, ln_arc_4],
        inDim=2,
        inTag=1
    )

    # Mesh
    gmsh.model.mesh.generate(dim=2)

    # Write ouput file(s) and finalize gmsh
    gmsh.write(file_name + ".msh")
    if export_geo:
        gmsh.write(file_name + ".geo_unrolled")
    gmsh.finalize()

    # Finally, create mdg
    mdg = dfm_from_gmsh(file_name=file_name + ".msh", dim=2)

    return mdg
