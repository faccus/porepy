import porepy as pp
import numpy as np
from porepy.fracs.fracture_importer import dfm_from_gmsh
import gmsh

#%%
# Parameters for Sneddon mdg
domain_size = (50, 50)
crack_length = 20
crack_angle = np.deg2rad(135)
tip_radii = 5
mesh_arguments = {"mesh_size_frac": 2, "mesh_size_bound": 2}
mesh_size = 2

domain = pp.Domain(
    {"xmin": 0, "xmax": domain_size[0], "ymin": 0, "ymax": domain_size[1]}
)

a = domain.bounding_box["xmax"]
b = domain.bounding_box["ymax"]

# We shall assume that the fracture network is given...

# -----> Compute tips coordinates

# Tip 0
x0 = (a / 2) - (crack_length / 2) * np.cos(crack_angle)
y0 = (b / 2) - (crack_length / 2) * np.sin(crack_angle)

# Tip 1
x1 = (a / 2) + (crack_length / 2) * np.cos(crack_angle)
y1 = (b / 2) + (crack_length / 2) * np.sin(crack_angle)

# -----> Compute endpoints coordinates of the tip circles

# Circle 0, Endpoint 0
x0_e0 = x0 - tip_radii * np.cos(crack_angle)
y0_e0 = y0 - tip_radii * np.sin(crack_angle)
# Circle 0, Endpoint 1
x0_e1 = x0 + tip_radii * np.cos(crack_angle)
y0_e1 = y0 + tip_radii * np.sin(crack_angle)

# Circle 1, Endpoint0
x1_e0 = x1 - tip_radii * np.cos(crack_angle)
y1_e0 = y1 - tip_radii * np.sin(crack_angle)
# Circle 1, Endpoint1
x1_e1 = x1 + tip_radii * np.cos(crack_angle)
y1_e1 = y1 + tip_radii * np.sin(crack_angle)

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 3)
gmsh.model.add("sneddon_api")

"""
Definition of geometrical objects.

    - 0d: Fracture tips, arcs endpoints around tips, boundary points.
    - 1d: Fracture lines, arcs around tips, boundary lines, boundary loop.
    - 2d: Domain surface.

"""

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
# We need to embedd the three fracture lines and the four arcs
gmsh.model.mesh.embed(
    dim=1,
    tags=[ln_frac_1, ln_frac_2, ln_frac_3, ln_arc_1, ln_arc_2, ln_arc_3, ln_arc_4],
    inDim=2,
    inTag=1
)

# Mesh
gmsh.model.mesh.generate(dim=2)


# Finally, write the ouput file and finalize gmsh
gmsh.write("sneddon_api.msh")
gmsh.finalize()

#%%
mdg = dfm_from_gmsh("sneddon_api.msh", dim=2)



