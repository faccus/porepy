import porepy as pp
import numpy as np
from porepy.fracs.gmsh_interface import GmshWriter
import gmsh


# Parameters for Sneddon mdg
domain_size = (50, 50)
crack_length = 20
crack_angle = 0
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

# Let's try to write in the API
if not GmshWriter.gmsh_initialized:
    gmsh.initialize()
    GmshWriter.gmsh_initialized = True

gmsh.option.setNumber("General.Verbosity", 3)

# ----> 0d-objects

# Boundary points
gmsh.model.geo.addPoint(0, 0, 0, meshSize=mesh_size, tag=1)
gmsh.model.geo.addPoint(a, 0, 0, meshSize=mesh_size, tag=2)
gmsh.model.geo.addPoint(a, b, 0, meshSize=mesh_size, tag=3)
gmsh.model.geo.addPoint(0, b, 0, meshSize=mesh_size, tag=4)
gmsh.model.geo.synchronize()

pg_point_1 = gmsh.model.addPhysicalGroup(0, [1])
pg_point_2 = gmsh.model.addPhysicalGroup(0, [2])
pg_point_3 = gmsh.model.addPhysicalGroup(0, [3])
pg_point_4 = gmsh.model.addPhysicalGroup(0, [4])

gmsh.model.setPhysicalName(0, pg_point_1, "DOMAIN_BOUNDARY_POINT_1")
gmsh.model.setPhysicalName(0, pg_point_2, "DOMAIN_BOUNDARY_POINT_2")
gmsh.model.setPhysicalName(0, pg_point_3, "DOMAIN_BOUNDARY_POINT_3")
gmsh.model.setPhysicalName(0, pg_point_4, "DOMAIN_BOUNDARY_POINT_4")

# Fracture tips
gmsh.model.geo.addPoint(x0, y0, 0, meshSize=mesh_size, tag=5)
gmsh.model.geo.addPoint(x1, y1, 0, meshSize=mesh_size, tag=6)
gmsh.model.geo.synchronize()

# Circle endpoints
gmsh.model.geo.addPoint(x0_e0, y0_e0, 0, meshSize=mesh_size, tag=7)
gmsh.model.geo.addPoint(x0_e1, y0_e1, 0, meshSize=mesh_size, tag=8)
gmsh.model.geo.addPoint(x1_e0, y1_e0, 0, meshSize=mesh_size, tag=9)
gmsh.model.geo.addPoint(x1_e1, y1_e1, 0, meshSize=mesh_size, tag=10)

# -----> 1d-objects

# Boundary lines
gmsh.model.geo.addLine(1, 2, tag=1)
gmsh.model.geo.addLine(2, 3, tag=2)
gmsh.model.geo.addLine(3, 4, tag=3)
gmsh.model.geo.addLine(4, 1, tag=4)
gmsh.model.geo.synchronize()

pg_line_1 = gmsh.model.addPhysicalGroup(1, [1])
pg_line_2 = gmsh.model.addPhysicalGroup(1, [2])
pg_line_3 = gmsh.model.addPhysicalGroup(1, [3])
pg_line_4 = gmsh.model.addPhysicalGroup(1, [4])

gmsh.model.setPhysicalName(1, pg_line_1, "DOMAIN_BOUNDARY_LINE_1")
gmsh.model.setPhysicalName(1, pg_line_2, "DOMAIN_BOUNDARY_LINE_2")
gmsh.model.setPhysicalName(1, pg_line_3, "DOMAIN_BOUNDARY_LINE_3")
gmsh.model.setPhysicalName(1, pg_line_4, "DOMAIN_BOUNDARY_LINE_4")

# Fracture lines
gmsh.model.geo.addLine(5, 8, tag=5)
gmsh.model.geo.addLine(8, 9, tag=6)
gmsh.model.geo.addLine(9, 6, tag=7)
gmsh.model.geo.synchronize()

pg_line_5 = gmsh.model.addPhysicalGroup(1, [5, 6, 7])
gmsh.model.setPhysicalName(1, pg_line_5, "FRACTURE_1")

# Arcs
gmsh.model.geo.addCircleArc(7, 5, 8, tag=8)
gmsh.model.geo.addCircleArc(8, 5, 7, tag=9)
gmsh.model.geo.addCircleArc(9, 6, 10, tag=10)
gmsh.model.geo.addCircleArc(10, 6, 9, tag=11)
gmsh.model.geo.synchronize()

pg_line_6 = gmsh.model.addPhysicalGroup(1, [8, 9])
pg_line_7 = gmsh.model.addPhysicalGroup(1, [10, 11])

gmsh.model.setPhysicalName(1, pg_line_6, "AUXILIARY_LINE_1")
gmsh.model.setPhysicalName(1, pg_line_7, "AUXILIARY_LINE_2")

# ----> Set loops, surfaces, and generate geo file

# Set loop
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.synchronize()

# Define surface
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()

# Embed lower-dimensional objects
gmsh.model.mesh.embed(1, [5, 6, 7, 8, 9, 10, 11], 2, 1)

gmsh.write("sneddon_api.geo_unrolled")
gmsh.clear()
gmsh.finalize()





