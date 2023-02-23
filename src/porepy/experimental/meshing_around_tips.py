import porepy as pp
import numpy as np
import gmsh

from porepy.fracs.fracture_importer import dfm_from_gmsh
from porepy.fracs.simplex import _read_gmsh_file

def create_2d_grid_from_geo(
        file_name: str,
):

    file_name = file_name[:-4]
    in_file = file_name + ".geo"
    out_file = file_name + ".msh"

    # initialize gmsh
    gmsh.initialize()
    # Reduce verbosity
    gmsh.option.setNumber("General.Verbosity", 3)
    # read the specified file.
    gmsh.merge(in_file)
    # Generate mesh, write
    gmsh.model.mesh.generate(dim=2)
    gmsh.write(out_file)
    # Wipe Gmsh's memory
    gmsh.finalize()

    pts, cells, cell_info, phys_names = _read_gmsh_file(out_file)
    triangles = cells["triangle"].transpose()
    g_2d: pp.Grid = pp.TriangleGrid(pts.transpose(), triangles)

    faces = np.reshape(g_2d.face_nodes.indices, (2, -1), order="F")
    faces = np.sort(faces, axis=0)



    return g_2d


#%% Parameters
crack_length = 20  # [m]
crack_angle = 0  # [rad]
domain_size = (50, 50)  # [m]
mesh_size = 5  # [m]


# Set domain
dom = pp.Domain({"xmin": 0, "xmax": domain_size[0], "ymin": 0, "ymax": domain_size[1]})

# Create fracture network
x0 = (domain_size[0] / 2) - (crack_length / 2) * np.cos(crack_angle)
x1 = (domain_size[0] / 2) + (crack_length / 2) * np.cos(crack_angle)
y0 = (domain_size[1] / 2) - (crack_length / 2) * np.sin(crack_angle)
y1 = (domain_size[1] / 2) + (crack_length / 2) * np.sin(crack_angle)
pts = np.array([[x0, x1], [y0, y1]])
edges = np.array([[0], [1]])
fn = pp.FractureNetwork2d(pts, edges, dom)

# Create mixed-dimensional grid
#mesh_arguments = {"mesh_size_frac": mesh_size, "mesh_size_bound": mesh_size}
#mdg = fn.mesh(mesh_arguments, file_name="sneddon_horizontal")

#%% Read and mesh from geo file
# mdg_with_circle = dfm_from_gmsh("sneddon_horizontal_circle.geo", dim=2)
# sd = mdg_with_circle.subdomains()[0]
#pp.plot_grid(sd, alpha=0, plot_2d=True)

mdg = fn.mesh({"mesh_size_frac": 2, "mesh_size_bound": 2})

#sd = create_2d_grid_from_geo("sneddon_horizontal_circle.geo")

