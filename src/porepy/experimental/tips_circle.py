import porepy as pp
import numpy as np

from porepy.grids.standard_grids.utils import unit_domain
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_importer import dfm_from_gmsh


#%% Create mesh
domain = unit_domain(2)
pts_coo = np.array([[0.25, 0.75], [0.5, 0.5]])
pts_idx = np.array([[0], [1]])
fn = FractureNetwork2d(pts_coo, pts_idx, domain)
mesh_args = {"mesh_size_frac": 0.1, "mesh_size_bound": 0.1}
fn.mesh(mesh_args=mesh_args, file_name="single_frac")
mdg = dfm_from_gmsh(file_name="single_frac.geo", dim=2)

#%% Create mesh with constraint
domain = unit_domain(2)
pts_coo = np.array([[0.25, 0.75, 0.5, 0.5], [0.5, 0.5, 0.25, 0.75]])
pts_idx = np.array([[0, 2], [1, 3]])
fn = FractureNetwork2d(pts_coo, pts_idx, domain)
mesh_args = {"mesh_size_frac": 0.1, "mesh_size_bound": 0.1}
mdg1 = fn.mesh(
    mesh_args=mesh_args,
    file_name="single_frac_constrain",
    constraints=np.array([1])
)

#%% Read from geo
