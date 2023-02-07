import porepy as pp
import numpy as np

# Create strip load half space

domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
network_2d = pp.fracture_importer.network_2d_from_csv("halfspace.csv", domain=domain)
mesh_args = {
    "mesh_size_bound": 0.02,
    "mesh_size_frac": 0.02
}
mdg = network_2d.mesh(mesh_args, constraints=[0, 1, 2, 3, 4])
sd = mdg.subdomains()[0]
#pp.plot_grid(sd, plot_2d=True)
exporter = pp.Exporter(sd, file_name="half_space_grid")
exporter.write_vtu()