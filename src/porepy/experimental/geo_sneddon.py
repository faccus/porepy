import porepy as pp
import numpy as np
import gmsh

# Parameters from sneddon
domain_size = (50, 50)
crack_length = 20
crack_angle = 0
tip_radii = 5

domain = pp.Domain(
    {"xmin": 0, "xmax": domain_size[0], "ymin": 0, "ymax": domain_size[1]}
)

a = domain.bounding_box["xmax"]
b = domain.bounding_box["ymax"]

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



