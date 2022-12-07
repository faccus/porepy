"""
This module contains a runner for Mandel's consolidation problem using default
parameters.

For an overview of the admissible parameters, please refer to the documentation of the
``Mandel`` class available in ``porepy.models.applications.mandel.mandel_model``.

"""

# Import modules
import porepy as pp
from time import time
from porepy.models.applications.mandel_poroelasticity.mandel_model import Mandel

# Runner
tic = time()
params = {"plot_results": True}
setup = Mandel(params)
print("Simulation started...")
pp.run_time_dependent_model(setup, setup.params)
toc = time()
print(f"Simulation finished in {round(toc - tic)} seconds.")
