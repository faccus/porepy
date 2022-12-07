"""
This module contains a runner for Terzaghi's consolidation problem using default
parameters.

For an overview of the admissible parameters, please refer to the documentation of the
``Terzaghi`` class available in ``porepy.models.applications.terzaghi.terzaghi_model``.

"""

# Import modules
import porepy as pp
from time import time
from porepy.models.applications.terzaghi.terzaghi_model import Terzaghi

# Runner
tic = time()
params = {"plot_results": True}
setup = Terzaghi(params)
print("Simulation started...")
pp.run_time_dependent_model(setup, setup.params)
toc = time()
print(f"Simulation finished in {round(toc - tic)} seconds.")
