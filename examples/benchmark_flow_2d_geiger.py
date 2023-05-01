"""Module containing the setup for the Geiger 2d benchmark for flow."""
import warnings

import porepy as pp
import numpy as np

import porepy.models.constitutive_laws
from porepy.applications.md_grids.model_geometries import BenchmarkFlow2dGeigerGeometry
from porepy.models.fluid_mass_balance import(
    BoundaryConditionsSinglePhaseFlow,
    SinglePhaseFlow,
)


# -----> Boundary conditions
class BenchmarkFlow2dGeigerBoundaryConditions(BoundaryConditionsSinglePhaseFlow):
    """Mixin class specifying the boundary conditions."""

    def bc_type_darcy(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """todo: add docstring.

        Parameters:
            sd: Subdomain grid on which to define boundary conditions.

        Returns:
            Boundary condition object.

        """
        sides = self.domain_boundary_sides(sd)
        if sides.all_bf.size != 0:
            return pp.BoundaryCondition(sd, sides.east, "dir")
        else:
            return pp.BoundaryCondition(sd)

    def bc_values_darcy(self, subdomains: list[pp.Grid]) -> pp.ad.DenseArray:
        """todo: add docstring

        """

        values: list[np.ndarray] = []
        for sd in subdomains:
            sides = self.domain_boundary_sides(sd)
            bound_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            val_loc = np.zeros(sd.num_faces)
            if bound_faces.size != 0:
                a_dim = np.power(
                    self.solid.residual_aperture(),
                    self.mdg.dim_max() - sd.dim
                )
                val_loc[sides.west] = -a_dim * sd.face_areas[sides.west]
                val_loc[sides.east] = 1.0
            values.append(val_loc)

        return pp.wrap_as_ad_array(np.hstack(values), name="bc_values_darcy")


# -----> Solution strategy
class BenchmarkFlow2dGeigerSolutionStrategy(
    porepy.models.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
    """Solution strategy for the benchmark study."""

    def __init__(self, params: dict):
        super().__init__(params=params)
        self.set_benchmark_parameters()

    def set_benchmark_parameters(self):
        """Set material constants for the benchmark."""
        is_conductive = self.params.get("is_conductive", True)
        if is_conductive:
            solid = pp.SolidConstants(
                {
                    "permeability": 1e4,
                    "normal_permeability": 1e4,
                    "residual_aperture": 1e-4,
                }
            )
        else:
            solid = pp.SolidConstants(
                {
                    "permeability": 1e-4,
                    "normal_permeability": 1e-4,
                    "residual_aperture": 1e-4,
                }
            )
        material_constants = {"solid": solid, "fluid": pp.FluidConstants()}
        if "material_constants" in self.params.keys():
            warnings.warn("Material constants' dictionary will be overridden.")
        else:
            self.params["material_constants"] = material_constants

    def permeability_tensor(self, sd: pp.Grid) -> pp.SecondOrderTensor:
        """Set the permeability tensor."""

        # Effective permeability, scaled with aperture.
        ones = np.ones(sd.num_cells)

        a = self.solid.residual_aperture()
        kf = self.solid.permeability()
        a_dim = np.power(a, self.mdg.dim_max() - sd.dim)
        aperture = ones * a_dim

        kxx = ones * np.power(kf, sd.dim < self. mdg.dim_max()) * aperture
        if sd.dim == 2:
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=ones)
        else:
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=ones, kzz=ones)

        return perm


# -----> Mixer class
class BenchmarkFlow2dGeigerSetup(
    BenchmarkFlow2dGeigerGeometry,
    BenchmarkFlow2dGeigerBoundaryConditions,
    BenchmarkFlow2dGeigerSolutionStrategy,
    SinglePhaseFlow,
):
    """Mixer class for the 2d flow benchmark 4.1 from [1].

    References:

        -[1] Flemisch, Bernd, et al. "Benchmarks for single-phase flow in fractured
        porous media." Advances in Water Resources 111 (2018): 239-258.

    Model parameters of special relevance for this example are:

        - grid_type ``(Literal["simplex", "cartesian", "tensor_grid"])``: Type of
          grid that should be used for running the simulation. Note that if
          ``cartesian`` or ``tensor_grid`` is selected, the mesh size should be selected
          accordingly so that a grid can be generated.
        - meshing_arguments ``(dict[str, float])``: Dictionary containing the meshing
          arguments.
        - is_conductive: Whether the benchmark should be run for the conductive
          version (4.1.1) or the blocking version (4.1.2).

    """
