"""Module containing a class for performing spatio-temporal convergence analysis."""
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Literal, Optional, Type, Union

import numpy as np
from scipy import stats

import porepy as pp


class ConvergenceAnalysis:
    """Perform spatio-temporal convergence analysis on a given model.

    The class :class:`~ConvergenceAnalysis` takes a PorePy model and its parameter
    dictionary to run a batch of simulations with successively refined mesh sizes and
    time steps and collect the results (i.e., the data classes containing the errors) in
    a list using the :meth:`~run_analysis` method. Levels of refinement (in time and
    space) are given at instantiation.

    Useful methods of this class include

        - :meth:`~export_errors_to_txt`: Exports errors into a ``txt`` file.
        - :meth:`~order_of_convergence`: Estimates the observed order of convergence
          of a given set of variables using linear regression.

    Note:
        Current support of the class includes

            - Simplicial and Cartesian grids in 2d and 3d. Tensor grids are not
              supported.

            - Static and time-dependent models.

            - If the model is time-dependent, we assume that the ``TimeManager`` is
              instantiated with a constant time step.

    Raises:

        ValueError
            If ``in_space`` and ``in_time`` are both ``False``.

        ValueError
            If ``model_class`` is stationary and ``in_time`` is ``True``.

        NotImplementedError
            If the time manager contains a non-constant time step.

        Warning
            If ``in_space=True`` and ``spatial_refinement_rate > 1``.

        Warning
            If ``in_time=True`` and ``temporal_refinement_rate > 1``.

    Parameters:
        model_class: Model class for which the analysis will be performed.
        model_params: Model parameters. We assume that it contains all parameters
            necessary to set up a valid instance of :data:`model_class`.
        levels: ``default=1``

            Number of refinement levels associated to the convergence analysis.
        in_space: ``default=True``

            Whether a convergence analysis in space should be performed.
        in_time:  ``default=False``

            Whether a convergence analysis in time should be performed.
        spatial_refinement_rate: ``default=1``

            Rate at which the mesh size(s) should be refined. For example, use
            ``spatial_refinement_rate=2`` for halving the mesh size(s) in-between
             levels.
        temporal_refinement_rate: ``default=1``

            Rate at which the time step size should be refined. For example, use
             ``temporal_refinement_rate=2`` for halving the time step size in-between
            levels.

    """

    def __init__(
        self,
        model_class,
        model_params: dict,
        levels: int = 1,
        in_space: bool = True,
        in_time: bool = False,
        spatial_refinement_rate: int = 1,
        temporal_refinement_rate: int = 1,
    ):
        # Sanity checks
        if not in_space and not in_time:
            raise ValueError("At least one type of analysis should be performed.")

        if not in_space and spatial_refinement_rate > 1:
            warnings.warn("'spatial_refinement_rate' is not being used.")
            spatial_refinement_rate = 1

        if not in_time and temporal_refinement_rate > 1:
            warnings.warn("'temporal_refinement_rate' is not being used.")
            temporal_refinement_rate = 1

        self.model_class = model_class
        """Model class that should be used to run the simulations and perform the
        convergence analysis.

        """

        self.levels: int = levels
        """Number of levels of the convergence analysis."""

        self.in_space: bool = in_space
        """Whether a spatial analysis should be performed."""

        self.in_time: bool = in_time
        """Whether a temporal analysis should be performed."""

        self.spatial_refinement_rate: int = spatial_refinement_rate
        """Rate at which the mesh size should be refined. A value of ``2``
        corresponds to halving the mesh size(s) between :attr:`levels`.

        """

        self.temporal_refinement_rate: int = temporal_refinement_rate
        """Rate at which the time step size should be refined. A value of ``2``
        corresponds to halving the time step size between :attr:`levels`.

        """

        # Initialize setup and retrieve spatial and temporal data
        setup = model_class(deepcopy(model_params))  # make a deep copy of dictionary
        setup.prepare_simulation()

        # Store initial setup
        self._init_setup = setup
        """Initial setup containing the 'base-line' information."""

        # We need to know whether the model is time-dependent or not
        self._is_time_dependent: bool = setup._is_time_dependent()
        """Whether the model is time-dependent."""

        if not self._is_time_dependent and self.in_time:
            raise ValueError("Analysis in time not available for stationary models.")

        # Retrieve list of meshing arguments
        # The list is of length ``levels`` and contains the ``meshing_arguments``
        # dictionaries needed to run the simulations.
        list_of_meshing_arguments: list[
            dict[str, float]
        ] = self._get_list_of_meshing_arguments()

        # Retrieve list of time managers
        # The list is of length ``levels`` and contains the ``pp.TimeManager``s
        # needed to run the simulations. ``None`` if the model is stationary.
        list_of_time_managers: Union[
            list[pp.TimeManager], None
        ] = self._get_list_of_time_managers()

        # Generate list of model parameters
        # Having the initial model parameter, the list of meshing arguments, and the
        # list of time managers, we can create the ready-to-be-fed list of model
        # parameters necessary for running the simulations
        list_of_params: list[dict] = []
        for lvl in range(self.levels):
            params = deepcopy(model_params)
            params["meshing_arguments"] = list_of_meshing_arguments[lvl]
            if list_of_time_managers is not None:
                params["time_manager"] = list_of_time_managers[lvl]
            list_of_params.append(params)

        self.model_params: list[dict] = list_of_params
        """List of model parameters associated to each simulation run."""

    def run_analysis(self) -> list:
        """Run convergence analysis.

        Returns:
            List of results (i.e., data classes containing the errors) for each
            refinement level. Note that for time-dependent models, only the result
            corresponding to the final time is appended to the list.

        """
        convergence_results: list = []
        for level in range(self.levels):
            setup = self.model_class(deepcopy(self.model_params[level]))
            if not setup._is_time_dependent():
                # Run stationary model
                pp.run_stationary_model(setup, deepcopy(self.model_params[level]))
                # Complement information in results
                setattr(setup.results[-1], "num_dofs", setup.equation_system.num_dofs())
                setattr(setup.results[-1], "cell_diameter", setup.mdg.diameter())
            else:
                # Run time-dependent model
                pp.run_time_dependent_model(setup, deepcopy(self.model_params[level]))
                # Complement information in results
                setattr(setup.results[-1], "num_dofs", setup.equation_system.num_dofs())
                setattr(setup.results[-1], "cell_diameter", setup.mdg.diameter())
                setattr(setup.results[-1], "dt", setup.time_manager.dt)

            convergence_results.append(setup.results[-1])
        return convergence_results

    def export_errors_to_txt(
        self,
        list_of_results: list,
        variables_to_export: Optional[list[str]] = None,
        file_name="error_analysis.txt",
    ) -> None:
        """Write errors into a ``txt`` file.

        The format is the following one:

            - First column contains the cell diameters.
            - Second column contains the time steps (if the model is time-dependent).
            - The rest of the columns contain the errors for each variable in
              ``variables``.

        Parameters:
            list_of_results: List containing the results of the convergence analysis.
                Typically, the output of :meth:`~run_analysis`.
            variables_to_export: names of the variables for which the TXT file will be
                generated. If ``variables`` is not given, all the variables present
                in the txt file will be collected.
            file_name: Name of the output file. Default is "error_analysis.txt".

        """
        cell_diameters = np.array([result.cell_diam for result in list_of_results])
        if self._is_time_dependent:
            time_steps = np.array([result.dt for result in list_of_results])
            non_vars = 2  # number of variables to be exported that are not errors
        else:
            time_steps = None
            non_vars = 1

        # Get variable names
        if variables_to_export is None:
            # Retrieve all attributes from the data class
            attributes: list[str] = list(vars(list_of_results[0]).keys())
            # Filter all attributes with the prefix ``error_``
            var_names: list[str] = [
                attr for attr in attributes if attr.startswith("error_")
            ]
        else:
            var_names = [attr for attr in variables_to_export]

        # Obtain errors
        var_errors: list[np.ndarray] = []
        for name in var_names:
            var_error: list[float] = []
            # Loop over lists of results
            for result in list_of_results:
                var_error.append(getattr(result, name))
            # Append to the ``var_errors`` list
            var_errors.append(np.array(var_error))

        # Initialize export table
        data_type: list[tuple[str, Type[float]]] = []
        for idx in range(non_vars + len(var_names)):
            data_type.append((f"var{idx}", float))
        export = np.zeros(self.levels, dtype=data_type)

        # Fill out the table
        export["var0"] = cell_diameters
        if non_vars == 2:
            export["var1"] = time_steps
        for idx, errors in zip(range(len(var_names)), var_errors):
            export[f"var{idx + non_vars}"] = errors

        # String format
        fmt = "%2.2e " * (non_vars + len(var_names))
        fmt.rstrip(" ")  # strip one space

        # Headers
        header = "cell_diameter"
        if non_vars == 2:
            header += " time_step"
        for var_name in var_names:
            header += " " + var_name

        # Write into txt
        np.savetxt(fname=file_name, X=export, header=header, fmt=fmt)  # type: ignore

    def order_of_convergence(
        self,
        list_of_results: list,
        variables: Optional[list[str]] = None,
        x_axis: Literal["cell_diameter", "time_step"] = "cell_diameter",
        base_log_x_axis: int = 2,
        base_log_y_axis: int = 2,
    ) -> dict[str, float]:
        """Compute order of convergence (OOC) for a given set of variables.

        Note:
            The OOC is computed by fitting a line for log_{base_log_y_axis}(error)
            vs. log_{base_log_x_axis}(x_axis).

        Raises:
            ValueError
                If ``x_axis`` is ``"time_step"`` and the model is stationary.

        Parameters:
            list_of_results: List containing the results of the convergence analysis.
                Typically, the output of :meth:`~run_analysis`.
            variables: ``default=None``

                Names of the variables for which the OOC should be computed. The
                ``item`` of the list must match the attribute ``"error_" + item`` from
                each item from the ``list_of_results``. If not given, all attributes
                starting with "error_" will be included in the analysis.
            x_axis: ``default=cell_diameter``

                Type of data in the x-axis used to compute the OOC.
            base_log_x_axis: ``default=2``

                Base of the logarithm for the data in the x-axis.
            base_log_y_axis: ``default=2``

                Base of the logarithm for the data in the y-axis.

        Returns:
            Dictionary containing the OOC for the given variables.

        """
        # Sanity check
        if x_axis == "time_step" and not self._is_time_dependent:
            msg = "Order of convergence cannot be estimated as a function of the time "
            msg += "step for a stationary model."
            raise ValueError(msg)

        # Get x-data
        if x_axis == "cell_diameter":
            x_data = np.array([result.cell_diameter for result in list_of_results])
        elif x_axis == "time_step":
            x_data = np.array([result.dt for result in list_of_results])
        else:
            msg = "'x_axis' must be either 'cell_diameter' or 'time_step'."
            raise NotImplementedError(msg)

        # Apply logarithm to x_data
        x_vals = np.emath.logn(base_log_x_axis, x_data)

        # Get variable names and labels
        if variables is None:
            # Retrieve all attributes from the data class. Note that we use the first
            # result from the list of results to retrieve this information. Thus, we
            # assume that all other results contain (minimally) the same information.
            attributes: list[str] = list(vars(list_of_results[0]).keys())
            # Filter attributes that whose names contain the prefix ``'error_'``
            names: list[str] = [
                attr for attr in attributes if attr.startswith("error_")
            ]
        else:
            # Not much to do here, since the user gives the variables that should be
            # retrieved
            names = [attr for attr in variables]

        # Obtain y-values
        y_vals: list[np.ndarray] = []
        for name in names:
            y_val: list[float] = []
            # Loop over lists of results
            for result in list_of_results:
                y_val.append(getattr(result, name))
            y_vals.append(np.emath.logn(base_log_y_axis, np.array(y_val)))

        # Perform linear regression and populate the return dictionary
        # Keywords of the dictionary will have the prefix "ooc_" before the `name`
        ooc_dict: dict[str, float] = {}
        for idx, name in enumerate(names):
            slope, *_ = stats.linregress(x_vals, y_vals[idx])
            ooc_name = "ooc_" + name.lstrip("error_")  # strip the prefix "error_"
            ooc_val = slope
            ooc_dict[ooc_name] = ooc_val

        return ooc_dict

    # -----> Helper methods
    def _get_list_of_meshing_arguments(self) -> list[dict[str, float]]:
        """Obtain list of meshing arguments dictionaries.

        Returns:
            List of meshing arguments dictionaries. Length of list is ``levels``.

        """
        # Retrieve initial meshing arguments
        init_mesh_args = deepcopy(self._init_setup.meshing_arguments())

        # Prepare factors for the spatial analysis
        factors = 1 / (self.spatial_refinement_rate ** np.arange(self.levels))

        # Loop through the levels and populate the list
        list_meshing_args: list[dict[str, float]] = []
        for lvl in range(self.levels):
            factor: pp.number = factors[lvl]
            meshing_args: dict[str, float] = {}
            for key in init_mesh_args:
                meshing_args[key] = init_mesh_args[key] * factor
            list_meshing_args.append(meshing_args)

        return list_meshing_args

    def _get_list_of_time_managers(self) -> Union[list[pp.TimeManager], None]:
        """Obtain list of time managers.

        Returns:
            List of time managers. Length of list is ``levels``. ``None`` is returned
            if the model is stationary.

        """
        if not self._is_time_dependent:
            return None

        # Retrieve initial time manager
        init_time_manager: pp.TimeManager = self._init_setup.time_manager

        # Sanity check
        if not init_time_manager.is_constant:
            msg = "Analysis in time only supports constant time step."
            raise NotImplementedError(msg)

        # Prepare factors for the temporal analysis
        factors = 1 / (self.temporal_refinement_rate ** np.arange(self.levels))

        # Loop over levels and populate the list
        list_time_managers: list[pp.TimeManager] = []
        for lvl in range(self.levels):
            factor = factors[lvl]
            time_manager = pp.TimeManager(
                schedule=init_time_manager.schedule,
                dt_init=init_time_manager.dt_init * factor,
                constant_dt=True,
            )
            list_time_managers.append(time_manager)

        return list_time_managers
