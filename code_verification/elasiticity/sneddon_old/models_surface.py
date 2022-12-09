"""
This module contains functions for solving a contact problem. It has functions
for solving the elastic equations and the time dependent Biot equations, both with a
non-linear contact condition. The non-linearity is handled by applying a semi-smooth
Newton method.
"""
import numpy as np
import scipy.sparse as sps
import porepy as pp
import pickle
import pdb

# import utils
# from contact_surface import contact_coulomb
# import solvers


def run_mechanics(setup, gb, a, theta, p0):
    """
    Function for solving linear elasticity with a non-linear Coulomb contact.

    There are some assumtions on the variable and discretization names given to the
    grid bucket:
        'u': The displacement variable
        'lam': The mortar variable
        'mpsa': The mpsa discretization

    In addition to the standard parameters for mpsa we also require the following
    under the contact mechanics keyword (returned from setup.set_parameters):
        'friction_coeff' : The coefficient of friction
        'c' : The numerical parameter in the non-linear complementary function.

    Arguments:
        setup: A setup class with methods:
                set_parameters(g, data_node, mg, data_edge): assigns data to grid bucket.
                    Returns the keyword for the linear elastic parameters and a keyword
                    for the contact mechanics parameters.
                create_grid(): Create and return the grid bucket
                initial_condition(): Returns initial guess for 'u' and 'lam'.
            and attributes:
                out_name(): returns a string. The data from the simulation will be
                written to the file 'res_data/' + setup.out_name and the vtk files to
                'res_plot/' + setup.out_name
    """
    # Extract the grids we use
    ambient_dim = gb.dim_max()

    # Pick up grid of highest dimension - there should be a single one of these
    g_2 = gb.grids_of_dimension(ambient_dim)[0]
    g_1 = gb.grids_of_dimension(ambient_dim - 1)[0]

    # Obtain simulation data for the grid, and the edge (in the GridBucket
    # sense) between the grid and itself, that is, the link over the fracture.
    # set simulation parameters
    setup.set_parameters(gb)

    # Get shortcut to some of the parameters
    # Friction coefficient
    c_num = 100

    # Define rotations
    pp.contact_conditions.set_projections(gb)

    # Set up assembler and discretize
    # setup.discretize(gb)

    assembler = pp.Assembler(gb)

    # prepare for iteration
    setup.initial_condition(assembler)
    T_contact = []
    u_contact = []
    save_sliding = []

    u0 = gb.node_props(g_2)[pp.STATE][setup.displacement_variable]
    errors = []

    # Bookkeeping within the GridBucket data dictionaries
    displacement_jump_key = "displacement_jump"

    counter_newton = 0
    converged_newton = False
    max_newton = 15

    c_num = 100

    assembler.discretize()

    viz = pp.Exporter(g_2, "output.vtu")

    while counter_newton <= max_newton and not converged_newton:
        # print('Newton iteration number: ', counter_newton, '/', max_newton)

        counter_newton += 1
        # Calculate numerical friction bound used in the contact condition
        # Clip the bound to be non-negative

        assembler.discretize(term_filter=setup.friction_coupling_term)

        # Re-discretize and solve
        A, b = assembler.assemble_matrix_rhs()

        # Store edge in variable e (there is a single edge )
        for e, d_m in gb.edges():
            mg = d_m["mortar_grid"]

        fine_dof_of_matrix = assembler.dof_ind(g_2, setup.displacement_variable)
        fine_dof_of_mortar = assembler.dof_ind(e, setup.surface_variable)
        mm_ind = np.hstack((fine_dof_of_matrix, fine_dof_of_mortar))

        mortar_ind_in_reduced = np.arange(fine_dof_of_matrix.size, mm_ind.size)

        fine_dof_fracture = assembler.dof_ind(g_1, setup.contact_variable)

        # The force in the fracture is set as unity in the normal direction,
        # zero in the
        normal_force_in_fracture = p0 * g_1.cell_volumes
        force_in_fracture = np.vstack(
            (np.zeros(g_1.num_cells), normal_force_in_fracture)
        ).ravel("F")

        # System matrix for the mortar and matrix variable
        A_matrixmortar = A[mm_ind][:, mm_ind]

        b_matrixmortar = b[mm_ind]

        A_matrixmortar_fracture = A[mm_ind][:, fine_dof_fracture]
        b_corrected = b_matrixmortar - A_matrixmortar_fracture * force_in_fracture

        u_mm = sps.linalg.spsolve(A_matrixmortar, b_corrected)

        # The last num_mortar_cells * nd elements in u_mm is the mortar variables
        # because of definition of mm_ind

        jump_u = (
            mg.mortar_to_slave_avg(nd=2)
            * mg.sign_of_mortar_sides(nd=2)
            * u_mm[mortar_ind_in_reduced]
        )

        jump_u = np.reshape(np.absolute(jump_u), (g_1.num_cells, 2))
        jump_u = jump_u[:, 0] * np.cos(theta) + jump_u[:, 1] * np.sin(theta)

        # Calculate analytical solution
        eta, jump_u_a = setup.run_analytical_displacements(gb)
        areas = g_1.cell_volumes
        error = L2_error(jump_u_a, jump_u, areas)
        i = eta < (0.9 * a)
        error_int = L2_error(jump_u_a[i], jump_u[i], areas[i])
        return error, error_int

        """

        # Split solution into displacement variable and mortar variable
        assembler.distribute_variable(u_mm)
        #u = data_node['u'].reshape((g.dim, -1), order='F')

        u = gb.node_props(g_max)[setup.displacement_variable]

        viz.write_vtk({'ux': u[::2], 'uy': u[1::2]})

        # Calculate the error
#        pdb.set_trace()
        print(mg.mortar_to_slave_avg(nd=2)
            * mg.sign_of_mortar_sides(nd=2) * d_m["mortar_u"])

        solution_norm = l2_error_cell(g_max, u)
        iterate_difference = l2_error_cell(g_max, u, u0)

        if iterate_difference / solution_norm < 1e-10:
            converged_newton = True

        print('error: ', np.sum((u - u0)**2) / np.sum(u**2))
        errors.append(np.sum((u - u0)**2) / np.sum(u**2))
#        breakpoint()
        # Prepare for next iteration
        u0 = u
        """


#    # Store vtk of solution:
#    viz.export_nodal_values(g, mg, data_node, u, None, Tc, key, key_m, setup.out_name, "res_plot")
#    if dim==3:
#        m_exp_name = setup.out_name + "_mortar_grid"
#        viz.export_mortar_grid(g, mg, data_edge, uc, Tc, key, key_m, m_exp_name, "res_plot")


def L2_norm(val, area=None):
    if area is None:
        area = np.ones(val.size) / val.size
    return np.sqrt(np.sum(np.multiply(area, np.square(val))))


def L2_error(v_ref, v_approx, area):
    enum = L2_norm(v_approx - v_ref, area)
    denom = L2_norm(v_ref, area)
    return enum / denom


def l2_error_cell(g, u, uref=None):
    if uref is None:
        norm = np.reshape(u**2, (g.dim, g.num_cells), order="F") * g.cell_volumes
    else:
        norm = (
            np.reshape((u - uref) ** 2, (g.dim, g.num_cells), order="F")
            * g.cell_volumes
        )
    print(np.sum(norm))
    return np.sum(norm)
