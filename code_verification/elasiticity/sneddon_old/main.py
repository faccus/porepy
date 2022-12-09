"""
This is a setup class for solving linear elasticity with contact between the fractures.
We do not consider any fluid, and solve only for the linear elasticity coupled to the contact.
"""
import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist
import porepy as pp
import copy

import models_surface as models
import math

def compute_eta(pointset_centers, point):
    """
    Compute the distance of bem segments centers to the
    fracture centre.

    Parameter
    ---------
    pointset_centers: array containing centers of bem segments
    point: fracture centre, middle point of the square domain

    """
    return pp.geometry.distances.point_pointset(pointset_centers, point)

def get_bem_centers(a, h, n, theta, center):
    """
    Compute coordinates of the centers of the bem segments

    Parameter
    ---------
    a: half fracture length
    h: bem segment length
    n: number of bem segments
    theta: orientation of the fracture
    center: center of the fracture
    """
    bem_centers = np.zeros((3, n))
    x_0 = center[0] - (a - 0.5 * h) * np.sin(theta)
    y_0 = center[1] - (a - 0.5 * h) * np.cos(theta)
    for i in range (0, n):
        bem_centers[0, i] = x_0 + i * h * np.sin(theta)
        bem_centers[1, i] = y_0 + i * h * np.cos(theta)

    return bem_centers

def analytical_displacements(a, eta, p0, mu, nu):
    """
    Compute Sneddon's analytical solution for the pressurized crack
    problem in question.

    Parameter
    ---------
    a: half fracture length
    eta: distance from fracture centre
    p0: pressure
    mu: shear modulus
    nu: poisson ratio
    """
    cons = (1 - nu) / mu * p0 * a * 2
    return cons * np.sqrt(1 - np.power(eta / a, 2))

def transform(xc, x, alpha):
    """
    Coordinate transofrmation for the BEM method

    Parameter
    ---------
    xc: coordinates of BEM segment centre
    x: coordinates of boundary faces
    alpha: fracture orientation
    """
    x_bar = np.zeros_like(x)
    x_bar[0,:] = (x[0,:]-xc[0])*np.cos(alpha) + (x[1,:]-xc[1])*np.sin(alpha)
    x_bar[1,:] = - (x[0,:]-xc[0])*np.sin(alpha) + (x[1,:]-xc[1])*np.cos(alpha)
    return x_bar
   
def get_bc_val(g, bound_faces, xf, h, poi, alpha, du):
    """
    Compute analytical displacement using the BEM method for the pressurized crack
    problem in question.

    Parameter
    ---------
    g: grid bucket
    bound_faces: boundary faces
    xf: coordinates of boundary faces
    h: bem segment length
    poi: Poisson ratio
    alpha: fracture orientation
    du: Sneddon's analytical relative normal displacement
    """
    f2 = np.zeros(bound_faces.size)
    f3 = np.zeros(bound_faces.size)
    f4 = np.zeros(bound_faces.size)
    f5 = np.zeros(bound_faces.size)

    u = np.zeros((g.dim, g.num_faces))

    m = 1 / (4 * np.pi * (1 - poi))

    f2[:]= m * (np.log(np.sqrt((xf[0,:]-h)**2+xf[1]**2))
                - np.log(np.sqrt((xf[0,:]+h)**2+xf[1]**2)))

    f3[:]= - m * (np.arctan2(xf[1,:], (xf[0,:]-h))
                   - np.arctan2(xf[1,:], (xf[0,:]+h)))

    f4[:]= m * (xf[1,:] / ((xf[0,:]-h)**2+xf[1,:]**2)
                 - xf[1,:] / ((xf[0,:]+h)**2+xf[1,:]**2))

    f5[:]= m * ((xf[0,:]-h) / ((xf[0,:]-h)**2+xf[1,:]**2)
                 - (xf[0,:]+h) / ((xf[0,:]+h)**2+xf[1,:]**2))

    u[0, bound_faces] = du * (-(1 -2 *poi)*np.cos(alpha)*f2[:]
                               -2*(1-poi)*np.sin(alpha)*f3[:]
                               -xf[1,:]*(np.cos(alpha)*f4[:]+np.sin(alpha)*f5[:]))
    u[1, bound_faces] = du * (-(1 -2 *poi)*np.sin(alpha)*f2[:]
                               +2*(1-poi)*np.cos(alpha)*f3[:]
                               -xf[1,:]*(np.sin(alpha)*f4[:]-np.cos(alpha)*f5[:]))

    return u
    
def assign_bem(g, h, bound_faces, theta, bem_centers, u_a, poi):

    """
    Compute analytical displacement using the BEM method for the pressurized crack
    problem in question.

    Parameter
    ---------
    g: grid bucket
    h: bem segment length
    bound_faces: boundary faces
    theta: fracture orientation
    bem_centers: bem segments centers
    u_a: Sneddon's analytical relative normal displacement
    poi: Poisson ratio
    """
    
    bc_val = np.zeros((g.dim, g.num_faces))

    alpha = np.pi / 2 - theta

    bound_face_centers = g.face_centers[:, bound_faces]
    
    for i in range (0, u_a.size):

        new_bound_face_centers = transform(bem_centers[:,i],
                                           bound_face_centers, alpha)

        u_bound = get_bc_val(g, bound_faces, new_bound_face_centers,
                             h, poi, alpha, u_a[i])

        bc_val += u_bound

    return bc_val

class SneddonSetup:
    """
    In this example, a square domain with a single pressurized fracture located in the middle of the domain, is considered.
    The fracture forms an angle beta with the horizontal direction, and is subjected to a constant pressure p_0 acting on its interior.
    The pressure can be interpreted as a pair of normal forces acting on either side of the fracture.
    An analytical solution for the relative normal displacement along the fracture was derived by Sneddon.
    Here, the conditions of infinite domain are replaced with a Dirichlet boundary.
    The prescribed displacement is set equal to the analytical solution calculated using the BEM method.
    The full study contains 20 x 7 x 6 = 840 simulations,
    6 fracture resolutions (from 4 to 128 fracture elements),
    20 independent grids,
    7 fracture orientations.
    Comparison is made between Sneddon's analytical solution and the average numerical solution.
    The method provides first-order convergence on average.
    """
    def __init__(self):

        self.displacement_variable = "u"
        self.surface_variable = "mortar_u"
        self.contact_variable = "contact_traction"

        self.friction_parameter_key = "friction"
        self.surface_parameter_key = "surface"
        self.mechanics_parameter_key = "mechanics"

        self.friction_coupling_term = "contact_conditions"

    def create_grid(self, mesh_args):
        """
        Method that creates and returns the GridBucket of a 2D domain with one
        single fracture. The two sides of the fractures are coupled together with a
        mortar grid.
        """

        network = pp.FractureNetwork2d(pts = frac_pts, edges=frac_edges, domain=box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(mesh_args)

        # Remove fracture grid as it is not needed
        # This operation will also create an edge between the 2d grid and
        # itself.

        return gb

    def set_parameters(self, gb):
        """
        Set the parameters for the simulation. The stress is given in GPa.
        """

        ambient_dim = gb.dim_max()

        for g, d in gb:
            if g.dim == ambient_dim:
                # Rock parameters
                lam = np.ones(g.num_cells) * 2 * G * poi / (1 - 2 * poi)
                mu = np.ones(g.num_cells) * G
                k = pp.FourthOrderTensor(g.dim, mu, lam)

                # Define boundary regions
                bound_faces = g.get_all_boundary_faces()
                box_faces = g.get_boundary_faces()

                # Define boundary condition on sub_faces
                bc = pp.BoundaryConditionVectorial(g, bound_faces, "dir")

                # Set the boundary values
                u_bc = np.zeros((g.dim, g.num_faces))

                # apply sneddon analytical solution through BEM method
                n = 1000
                h = 2 * a / n
                center = np.array([length / 2, height / 2, 0])
                bem_centers = get_bem_centers(a, h, n, theta, center)
                eta = compute_eta(bem_centers, center)
                u_a = analytical_displacements(a, eta, p0, G, poi)

                u_bc = assign_bem(g, h/2, box_faces, theta, bem_centers, u_a, poi)

                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": u_bc.ravel("F"),
                        "source": 0,
                        "fourth_order_tensor": k,
                    },
                )

            elif g.dim == 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.friction_parameter_key,
                    {"friction_coefficient": friction},
                )

        for e, d in gb.edges():
            mg = d["mortar_grid"]

            # Parameters for the surface diffusion. No clue about values
            mu = 1
            lmbda = 1

            pp.initialize_data(
                mg, d, self.friction_parameter_key, {"mu": mu, "lambda": lmbda}
            )

        # Define discretization
        # For the 2D domain we solve linear elasticity with mpsa.
        mpsa = pp.Mpsa(self.mechanics_parameter_key)

        empty_discr = pp.VoidDiscretization(
            self.friction_parameter_key, ndof_cell=ambient_dim
        )

        coloumb = pp.ColoumbContact(self.friction_parameter_key, ambient_dim)

        # Define discretization parameters

        for g, d in gb:
            if g.dim == ambient_dim:
                d[pp.PRIMARY_VARIABLES] = {
                    self.displacement_variable: {"cells": ambient_dim}
                }
                d[pp.DISCRETIZATION] = {self.displacement_variable: {"mpsa": mpsa}}
            elif g.dim == ambient_dim - 1:
                d[pp.PRIMARY_VARIABLES] = {
                    self.contact_variable: {"cells": ambient_dim}
                }
                d[pp.DISCRETIZATION] = {self.contact_variable: {"empty": empty_discr}}
            else:
                d[pp.PRIMARY_VARIABLES] = {}

        # And define a Robin condition on the mortar grid
        contact = pp.PrimalContactCoupling(self.friction_parameter_key, mpsa, coloumb)

        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)

            if g_h.dim == ambient_dim:
                d[pp.PRIMARY_VARIABLES] = {
                    self.surface_variable: {"cells": ambient_dim}
                }

                #                d[pp.DISCRETIZATION] = {self.surface_variable: {'surface_mpsa': mpsa_surface}}
                d[pp.COUPLING_DISCRETIZATION] = {
                    self.friction_coupling_term: {
                        g_h: (self.displacement_variable, "mpsa"),
                        g_l: (self.contact_variable, "empty"),
                        (g_h, g_l): (self.surface_variable, contact),
                    }
                }
            else:
                d[pp.PRIMARY_VARIABLES] = {}

    def discretize(self, gb):
        g_max = gb.grids_of_dimension(gb.dim_max())[0]
        d = gb.node_props(g_max)

        mpsa = d[pp.DISCRETIZATION][self.displacement_variable]["mpsa"]
        mpsa.discretize(g_max, d)

    def initial_condition(self, assembler):
        """
        Initial guess for Newton iteration.
        """
        gb = assembler.gb

        ambient_dimension = gb.dim_max()

        for g, d in gb:
            d[pp.STATE] = {}
            if g.dim == ambient_dimension:
                # Initialize displacement variable
                ind = assembler.dof_ind(g, self.displacement_variable)
                d[pp.STATE][self.displacement_variable] = np.zeros_like(ind)

            elif g.dim == ambient_dimension - 1:
                # Initialize contact variable
                ind = assembler.dof_ind(g, self.contact_variable)

                # Initialize with positive normal components, this is
                # interpreted as opening. 
                traction = np.vstack((np.zeros(g.num_cells),
                                      100 * np.ones(g.num_cells))).ravel(order='F')

                d[pp.STATE][self.contact_variable] = traction
                d[pp.STATE]["previous_iterate"] = {self.contact_variable: traction}

        for e, d in gb.edges():
            d[pp.STATE] = {}

            mg = d["mortar_grid"]

            if mg.dim == 1:
                ind = assembler.dof_ind(e, self.surface_variable)
                d[pp.STATE][self.surface_variable] = np.zeros_like(ind)
                d[pp.STATE]["previous_iterate"] = {self.surface_variable : np.zeros_like(ind)}



    def _set_friction_coefficient(self, g):

        nodes = g.nodes

        tips = nodes[:, [0, -1]]

        fc = g.cell_centers
        D = cdist(fc.T, tips.T)
        D = np.min(D, axis=1)
        R = 200
        beta = 10
        friction_coefficient = 0.5 * (1 + beta * np.exp(-R * D ** 2))
        friction_coefficient = 0.5 * np.ones(g.num_cells)
        return friction_coefficient

    def run_analytical_displacements(self, gb):

        ambient_dim = gb.dim_max()
        g_1 = gb.grids_of_dimension(ambient_dim - 1)[0]

        fracture_center = np.array([length / 2, height / 2, 0])

        fracture_faces = g_1.cell_centers
        eta = compute_eta(fracture_faces, fracture_center)
        
        cons = (1 - poi) / G * p0 * a * 2
        apertures = cons * np.sqrt(1 - np.power(eta / a, 2))

        return eta, apertures

if __name__ == "__main__":

    # Geometry
    height = 50
    length = 50
    a = 10

    # physics
    p0 = 1e-4 # crack pressure
    G = 1  # shear modulus
    poi = .25 # poisson modulus

    theta0 = 0
    num_theta = 7
    delta_theta = 5

    nf0 = 4
    num_refs = 6

    num_grids = 20
    percent = 2 / 100

    avg_err = []

    for i in range (0, num_refs):
        nf = nf0 * 2**i
        h = 2 * a / nf

        print('number of faces', nf)

        theta_list = []
        avg_err_theta = []

        for k in range (0, num_theta):
            theta = theta0 + k * delta_theta
            print('theta', theta)
            theta = math.radians(90-theta)

            y_0 = height / 2 - a * np.cos(theta)
            x_0 = length / 2 - a * np.sin(theta)
            y_1 = height / 2 + a * np.cos(theta)
            x_1 = length / 2 + a * np.sin(theta)

            frac_pts = np.array([[x_0, y_0], [x_1, y_1]]).T
            frac_edges = np.array([[0,1]]).T

            box = {'xmin': 0, 'ymin': 0, 'xmax': length, 'ymax': height}

            err_pert = []

            for j in range (0, num_grids):

                pert = percent * h
                hb = (1 + j * pert) * h

                mesh_args = {
                    "mesh_size_frac": h,
                    "mesh_size_min": h/200,
                    "mesh_size_bound": hb,
                }

                setup = SneddonSetup()

                gb = setup.create_grid(mesh_args)
                    
                e, _ = models.run_mechanics(setup, gb, a, theta, p0)
                err_pert.append(e)

            err_pert_vec = np.array(err_pert)
            err_pert_avg = np.sum(err_pert_vec) / err_pert_vec.size
            avg_err_theta.append(err_pert_avg)

        avg_err_theta_vec = np.array(avg_err_theta)
        avg_err_theta_avg = np.sum(avg_err_theta_vec) / avg_err_theta_vec.size
        avg_err.append(avg_err_theta_avg)
        print('average error per theta', avg_err_theta)
        print('average error', avg_err)

