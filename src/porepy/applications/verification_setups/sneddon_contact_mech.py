"""
This module contains the implementation of Sneddon's crack pressurization problem.

The problem consists of a one-dimensional crack of length :math:`2h` immersed in an
unbounded elastic solid, forming an angle :math:`\\theta` with the horizontal axis.

A pressure :math:`p_0` is exherted on both sides of the interior faces of the crack,
causing a relative normal displacement :math:`[\\mathbf{u}]_n`. Since there are no
shear forces, the relative tangential displacement is zero.

Sneddon [1] found an exact solution for the normal relative displacement:

.. math::

    [\\mathbf{u}]_n (\\eta) = 2h p_0 (1 - \\nu) G^{-1}
        \\left[1 - \\left(\\frac{\\eta}{h}\\right)^2 \\right]^{1/2},

where :math:`\\eta` is the distance from a point in the crack to its centroid,
:math:`\\nu` is the Poisson's coefficient, and :math:`G` is the shear modulus.

Using Sneddon's exact solution, Crouch and Starfield [2] proposed a semi-analytical
procedure based on the Boundary Element Method (BEM) to replace the infinite elastic
solid by a finite, two-dimensional elastic solid of length :math:`a` and height
:math:`b`,

In this implementation, we follow the BEM recipe from [2] to obtain the displacement
at the exterior boundary sides of the solid. Moreover, since the traction force on the
fracture is known, e.g., :math:`p_0` in the normal direction and zero in the
tangential direction, we effectively do not need the fracture equations. This means
that instead of solving the full contact mechanics problem, we solve a reduced
system, where the unknowns become the displacement field in the ambient subdomain and
the displacement on the interface.

References:

    - [1] Sneddon, I.N.: Fourier Transforms. McGraw Hill Book Co, Inc., New York (1951).

    - [2] Crouch, S.L., Starfield, A.: Boundary Element Methods in Solid Mechanics:
      With Applications in Rock Mechanics and Geological Engineering. Allen & Unwin,
      London (1982).

"""