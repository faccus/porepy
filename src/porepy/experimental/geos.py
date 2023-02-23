"""
Single fracture decomposed into two segmenets + circle arcs as auxiliary lines.

// Mesh target
cl__1 = 0.1;

// ##### Points
// Fracture Tips
Point(1) = {0.25, 0.5, 0, cl__1};
Point(2) = {0.75, 0.5, 0, cl__1};
// Bounding box
Point(3) = {0, 0, 0, cl__1};
Point(4) = {1, 0, 0, cl__1};
Point(5) = {1, 1, 0, cl__1};
Point(6) = {0, 1, 0, cl__1};
// Circle endopoints
Point(7) = {0.20, 0.50, 0, cl__1};
Point(8) = {0.30, 0.50, 0, cl__1};

// ##### Lines & Arcs
// Boundary lines
Line(1) = {3, 4};
Line(2) = {4, 5};
Line(3) = {5, 6};
Line(4) = {3, 6};
// Fracture
Line(5) = {1, 8};
Line(6) = {8, 2};
// "Left" circle arcs
Circle(7) = {7, 1, 8};
Circle(8) = {8, 1, 7};

// ##### Surfaces
// Define curve loop from boundary lines
Curve Loop(1) = {1, 2, 3, -4};
// Create surface
Plane Surface(1) = {1};

// ##### Embed objects
Line {5} In Surface {1};  // fracture
Line {6} In Surface {1};  // fracture
Line {7} In Surface {1};  // arc
Line {8} In Surface {1};  // arc

// ##### Physical groups
Physical Point("DOMAIN_BOUNDARY_POINT_2") = {3};
Physical Point("DOMAIN_BOUNDARY_POINT_3") = {4};
Physical Point("DOMAIN_BOUNDARY_POINT_4") = {5};
Physical Point("DOMAIN_BOUNDARY_POINT_5") = {6};
Physical Curve("DOMAIN_BOUNDARY_LINE_1") = {1};
Physical Curve("DOMAIN_BOUNDARY_LINE_2") = {2};
Physical Curve("DOMAIN_BOUNDARY_LINE_3") = {3};
Physical Curve("DOMAIN_BOUNDARY_LINE_4") = {4};
Physical Curve("FRACTURE_0") = {5, 6};  // Physical fracture = lines 5 + 6
Physical Curve("AUXILIARY_LINE_1") = {7, 8};  // Constrained left arc = curves 7 + 8
Physical Surface("DOMAIN") = {1};

"""


