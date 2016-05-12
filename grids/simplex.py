# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:31:31 2016

@author: keile
"""

import numpy as np
import scipy.sparse as sps
import scipy.spatial

from core.grids.grid import Grid, GridType
from utils import setmembership
from utils import accumarray


class TriangleGrid(Grid):

    def __init__(self, p, tri=None):
        """
        Create triangular grid from point cloud.

        If no triangulation is provided, Delaunay will be applied.

        Examples:
        >>> p = np.random.rand(2, 10)
        >>> tri = scipy.spatial.Delaunay(p.transpose()).simplices
        >>> g = TriangleGrid(p, tri.transpose())

        Parameters
        ----------
        p (np.ndarray, 2 x num_nodes): Point coordinates
        tri (np.ndarray, 3 x num_cells): Cell-node connections. If not
        provided, a Delaunay triangulation will be applied
        """

        self.dim = 2
        self.type = GridType.triangle

        # Transform points to column vector if necessary (scipy.Delaunay
        # requires this format)
        pdims = p.shape

        if p.shape[0] != 2:
            raise NotImplementedError("Have not yet implemented triangle grids "
                                      "embeded in 2D")
        if tri is None:
            tri = scipy.spatial.Delaunay(p.transpose())
            tri = tri.simplices
            tri = tri.transpose()

        nodes = p
        num_nodes = nodes.shape[1]
        assert num_nodes > 2   # Check of transposes of point array

        # Face node relations
        face_nodes = np.hstack((tri[[0, 1]],
                                tri[[1, 2]],
                                tri[[2, 0]])).transpose()
        face_nodes.sort(axis=1)
        face_nodes, tmp, cell_faces = setmembership.unique_rows(face_nodes)

        num_faces = face_nodes.shape[0]
        num_cells = tri.shape[1]

        num_nodes_per_face = 2
        face_nodes = face_nodes.ravel(0)
        indptr = np.hstack((np.arange(0, num_nodes_per_face * num_faces,
                                      num_nodes_per_face),
                            num_nodes_per_face * num_faces))
        data = np.ones(face_nodes.shape, dtype=bool)
        face_nodes = sps.csc_matrix((data, face_nodes, indptr),
                                    shape=(num_nodes, num_faces))

        # Cell face relation
        num_faces_per_cell = 3
        cell_faces = cell_faces.reshape(num_faces_per_cell, num_cells).ravel(1)
        indptr = np.hstack((np.arange(0, num_faces_per_cell*num_cells,
                                      num_faces_per_cell),
                            num_faces_per_cell * num_cells))
        data = -np.ones(cell_faces.shape)
        tmp, sgns = np.unique(cell_faces, return_index=True)
        data[sgns] = 1
        cell_faces = sps.csc_matrix((data, cell_faces, indptr),
                                    shape=(num_faces, num_cells))

        super(TriangleGrid, self).__init__(2, nodes, face_nodes, cell_faces,
                                           'TriangleGrid')

    def cell_node_matrix(self):
        """ Get cell-node relations in a Nc x 3 matrix
        Perhaps move this method to a superclass when tet-grids are implemented
        """

        # Absolute value needed since cellFaces can be negative
        cn = self.face_nodes * np.abs(self.cell_faces) \
             * sps.eye(self.num_cells)
        row, col = cn.nonzero()
        scol = np.argsort(col)

        # Consistency check
        assert np.all(accumarray.accum(col, np.ones(col.size)) ==
                      (self.dim + 1))

        return row[scol].reshape(self.num_cells, 3)


