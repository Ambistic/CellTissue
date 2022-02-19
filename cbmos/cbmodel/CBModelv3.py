import numpy as _np
import numpy.random as _npr
import heapq as _hq
import logging as _logging

import time

from .. import cell as _cl

from ._eventqueue import EventQueue
from ..utils import generate_cartesian_coordinates

_NU = 1


class CBModelv3:
    """
    Parameters
    -----------
        force: `f(ndarray(dtype=float), **kwargs)` -> float
            forces to be applied between cells
        solver: `f(fun, t_span, y0)` -> scipy.intergrade._ivp.ivp.OdeResult
            ODE solver, e.g. solve_ivp from scipy.integrate
        dimension: int
            dimension of the system, usually 2D or 3D
        separation: float
            distance between parent cell and child cell after separation
        hpc_backend: module
            module implementing Numpy's API (e.g. Cupy, Dask). Default is
            Numpy itself.

    """
    def __init__(self, force, solver, force_args, solver_args, cells=None,
                 dimension=3, separation=0.3, hpc_backend=_np,
                ):
        self.force = force
        self.solver = solver
        self.force_args = force_args
        self.solver_args = solver_args
        if cells is not None:
            self.build_cells(cells)
        else:
            self.cells = list()
            self.ids_to_cells = dict()
            self.next_cell_index = 0
        
        self.dim = dimension
        self.separation = separation
        self.hpc_backend = hpc_backend
        
        self.history = []
        
    def build_cells(self, cells):
        self.cells = cells
        self.ids_to_cells = {c.ID: c for c in self.cells}
        self.next_cell_index = max([c.ID for c in self.cells]) + 1
        
    def init_square(self, size=6):
        coordinates = generate_cartesian_coordinates(size, size)
        cells = [_cl.Cell(i, [x, y])
                 for i, (x, y) in enumerate(coordinates)]
        self.build_cells(cells)
        
    def export(self):
        return {c.ID: c.position for c in self.history[-1]}
        
    def get_ids(self):
        return [c.ID for c in self.cells]
        
    def tick(self, current_time, time_step):
        y0 = _np.array([cell.position for cell in self.cells]).reshape(-1)

        sol = self._calculate_next_positions(current_time, current_time + time_step,
                                             y0)

        # save data for all t_data points passed
        self._save_data(sol.y[:, -1].reshape(-1, self.dim))

        # update the positions for the current time point
        self._update_positions(sol.y[:, -1].reshape(-1, self.dim).tolist())
        
    def _calculate_next_positions(self, time_start, time_end, y0, raw_t=True):
        _logging.debug("Calling solver with: t0={}, tf={}".format(
            time_start, time_end))
        return self.solver(self._ode_system(self.force_args),
                           (time_start, time_end),
                           y0,
                           **self.solver_args)
    
    def divide_cell(self, cell_id):
        cell = self.ids_to_cells[cell_id]
        del self.ids_to_cells[cell_id]
        cell, daughter_cell = self._apply_division(cell, 0)
        self.ids_to_cells[daughter_cell.ID] = daughter_cell
        self.ids_to_cells[cell.ID] = cell
        return cell.ID, daughter_cell.ID
    
    def remove_cell(self, cell_id):
        cell = self.ids_to_cells[cell_id]
        del self.ids_to_cells[cell_id]
        self.cells.remove(cell)
        
    def get_neighbours(self, cell_id):
        # compute all dists, then pick ones below a threshold
        ref = self.ids_to_cells[cell_id]
        dists = [(self.hpc_backend.linalg.norm(ref.position - cell.position), cell.ID)
                  for cell in self.cells]
        dists.sort()
        idx = 0
        _len = len(dists)
        while idx < _len and dists[idx][0] < 1.5:
            idx += 1
        return [x[1] for x in dists[1:idx]]
    
    def get_all_neighbours(self):
        pass

    def _save_data(self, positions=None):
        """
        Save the current positions of the cells to `self.history`. If
        `positions` is provided, uses theses positions instead of the cells'
        own positions.
        Note
        ----
        self.history has to be instantiated before the first call to _save_data
        as an empty list.
        """
        # copy correct positions to cell list that is stored
        if positions is not None:
            self.history.append([_cl.Cell(
                    cell.ID, pos, cell.birthtime, cell.proliferating,
                    cell.generate_division_time, cell.division_time,
                    cell.parent_ID)
                for cell, pos in zip(self.cells, positions)])
        else:
            self.history.append([_cl.Cell(
                    cell.ID, cell.position, cell.birthtime, cell.proliferating,
                    cell.generate_division_time, cell.division_time,
                    cell.parent_ID)
                for cell in self.cells])

    def _get_division_direction(self):

        if self.dim == 1:
            division_direction = _np.array([-1.0 + 2.0 * _npr.randint(2)])

        elif self.dim == 2:
            random_angle = 2.0 * _np.pi * _npr.rand()
            division_direction = _np.array([
                _np.cos(random_angle),
                _np.sin(random_angle)])

        elif self.dim == 3:
            u = _npr.rand()
            v = _npr.rand()
            random_azimuth_angle = 2 * _np.pi * u
            random_zenith_angle = _np.arccos(2 * v - 1)
            division_direction = _np.array([
                _np.cos(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.sin(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.cos(random_zenith_angle)])
        return division_direction

    def _apply_division(self, cell, tau):
        """
        Note
        ----
        The code assumes that all cell events are division events,
        """

        division_direction = self._get_division_direction()
        updated_position_parent = cell.position -\
            0.5 * self.separation * division_direction
        position_daughter = cell.position +\
            0.5 * self.separation * division_direction
        
        current_cell_index = self.next_cell_index

        daughter_cell = _cl.Cell(
                current_cell_index, position_daughter, birthtime=tau,
                proliferating=True,
                division_time_generator=cell.generate_division_time,
                parent_ID=cell.ID)
        self.next_cell_index = self.next_cell_index + 1
        self.cells.append(daughter_cell)

        cell.ID = self.next_cell_index
        self.next_cell_index = self.next_cell_index + 1

        cell.position = updated_position_parent
        cell.division_time = cell.generate_division_time(tau)

        _logging.debug("Division event: t={}, direction={}".format(
            tau, division_direction))
        
        return cell, daughter_cell

    def _calculate_positions(self, t_eval, y0, force_args, solver_args, raw_t=True):
        _logging.debug("Calling solver with: t0={}, tf={}".format(
            t_eval[0], t_eval[-1]))
        return self.solver(self._ode_system(force_args),
                           (t_eval[0], t_eval[-1]),
                           y0,
                           t_eval=t_eval if not raw_t else None,
                           **solver_args)

    def _update_positions(self, y):
        """
        Note
        ----
        The ordering in cell_list and sol.y has to match.
        """

        for cell, pos in zip(self.cells, y):
            cell.position = _np.array(pos)

    def _ode_system(self, force_args):
        """ Generate ODE force function from cell-cell force function

        Parameters
        ----------
        force_args: {str: float}
            extra arguments for the force function

        Returns
        -------
        f: (t, y) -> dy/dt

        """
        def f(t, y):
            y_r = _np.expand_dims(
                    self.hpc_backend.asarray(y).reshape((-1, self.dim)),
                    axis=-1,
                    ) # shape (n, d, 1)
            cross_diff = y_r.transpose([2, 1, 0]) - y_r # shape (n, d, n)
            norm = _np.sqrt((cross_diff**2).sum(axis=1)) # shape (n, n)
            forces = _np.expand_dims(
                self.force(norm, **force_args)\
                    / (norm + _np.diag(self.hpc_backend.ones(y_r.shape[0]))),
                axis=1,
                ) # shape (n, 1, n)
            total_force = (forces * cross_diff).sum(axis=2) # shape (n, d)

            fty = (_NU*total_force).reshape(-1)

            if self.hpc_backend.__name__ == "cupy":
                return self.hpc_backend.asnumpy(fty)
            else:
                return _np.asarray(fty)

        return f

    def jacobian(self, y, force_args):
        """ Compute the jacobian of the given ode system.

        Parameters
        ----------
        y: np.ndarray(size=(n_cell*dim,))
            cell vector
        force_args: {str: float}
            extra arguments for the force function

        Returns
        -------
        np.ndarray(size=(n_cell*dim, n_cell*dim))
        """

        y_r = self.hpc_backend.asarray(_np.expand_dims(y.reshape((-1, self.dim)), axis=-1))
        n = y_r.shape[0]
        cross_diff = y_r - y_r.transpose([2, 1, 0]) # shape (n, d, n)
        norm = _np.sqrt((cross_diff**2).sum(axis=1))
        r_hat = _np.expand_dims(_np.moveaxis(cross_diff, 1, 2), axis=-1) # shape (n, n, d, 1)

        B = r_hat @ r_hat.transpose([0, 1, 3, 2]) # shape (n, n, d, d)

        with _np.errstate(divide='ignore', invalid='ignore'):
            # Ignore divide by 0 warnings
            # All NaNs are removed below

            # add normalization
            B = B / (norm*norm)[:, :, _np.newaxis, _np.newaxis]

            B = (
                    B*(self.force.derive()(norm, **force_args)-self.force(norm, **force_args)/norm)[:, :, _np.newaxis, _np.newaxis]
                    + (self.hpc_backend.identity(self.dim))[_np.newaxis, _np.newaxis, :, :]
                    * (self.force(norm, **force_args)/norm)[:, :, _np.newaxis, _np.newaxis]
                    )

            B[_np.isnan(B)] = 0

        # Step 2: compute the diagonal
        B[_np.array(range(n)), _np.array(range(n)), :, :] = - B.sum(axis=0)

        # Step 3: Build block matrix
        B_block =  B.reshape(n, n, self.dim, self.dim).swapaxes(1, 2).reshape(self.dim*n, -1)

        if self.hpc_backend.__name__ == "cupy":
            return self.hpc_backend.asnumpy(B_block)
        else:
            return _np.asarray(B_block)
