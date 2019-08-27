# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import multiprocessing
import platform
import logging

import numpy as np
from scipy import optimize as sciopt

from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import Optimizer

logger = logging.getLogger(__name__)


class P_BFGS(Optimizer):
    """Limited-memory BFGS algorithm. Parallel instantiations.

    Uses scipy.optimize.fmin_l_bfgs_b
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    """

    CONFIGURATION = {
        'name': 'P_BFGS',
        'description': 'Parallelized l_bfgs_b Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'p_bfgs_b_schema',
            'type': 'object',
            'properties': {
                'maxfun': {
                    'type': 'integer',
                    'default': 1000
                },
                'factr': {
                    'type': 'integer',
                    'default': 10
                },
                'iprint': {
                    'type': 'integer',
                    'default': -1
                },
                'max_processes': {
                    'type': ['integer', 'null'],
                    'minimum': 1,
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxfun', 'factr', 'iprint'],
        'optimizer': ['local', 'parallel']
    }

    def __init__(self, maxfun=1000, factr=10, iprint=-1, max_processes=None):
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

        Args:
            maxfun (int): Maximum number of function evaluations.
            factr (float): The iteration stops when (f^k - f^{k+1})/max{|f^k|,
                           |f^{k+1}|,1} <= factr * eps, where eps is the machine precision,
                           which is automatically generated by the code. Typical values for
                           factr are: 1e12 for low accuracy; 1e7 for moderate accuracy;
                           10.0 for extremely high accuracy. See Notes for relationship to ftol,
                           which is exposed (instead of factr) by the scipy.optimize.minimize
                           interface to L-BFGS-B.
            iprint (int): Controls the frequency of output. iprint < 0 means no output;
                          iprint = 0 print only one line at the last iteration; 0 < iprint < 99
                          print also f and |proj g| every iprint iterations; iprint = 99 print
                          details of every iteration except n-vectors; iprint = 100 print also the
                          changes of active set and final x; iprint > 100 print details of
                          every iteration including x and g.
            max_processes (int): maximum number of processes allowed.
        """
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v
        self._max_processes = max_processes

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        num_procs = multiprocessing.cpu_count() - 1
        num_procs = num_procs if self._max_processes is None else min(num_procs, self._max_processes)
        num_procs = num_procs if num_procs >= 0 else 0

        if platform.system() == "Windows":
            num_procs = 0
            logger.warning("Using only current process. Multiple core use not supported in Windows")

        queue = multiprocessing.Queue()
        threshold = 2*np.pi  # bounds for additional initial points in case bounds has any None values
        low = [(l if l is not None else -threshold) for (l, u) in variable_bounds]
        high = [(u if u is not None else threshold) for (l, u) in variable_bounds]

        def optimize_runner(_queue, _i_pt):  # Multi-process sampling
            _sol, _opt, _nfev = self._optimize(num_vars, objective_function, gradient_function, variable_bounds, _i_pt)
            _queue.put((_sol, _opt, _nfev))

        # Start off as many other processes running the optimize (can be 0)
        processes = []
        for i in range(num_procs):
            i_pt = aqua_globals.random.uniform(low, high)  # Another random point in bounds
            p = multiprocessing.Process(target=optimize_runner, args=(queue, i_pt))
            processes.append(p)
            p.start()

        # While the one _optimize in this process below runs the other processes will be running to. This one runs
        # with the supplied initial point. The process ones have their own random one
        sol, opt, nfev = self._optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        for p in processes:
            # For each other process we wait now for it to finish and see if it has a better result than above
            p.join()
            p_sol, p_opt, p_nfev = queue.get()
            if p_opt < opt:
                sol, opt = p_sol, p_opt
            nfev += p_nfev

        return sol, opt, nfev

    def _optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)

        approx_grad = True if gradient_function is None else False
        sol, opt, info = sciopt.fmin_l_bfgs_b(objective_function, initial_point, bounds=variable_bounds,
                                              fprime=gradient_function, approx_grad=approx_grad, **self._options)
        return sol, opt, info['funcalls']
