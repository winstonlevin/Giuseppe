from __future__ import annotations
from collections.abc import Hashable, Mapping, Iterable
from typing import Union, Optional
from copy import copy

import numpy as np

from giuseppe.data_classes import SolutionSet, Annotations
from .abstract import ContinuationSeries
from ...utils.typing import NPArray


class UntilFailureSeries(ContinuationSeries):
    def __init__(
            self, step_mapping: Mapping[Hashable: float], solution_set: SolutionSet,
            max_bisections: int = 3, constant_names: Optional[Union[Iterable[Hashable, ...], Annotations]] = None,
            keep_bisections: bool = True
    ):

        super().__init__(solution_set)
        self.step_mapping: Mapping[Hashable: float] = step_mapping

        if constant_names is None:
            self.constant_names: tuple[Hashable, ...] = tuple(range(len(self.solution_set[-1].k)))
        elif isinstance(constant_names, Annotations):
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names.constants)
        else:
            self.constant_names: tuple[Hashable, ...] = tuple(constant_names)

        self.constant_indices = self._get_constant_indices()
        self.constant_steps = np.fromiter(self.step_mapping.values(), dtype=float)
        self.idx_tar_pairs: list[tuple[int, float]] = \
            [(idx, tar) for idx, tar in zip(self._get_constant_indices(), step_mapping.values())]

        self._step_size: NPArray
        self.max_bisections: int = max_bisections
        self.bisection_counter: int = 0
        self.second_bisection_half: bool = True
        self.substeps_left: int = 1
        self.keep_bisections = keep_bisections
        self.last_converged_solution = None

    def __iter__(self):
        super().__iter__()
        self.bisection_counter = 0
        self.second_bisection_half: bool = True
        self.substeps_left: int = 1
        return self

    def __next__(self):
        if self.solution_set[-1].converged:
            self.last_converged_solution = self.solution_set[-1]

            self.current_step += 1

            if self.bisection_counter > 0:
                # If the first half of a bisection is completed, move to the second half.
                # If the second half is completed, move up a bisection level.
                if not self.second_bisection_half:
                    self.second_bisection_half = True
                else:
                    self.bisection_counter -= 1

                # The bisection being successfully completed, there is one fewer substeps left.
                self.substeps_left -= 1

                # Reset Bisection counter when the original step is completed
                if self.substeps_left == 0:
                    self.bisection_counter = 0
                    self.substeps_left = 1

                # By default, the bisected solutions are inserted into the solution set. If the user does not want these
                # solutions, damn them now.
                if not self.keep_bisections:
                    self.solution_set.damn_sol()

            next_constants = self._generate_next_constants()

        else:
            self.solution_set.damn_sol()
            if self.bisection_counter < self.max_bisections:
                # Begin first half of a new bisection. This lowers the bisection level and introduces a new solution,
                # Requiring the substeps and to be incremented.
                self.bisection_counter += 1
                self.second_bisection_half = False
                self.substeps_left += 1
                next_constants = self._generate_next_constants()

            else:
                raise StopIteration

        return next_constants, self.last_converged_solution

    def __repr__(self):
        return f'UntilFailureSeries({self.generate_target_mapping_str()})'

    def _generate_next_constants(self):
        next_constants = copy(self.last_converged_solution.k)
        next_constants[self.constant_indices] += self.constant_steps * 2 ** -self.bisection_counter
        return next_constants

    def _get_constant_indices(self) -> list[int]:
        indices = []
        for constant_key, target_value in self.step_mapping.items():
            try:
                indices.append(self.constant_names.index(constant_key))
            except ValueError:
                raise KeyError(f'Cannot perform continuation on {constant_key} because it is not a defined constant')

        return indices

    def generate_target_mapping_str(self):
        return self.generate_mapping_str(self.step_mapping.values())

    def generate_mapping_str(self, values):
        name_str = ', '.join(self.step_mapping.keys())
        val_str = ', '.join(f'{float(val):.2}' for val in values)
        return f'{name_str} += {val_str}'
