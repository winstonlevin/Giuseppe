from collections.abc import Hashable, Mapping, Iterator, Iterable
from typing import Union, Optional, Callable
from copy import copy

from giuseppe.data_classes import SolutionSet, Annotations
from .abstract import ContinuationSeries
from ...utils.exceptions import ContinuationError


class CustomSeries(ContinuationSeries):
    def __init__(
            self, num_steps: int, get_next_constants: Callable, solution_set: SolutionSet,
            max_bisections: int = 3, keep_bisections: bool = True, series_name: Optional[str] = None
    ):
        super().__init__(solution_set)
        self.num_steps: int = num_steps
        self.get_next_constants: Callable = get_next_constants

        self._step_size: float = 1. / self.num_steps
        self.max_bisections: int = max_bisections
        self.bisection_counter: int = 0
        self.second_bisection_half: bool = True
        self.substeps_left: int = 1
        self.keep_bisections = keep_bisections
        self.last_converged_solution = None

        if series_name is not None:
            self.series_name = series_name
        else:
            self.series_name = ''

    def __iter__(self):
        super().__iter__()
        self.current_step = 0
        self.bisection_counter = 0
        self.second_bisection_half: bool = True
        self.substeps_left: int = 1
        return self

    def __next__(self):
        if self.solution_set[-1].converged:
            self.last_converged_solution = self.solution_set[-1]

            if self.current_step == self.num_steps:
                raise StopIteration

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

            # Increment current step as a fraction
            self.current_step += 2 ** -self.bisection_counter

            next_constants = self._generate_next_constants()

        else:
            self.solution_set.damn_sol()
            if self.bisection_counter < self.max_bisections:
                # Begin first half of a new bisection. This lowers the bisection level and introduces a new solution,
                # Requiring the substeps to be incremented and the current step to be moved back by half the previous
                # step size.
                self.bisection_counter += 1
                self.second_bisection_half = False
                self.substeps_left += 1
                self.current_step -= 2 ** -self.bisection_counter
                next_constants = self._generate_next_constants()

            else:
                raise ContinuationError('Bisection limit exceeded!')

        return next_constants, self.last_converged_solution

    def _generate_next_constants(self):
        fraction_complete = float(self.current_step / self.num_steps)
        next_constants = self.get_next_constants(
            copy(self.last_converged_solution), fraction_complete
        )
        return next_constants

    def __repr__(self):
        return f'CustomSeries({self.series_name})'
