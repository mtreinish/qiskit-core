# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
ParameterExpression Class to enable creating simple expressions of Parameters.
"""
from typing import Callable, Dict, Set, Union

import cmath
import math
import numbers
import operator

import numpy

from qiskit.circuit.exceptions import CircuitError

ParameterValueType = Union['ParameterExpression', float, int]


class ParameterExpression:
    """ParameterExpression class to enable creating expressions of Parameters."""

    __slots__ = ['_parameter_symbols', '_parameters', '_symbol_expr', '_names', '_bound_values',
                 '_parameter_map', '_str_expr']

    def __init__(self, symbol_map: Dict, expr, str_expr, bounds=None, parameter_map=None):
        """Create a new :class:`ParameterExpression`.

        Not intended to be called directly, but to be instantiated via operations
        on other :class:`Parameter` or :class:`ParameterExpression` objects.

        Args:
            symbol_map (Dict[Parameter, [ParameterExpression, float, or int]]):
                Mapping of :class:`Parameter` instances to the :class:`sympy.Symbol`
                serving as their placeholder in expr.
            expr (callable): Expression of s.
        """
        self._parameter_symbols = symbol_map
        self._parameters = set(self._parameter_symbols)
        if parameter_map is None:
            self._parameter_map = {}
        else:
            self._parameter_map = parameter_map
        if bounds is None:
            self._bound_values = {}
        else:
            self._bound_values = bounds
        self._str_expr = str_expr
        self._symbol_expr = expr
        self._names = None

    @property
    def parameters(self) -> Set:
        """Returns a set of the unbound Parameters in the expression."""
        return self._parameters

    def conjugate(self) -> 'ParameterExpression':
        """Return the conjugate."""

        def _conj(**kwargs):
            return self._symbol_expr(**kwargs).conjugate()

        conjugated = ParameterExpression(self._parameter_symbols, _conj)
        return conjugated

    def assign(self, parameter, value: ParameterValueType) -> 'ParameterExpression':
        """
        Assign one parameter to a value, which can either be numeric or another parameter
        expression.

        Args:
            parameter (Parameter): A parameter in this expression whose value will be updated.
            value: The new value to bind to.

        Returns:
            A new expression parameterized by any parameters which were not bound by assignment.
        """
        if isinstance(value, ParameterExpression):
            return self.subs({parameter: value})
        return self.bind({parameter: value})

    def bind(self, parameter_values: Dict) -> 'ParameterExpression':
        """Binds the provided set of parameters to their corresponding values.

        Args:
            parameter_values: Mapping of Parameter instances to the numeric value to which
                              they will be bound.

        Raises:
            CircuitError:
                - If parameter_values contains Parameters outside those in self.
                - If a non-numeric value is passed in parameter_values.
            ZeroDivisionError:
                - If binding the provided values requires division by zero.

        Returns:
            A new expression parameterized by any parameters which were not bound by
            parameter_values.
        """

        self._raise_if_passed_unknown_parameters(parameter_values.keys())
        self._raise_if_passed_nan(parameter_values)

        symbol_values = {key._name: value for key, value in parameter_values.items()}

        free_parameters = self.parameters - parameter_values.keys() - self._bound_values.keys()
        free_parameter_symbols = {p: s for p, s in self._parameter_symbols.items()
                                  if p in free_parameters or p in self._bound_values}
        if not free_parameters:
            bound_symbol_expr = self._symbol_expr(**symbol_values)
            bounds = None
        else:
            bound_symbol_expr = self._symbol_expr
            bounds = {}
            bounds.update(self._bound_values)
            bounds.update(parameter_values)

        return ParameterExpression(free_parameter_symbols, bound_symbol_expr, self._str_expr, bounds=bounds)

    def subs(self,
             parameter_map: Dict) -> 'ParameterExpression':
        """Returns a new Expression with replacement Parameters.

        Args:
            parameter_map: Mapping from Parameters in self to the ParameterExpression
                           instances with which they should be replaced.

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            A new expression with the specified parameters replaced.
        """

        inbound_parameters = {p
                              for replacement_expr in parameter_map.values()
                              for p in replacement_expr.parameters}

        self._raise_if_passed_unknown_parameters(parameter_map.keys())
        self._raise_if_parameter_names_conflict(inbound_parameters, parameter_map.keys())

        def _subs(**kwargs):
            new_kwargs = {}
            for parameter, expr in parameter_map.items():
                new_kwargs[parameter] = expr(**kwargs)

            return self._symbol_expr(**new_kwargs)

        # Include existing parameters in self not set to be replaced.
        new_parameter_symbols = parameter_map
        new_parameter_symbols.update({p: s
                                      for p, s in self._parameter_symbols.items()
                                      if p not in parameter_map})
        str_expr = str(self)
        for key, val in parameter_map.items():
            str_expr.replace(key._name, str(val))

        return ParameterExpression(new_parameter_symbols, _subs, str_expr, parameter_map=parameter_map)

    def _raise_if_passed_unknown_parameters(self, parameters):
        unknown_parameters = parameters - self.parameters
        if unknown_parameters:
            raise CircuitError('Cannot bind Parameters ({}) not present in '
                               'expression.'.format([str(p) for p in unknown_parameters]))

    def _raise_if_passed_nan(self, parameter_values):
        nan_parameter_values = {p: v for p, v in parameter_values.items()
                                if not isinstance(v, numbers.Number)}
        if nan_parameter_values:
            raise CircuitError('Expression cannot bind non-numeric values ({})'.format(
                nan_parameter_values))

    def _raise_if_parameter_names_conflict(self, inbound_parameters, outbound_parameters=None):
        if outbound_parameters is None:
            outbound_parameters = set()

        if self._names is None:
            self._names = {p.name: p for p in self._parameters}

        inbound_names = {p.name: p for p in inbound_parameters}
        outbound_names = {p.name: p for p in outbound_parameters}

        shared_names = (self._names.keys() - outbound_names.keys()) & inbound_names.keys()
        conflicting_names = {name for name in shared_names
                             if self._names[name] != inbound_names[name]}
        if conflicting_names:
            raise CircuitError('Name conflict applying operation for parameters: '
                               '{}'.format(conflicting_names))

    def _apply_operation(self, operation: Callable,
                         other: ParameterValueType,
                         str_op: str,
                         reflected: bool = False) -> 'ParameterExpression':
        """Base method implementing math operations between Parameters and
        either a constant or a second ParameterExpression.

        Args:
            operation: One of operator.{add,sub,mul,truediv}.
            other: The second argument to be used with self in operation.
            str_op: The sting representing the operator
            reflected: Optional - The default ordering is "self operator other".
                       If reflected is True, this is switched to "other operator self".
                       For use in e.g. __radd__, ...

        Raises:
            CircuitError:
                - If parameter_map contains Parameters outside those in self.
                - If the replacement Parameters in parameter_map would result in
                  a name conflict in the generated expression.

        Returns:
            A new expression describing the result of the operation.
        """
        self_expr = self._symbol_expr

        if isinstance(other, ParameterExpression):
            self._raise_if_parameter_names_conflict(other._parameter_symbols.keys())

            parameter_symbols = {**self._parameter_symbols, **other._parameter_symbols}
            other_expr = other._symbol_expr
        elif isinstance(other, numbers.Number) and numpy.isfinite(other):
            parameter_symbols = self._parameter_symbols.copy()
            other_expr = other
        else:
            return NotImplemented

        if reflected:
            str_op = '%s %s %s' % (str(other), str_op, str(self))

            def expr(**kwargs):
                self_value = self_expr(**kwargs)
                if callable(other_expr):
                    other_value = other_expr(**kwargs)
                else:
                    other_value = other_expr

                return operation(other_value, self_value)

        else:
            str_op = '%s %s %s' % (str(self), str_op, str(other))

            def expr(**kwargs):
                self_value = self_expr(**kwargs)
                if callable(other_expr):
                    other_value = other_expr(**kwargs)
                else:
                    other_value = other_expr
                return operation(self_value, other_value)

        return ParameterExpression(parameter_symbols, expr, str_op)

    def gradient(self, param) -> Union['ParameterExpression', float]:
        """Get the derivative of a parameter expression w.r.t. a specified parameter expression.

        Args:
            param (Parameter): Parameter w.r.t. which we want to take the derivative

        Returns:
            ParameterExpression representing the gradient of param_expr w.r.t. param
        """
        # Check if the parameter is contained in the parameter expression
        if param not in self._parameter_symbols.keys():
            # If it is not contained then return 0
            return 0.0

        # Compute the gradient of the parameter expression w.r.t. param
        import sympy as sy
        key = self._parameter_symbols[param]
        symbol = sy.Symbol(key)
        sympy_expr = sy.parse_expr(self._str_expr)
        # TODO enable nth derivative
        expr_grad = sy.Derivative(sympy_expr, symbol).doit()

        # generate the new dictionary of symbols
        # this needs to be done since in the derivative some symbols might disappear (e.g.
        # when deriving linear expression)
        parameter_symbols = {}
        for parameter, symbol in self._parameter_symbols.items():
            if symbol in expr_grad.free_symbols:
                parameter_symbols[parameter] = symbol
        # If the gradient corresponds to a parameter expression then return the new expression.
        if len(parameter_symbols) > 0:
            return ParameterExpression(parameter_symbols, expr_grad, 'deriv(%s)' % str(self))
        # If no free symbols left, return a float corresponding to the gradient.
        return float(expr_grad)

    def __add__(self, other):
        return self._apply_operation(operator.add, other, '+')

    def __radd__(self, other):
        return self._apply_operation(operator.add, other, '+', reflected=True)

    def __sub__(self, other):
        return self._apply_operation(operator.sub, other, '-')

    def __rsub__(self, other):
        return self._apply_operation(operator.sub, other, '-', reflected=True)

    def __mul__(self, other):
        return self._apply_operation(operator.mul, other, '*')

    def __neg__(self):
        return self._apply_operation(operator.mul, -1.0, '*')

    def __rmul__(self, other):
        return self._apply_operation(operator.mul, other, '*', reflected=True)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError('Division of a ParameterExpression by zero.')
        return self._apply_operation(operator.truediv, other, '/')

    def __rtruediv__(self, other):
        return self._apply_operation(operator.truediv, other, '/',
                                     reflected=True)

    def _call(self, ufunc, str_repr):
        return ParameterExpression(
            self._parameter_symbols,
            ufunc,
            str_repr
        )

    def sin(self):
        """Sine of a ParameterExpression"""

        def _sin(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.sin(value)
            else:
                return math.sin(value)

        return self._call(_sin, 'sin(%s)' % self._str_expr)

    def cos(self):
        """Cosine of a ParameterExpression"""

        def _cos(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.cos(value)
            else:
                return math.cos(value)

        return self._call(_cos, 'cos(%s)' % self._str_expr)

    def tan(self):
        """Tangent of a ParameterExpression"""

        def _tan(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.tan(value)
            else:
                return math.tan(value)

        return self._call(_tan, 'tan(%s)' % self._str_expr)

    def arcsin(self):
        """Arcsin of a ParameterExpression"""

        def _asin(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.asin(value)
            else:
                return math.asin(value)

        return self._call(_asin, 'asin(%s)' % self._str_expr)

    def arccos(self):
        """Arccos of a ParameterExpression"""

        def _acos(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.acos(value)
            else:
                return math.acos(value)

        return self._call(_acos, 'acos(%s)' % self._str_expr)

    def arctan(self):
        """Arctan of a ParameterExpression"""

        def _atan(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.atan(value)
            else:
                return math.atan(value)

        return self._call(_atan, 'atan(%s)' % self._str_expr)

    def exp(self):
        """Exponential of a ParameterExpression"""

        def _exp(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.exp(value)
            else:
                return math.exp(value)

        return self._call(_exp, 'exp(%s)' % self._str_expr)

    def log(self):
        """Logarithm of a ParameterExpression"""

        def _log(**kwargs):
            value = self._symbol_expr(**kwargs)
            if isinstance(value, complex):
                return cmath.log(value)
            else:
                return math.log(value)

        return self._call(_log, 'log(%s)' % self._str_expr)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, str(self))

    def __str__(self):
        return str(self._str_expr)

    def __float__(self):
        if self.parameters:
            raise TypeError('ParameterExpression with unbound parameters ({}) '
                            'cannot be cast to a float.'.format(self.parameters))
        return float(self._symbol_expr)

    def __complex__(self):
        if self.parameters:
            raise TypeError('ParameterExpression with unbound parameters ({}) '
                            'cannot be cast to a complex.'.format(self.parameters))
        return complex(self._symbol_expr)

    def __int__(self):
        if self.parameters:
            raise TypeError('ParameterExpression with unbound parameters ({}) '
                            'cannot be cast to an int.'.format(self.parameters))
        return int(self._symbol_expr)

    def __hash__(self):
        return hash((frozenset(self._parameter_symbols), self._symbol_expr))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __eq__(self, other):
        """Check if this parameter expression is equal to another parameter expression
           or a fixed value (only if this is a bound expression).
        Args:
            other (ParameterExpression or a number):
                Parameter expression or numeric constant used for comparison
        Returns:
            bool: result of the comparison
        """
        if isinstance(other, ParameterExpression):
            return (self.parameters == other.parameters
                    and self._str_expr == other._str_expr)
        elif isinstance(other, numbers.Number):
            return (len(self.parameters) == 0
                    and complex(self._symbol_expr) == other)
        return False
