import math
from collections import defaultdict
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
import sympy as sym
from typing import Sequence, Tuple, Dict, List

from sympy import symbols
from sparseSpACE import Function

# -----------------------------------------------------------------------------------------------------------------
# ---  Extrapolation Coefficients

class ExtrapolationVersion(Enum):
    ROMBERG_DEFAULT = 1
    ROMBERG_LINEAR = 2
    ROMBERG_SIMPSON = 3


# This class controls the coefficients for the extrapolation
class ExtrapolationCoefficients(ABC):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    @abstractmethod
    def get_coefficient(self, m: int, j: int) -> float:
        pass

    def get_step_width(self, j: int) -> float:
        return (self.b - self.a) / (2 ** j)

    def get_romberg_coefficient(self, m: int, j: int, exponent: int) -> float:
        coefficient = 1
        h_j = self.get_step_width(j)

        for i in range(m + 1):
            h_i = self.get_step_width(i)
            coefficient *= (h_i ** exponent) / (h_i ** exponent - h_j ** exponent) if i != j else 1

        return coefficient


class RombergLinearCoefficients(ExtrapolationCoefficients):
    def __init__(self, a: float, b: float):
        super(RombergLinearCoefficients, self).__init__(a, b)

    def get_coefficient(self, m: int, j: int) -> float:
        return self.get_romberg_coefficient(m, j, 1)


class RombergDefaultCoefficients(ExtrapolationCoefficients):
    def __init__(self, a: float, b: float):
        super(RombergDefaultCoefficients, self).__init__(a, b)

    def get_coefficient(self, m: int, j: int) -> float:
        return self.get_romberg_coefficient(m, j, 2)


class RombergSimpsonCoefficients(ExtrapolationCoefficients):
    def __init__(self, a: float, b: float):
        super(RombergSimpsonCoefficients, self).__init__(a, b)

    def get_coefficient(self, m: int, j: int) -> float:
        return self.get_romberg_coefficient(m, j, 3)


class ExtrapolationCoefficientsFactory:
    def __init__(self, extrapolation_version: ExtrapolationVersion):
        self.extrapolation_version = extrapolation_version

    def get(self, a: float, b: float, support_sequence: Sequence[Tuple[float, float]] =None) \
            -> ExtrapolationCoefficients:
        if self.extrapolation_version == ExtrapolationVersion.ROMBERG_DEFAULT:
            return RombergDefaultCoefficients(a, b)

        elif self.extrapolation_version == ExtrapolationVersion.ROMBERG_LINEAR:
            return RombergLinearCoefficients(a, b)

        elif self.extrapolation_version == ExtrapolationVersion.ROMBERG_SIMPSON:
            return RombergSimpsonCoefficients(a, b)

        else:
            raise RuntimeError("Wrong ExtrapolationVersion provided.")


# Returns the correct class for the computation of weights
class RombergWeightFactory:
    @staticmethod
    def get(a: float, b: float, version: ExtrapolationVersion) -> 'RombergWeights':
        if version == ExtrapolationVersion.ROMBERG_DEFAULT or version == ExtrapolationVersion.ROMBERG_LINEAR:
            return RombergTrapezoidalWeights(a, b, version)

        elif version == ExtrapolationVersion.ROMBERG_SIMPSON:
            return RombergSimpsonWeights(a, b)

        else:
            raise RuntimeError("Wrong ExtrapolationVersion provided.")


class RombergWeights(ABC):
    def __init__(self, a: float, b: float, version: ExtrapolationVersion):
        self.a = a
        self.b = b
        self.version = version
        self.extrapolation_factory = ExtrapolationCoefficientsFactory(version).get(a, b)

    @abstractmethod
    def get_boundary_point_weight(self, max_level: int) -> float:
        pass

    @abstractmethod
    def get_inner_point_weight(self, level: int, max_level: int) -> float:
        pass

    def get_extrapolation_coefficient(self, m: int, j: int) -> float:
        return self.extrapolation_factory.get_coefficient(m, j)

    def get_step_width(self, j: int) -> float:
        return self.extrapolation_factory.get_step_width(j)


class RombergTrapezoidalWeights(RombergWeights):
    def __init__(self, a: float, b: float, version: ExtrapolationVersion = ExtrapolationVersion.ROMBERG_DEFAULT):
        super(RombergTrapezoidalWeights, self).__init__(a, b, version)

    def get_boundary_point_weight(self, max_level: int) -> float:
        weight = 0

        for j in range(max_level + 1):
            coefficient = self.extrapolation_factory.get_coefficient(max_level, j)
            step_width = self.get_step_width(j)
            weight += (coefficient * step_width) / 2

        return weight

    def get_inner_point_weight(self, level: int, max_level: int) -> float:
        assert 1 <= level <= max_level

        weight = 0

        for j in range(level, max_level + 1):
            coefficient = self.extrapolation_factory.get_coefficient(max_level, j)
            step_width = self.get_step_width(j)
            weight += coefficient * step_width

        return weight


class RombergSimpsonWeights(RombergWeights):
    def __init__(self, a: float, b: float):
        super(RombergSimpsonWeights, self).__init__(a, b, ExtrapolationVersion.ROMBERG_SIMPSON)

    def get_boundary_point_weight(self, max_level: int) -> float:
        weight = 0

        for j in range(max_level + 1):
            coefficient = self.extrapolation_factory.get_coefficient(max_level, j)
            step_width = self.get_step_width(j)
            weight += (coefficient * step_width)

        return weight / 3

    def get_inner_point_weight(self, level: int, max_level: int) -> float:
        assert 1 <= level <= max_level
        factory = self.extrapolation_factory

        weight = ((factory.get_coefficient(max_level, level) * 4) / 3) * self.get_step_width(level)

        if level + 1 > max_level:
            return weight

        for j in range(level + 1, max_level + 1):
            coefficient = factory.get_coefficient(max_level, j)
            step_width = self.get_step_width(j)
            weight += ((coefficient * step_width) * 2) / 3

        return weight


# -----------------------------------------------------------------------------------------------------------------
# ---  Binary Tree Grid

# Singleton wrapper is defined below
class GridBinaryTree:
    class __GridBinaryTree:
        def __init__(self, use_caching: bool = False, print_debug: bool = False):
            self.grid = None
            self.grid_levels = None
            self.a = None
            self.b = None

            # Root node has level 1, since boundary points are note store in the tree
            self.root_node = None
            self.active_node_queue = []

            # Stores mapping of grid and its (full) tree grid
            self.tree_dict = {}

            self.use_caching = use_caching
            self.print_debug = print_debug

        def __init_attributes(self, grid: Sequence[float], grid_levels: Sequence[int]) -> None:
            self.grid = grid
            self.grid_levels = grid_levels
            self.a = grid[0]
            self.b = grid[-1]

            self.root_node = None
            self.active_node_queue = []
            self.full_tree_dict = {}

        # Given a grid level array, this method initializes a appropriate binary tree structure
        def init_tree(self, grid: Sequence[float], grid_levels: Sequence[int]) -> None:
            self.__init_attributes(grid, grid_levels)

            # Remove boundary levels
            assert grid_levels[0] == 0 and grid_levels[-1] == 0

            # build tree based on inner points (neglect index 0 and -1)
            self.__init_tree_rec(None, grid, grid_levels, 1, len(grid_levels) - 2)

            assert self.root_node is not None

        def __init_tree_rec(self, node: 'Node', grid: Sequence[float], grid_levels: Sequence[int],
                            start_index: int, stop_index: int,
                            inside_left_subtree=None) -> None:
            # Find index of minimal element in active interval
            current_grid_level_slice = grid_levels[start_index:stop_index + 1]
            if start_index < stop_index:
                split_index = start_index + current_grid_level_slice.index(min(current_grid_level_slice))
            else:
                split_index = start_index

            # Termination criterion
            if stop_index < start_index or split_index > stop_index or split_index < start_index:
                return
            # Root node
            elif node is None:
                self.root_node = GridBinaryTree.__GridBinaryTree.Node(grid[split_index], print_debug=self.print_debug)
                self.__init_tree_rec(self.root_node, grid, grid_levels,
                                     start_index, split_index - 1, inside_left_subtree=True)
                self.__init_tree_rec(self.root_node, grid, grid_levels,
                                     split_index + 1, stop_index, inside_left_subtree=False)
            # Other nodes
            else:
                if inside_left_subtree:
                    node.set_left_child(GridBinaryTree.__GridBinaryTree.Node(grid[split_index], parent=node))
                    node = node.get_left_child()
                else:
                    node.set_right_child(GridBinaryTree.__GridBinaryTree.Node(grid[split_index], parent=node))
                    node = node.get_right_child()

                self.__init_tree_rec(node, grid, grid_levels,
                                     start_index, split_index - 1, inside_left_subtree=True)
                self.__init_tree_rec(node, grid, grid_levels,
                                     split_index + 1, stop_index, inside_left_subtree=False)

        # Expand tree until it is a full binary tree
        def force_full_tree_invariant(self) -> None:
            if self.root_node is None:
                return

            if self.use_caching and self.__is_cached(self.grid):
                return

            nodes = self.root_node.get_nodes_using_dfs_in_order()

            for node in nodes:
                if node.has_only_one_child():
                    if node.has_left_child():
                        point = node.point + (node.point - node.left_child.point)
                        right_child = GridBinaryTree.__GridBinaryTree.Node(point)
                        node.set_right_child(right_child)
                    elif node.has_right_child():
                        point = node.point - (node.right_child.point - node.point)
                        left_child = GridBinaryTree.__GridBinaryTree.Node(point)
                        node.set_left_child(left_child)
                    else:
                        raise Exception("This shouldn't happen in a binary tree")

            if self.use_caching:
                self.__cache_grid()

        # Method for numerical experiments and testing
        # This method initialises a perfect binary tee (each node has two children, all leafs are on max_level)
        def init_perfect_tree_with_max_level(self, a: float, b: float, max_level: int) -> None:
            queue = []

            self.a = a
            self.b = b

            root_point = (b - a) / 2
            self.root_node = GridBinaryTree.__GridBinaryTree.Node(root_point)
            queue.append(self.root_node)

            # Boundary points are already created before
            for i in range(1, max_level):
                queue = self.__increment_level_in_each_subtree(queue)

            assert self.root_node is not None

        # This method creates two new children for each leaf node
        #   => level of each subtree is increased by one
        # Note: This works for each tree type (default, full, perfect)
        # Important: non-leaf nodes (e.g. nodes with one children) are not considered!!!
        def increment_level_in_each_subtree(self) -> None:
            leafs = self.__get_leafs()

            self.__increment_level_in_each_subtree(leafs)

        def __increment_level_in_each_subtree(self, node_queue: Sequence['Node'] = []) -> Sequence['Node']:
            new_queue = []
            H = self.b - self.a

            while len(node_queue) > 0:
                node = node_queue.pop(0)

                node.set_left_child(GridBinaryTree.__GridBinaryTree.Node(
                    node.point - H / (2 ** (node.level + 1))
                ))

                node.set_right_child(GridBinaryTree.__GridBinaryTree.Node(
                    node.point + H / (2 ** (node.level + 1))
                ))

                new_queue.append(node.get_left_child())
                new_queue.append(node.get_right_child())

            return new_queue

        # Return grid with boundary points
        def get_grid(self) -> Sequence[float]:
            if not self.use_caching:
                return self.__get_grid()

            dict_key = self.__get_dict_key(self.grid)

            if self.__is_cached(self.grid):
                return self.__get_grid_from_cache(dict_key)

            self.__cache_grid()

            return self.__get_grid_from_cache(dict_key)

        def __get_grid(self) -> Sequence[float]:
            return [self.a] + self.root_node.get_grid() + [self.b]

        # Return levels, with boundaries
        def get_grid_levels(self) -> Sequence[int]:
            if not self.use_caching:
                return self.__get_grid_levels()

            dict_key = self.__get_dict_key(self.grid)

            if self.__is_cached(self.grid):
                return self.__get_grid_levels_from_cache(dict_key)

            self.__cache_grid()
            return self.__get_grid_levels_from_cache(dict_key)

        def __get_grid_levels(self) -> Sequence[int]:
            return [0] + self.root_node.get_grid_levels() + [0]

        def __get_leafs(self) -> Sequence['Node']:
            if self.root_node is None:
                return []

            nodes = self.root_node.get_nodes_using_dfs_in_order()

            leafs = []
            for node in nodes:
                if node.is_leaf():
                    leafs.append(node)

            return leafs

        # Caching
        def __is_cached(self, grid) -> bool:
            dict_key = self.__get_dict_key(grid)

            return dict_key in self.full_tree_dict

        def __cache_grid(self) -> None:
            dict_key = self.__get_dict_key(self.grid)
            self.full_tree_dict[dict_key] = (self.__get_grid(), self.__get_grid_levels())

        def __get_grid_from_cache(self, dict_key: Sequence[float]) -> Sequence[float]:
            grid, _ = self.full_tree_dict[dict_key]

            return grid

        def __get_grid_levels_from_cache(self, dict_key: Tuple[float]) -> Sequence[int]:
            _, grid_levels = self.full_tree_dict[dict_key]

            return grid_levels

        @staticmethod
        def __get_dict_key(grid: Sequence[float]) -> Tuple[float, ...]:
            return tuple(grid)

        class Node:
            def __init__(self, point: float, left_child: 'Node' = None, right_child: 'Node' = None,
                         parent: 'Node' = None, start_level: int = 1,
                         print_debug: bool = False):
                self.point = point
                self.left_child = left_child
                self.right_child = right_child
                self.parent = parent
                self.level = start_level if parent is None else parent.level + 1

                self.print_debug = print_debug

            def get_nodes_using_dfs_in_order(self, max_level: int = None) -> Sequence['Node']:
                return self.__get_nodes_using_dfs_in_order_rec(self, max_level)

            def __get_nodes_using_dfs_in_order_rec(self, node: 'Node', max_level: int = None) -> List['node']:
                if node is None or (max_level is not None and node.level > max_level):
                    return []

                left_subtree = self.__get_nodes_using_dfs_in_order_rec(node.get_left_child(), max_level)
                right_subtree = self.__get_nodes_using_dfs_in_order_rec(node.get_right_child(), max_level)

                return left_subtree + [node] + right_subtree

            # Returns the grid of current subtree
            def get_grid(self, max_level: int = None) -> Sequence[float]:
                return list(map(lambda node: node.point, self.get_nodes_using_dfs_in_order(max_level)))

            # Returns the levels of the current subtree
            def get_grid_levels(self, max_level: int = None) -> Sequence[int]:
                return list(map(lambda node: node.level, self.get_nodes_using_dfs_in_order(max_level)))

            # Setter methods for children nodes
            def set_left_child(self, left_child: 'Node') -> None:
                self.left_child = left_child
                left_child.level = self.level + 1

            def set_right_child(self, right_child: 'Node') -> None:
                self.right_child = right_child
                right_child.level = self.level + 1

            # Getter methods
            def get_left_child(self) -> 'Node':
                return self.left_child

            def get_right_child(self) -> 'Node':
                return self.right_child

            def has_left_child(self) -> bool:
                return self.get_left_child() is not None

            def has_right_child(self) -> bool:
                return self.get_right_child() is not None

            def has_both_children(self) -> bool:
                return self.has_left_child() and self.has_right_child()

            def has_only_one_child(self) -> bool:
                return (self.has_left_child() and (not self.has_right_child())) \
                       or (self.has_right_child() and (not self.has_left_child()))

            def is_leaf(self) -> bool:
                return (not self.has_left_child()) and (not self.has_right_child())

            def is_root_node(self) -> bool:
                return self.parent is None

            def has_parent(self) -> bool:
                return not self.is_root_node()

            def is_left_child(self) -> bool:
                return self.parent.get_left_child() is self

            def is_right_child(self) -> bool:
                return self.parent.get_right_child() is self

            @staticmethod
            def node_to_string(node) -> str:
                return "Node: Point {} of level {})".format(node.point, node.level)

    instance = None

    def __init__(self, print_debug: bool = False):
        if not GridBinaryTree.instance:
            GridBinaryTree.instance = GridBinaryTree.__GridBinaryTree(print_debug=print_debug)
        else:
            GridBinaryTree.instance.print_debug = print_debug

    def __getattr__(self, name: str):
        return getattr(self.instance, name)

    # Interface methods
    def init_tree(self, grid: Sequence[float], grid_levels: Sequence[int]) -> None:
        return self.instance.init_tree(grid, grid_levels)

    def force_full_tree_invariant(self) -> None:
        self.instance.force_full_tree_invariant()

    def init_perfect_tree_with_max_level(self, a: float, b: float, max_level: int) -> None:
        self.instance.init_perfect_tree_with_max_level(a, b, max_level)

    def increment_level_in_each_subtree(self) -> None:
        self.instance.increment_level_in_each_subtree()

    def get_grid(self) -> Sequence[float]:
        return self.instance.get_grid()

    def get_grid_levels(self) -> Sequence[int]:
        return self.instance.get_grid_levels()


# -----------------------------------------------------------------------------------------------------------------
# ---  Extrapolation Grid

class GridVersion(Enum):
    DEFAULT = 1  # No interpolation are inserted
    INTERPOLATE_SUB_GRIDS = 2  # Important missing grid points are inserted as interpolation points
    INTERPOLATE_FULL_GRID = 3  # All missing points needed for a full grid are inserted as interpolation points


class SliceGrouping(Enum):
    """
    This enum specifies the available options for slice grouping.
    """
    UNIT = 1  # Only unit slices (no grouping)
    GROUPED = 2  # Group slices if size 2^k, split into unit containers otherwise
    GROUPED_OPTIMIZED = 3  # Group slices until container size is the max possible 2^k


class SliceContainerVersion(Enum):
    """
    This enum specifies the available slice types.
    """
    # Only extrapolation
    ROMBERG_DEFAULT = 1  # Default Romberg extrapolation in non-unit containers
    SIMPSON_ROMBERG = 2  # Romberg extrapolation with simpson rule as base quadrature rule

    # Interpolation before extrapolation
    LAGRANGE_ROMBERG_OLD = 3  # Interpolated grid points are interpolated using Lagrange

    # Interpolation before extrapolation using sparseSpACE grids
    LAGRANGE_ROMBERG = 4  # Interpolated grid points are interpolated using Lagrange
    HIERARCHICAL_LAGRANGE_ROMBERG = 5 # Interpolated grid points are interpolated using hierarchical Lagrange
    BSPLINE_ROMBERG = 6 # Interpolated grid points are interpolated using BSplines


class SliceVersion(Enum):
    """
    This enum specifies the available slice types.
    """
    ROMBERG_DEFAULT = 1  # Sliced Romberg with default Romberg extrapolation
    TRAPEZOID = 2  # Default trapezoidal rule without extrapolation
    ROMBERG_DEFAULT_CONST_SUBTRACTION = 3  # Sliced Romberg with default Romberg extrapolation and constant subtraction


class ExtrapolationGrid:
    """
    This is the main extrapolation class. It initializes the containers and slices.
    Collects their (extrapolated) weights and returns the weights for each grid point.

    :param slice_grouping: Specifies which type of grouping is used for the slices (e.g. UNIT, or GROUPED).
    :param slice_version: This parameters specifies the type of extrapolation within a slice.
    :param container_version: Specifies the extrapolation type within a container.
    :param force_balanced_refinement_tree: If enabled, the provided grid will be extended to a full binary grid
            (each point, except the boundary points, has two children).
    :param print_debug: Print output to console.
    """

    def __init__(self,
                 grid_version: GridVersion = GridVersion.DEFAULT,
                 container_version: SliceContainerVersion = SliceContainerVersion.ROMBERG_DEFAULT,
                 slice_grouping: SliceGrouping = SliceGrouping.UNIT,
                 slice_version: SliceVersion = SliceVersion.ROMBERG_DEFAULT,
                 force_balanced_refinement_tree: bool = False,
                 print_debug: bool = False):
        self.print_debug = print_debug

        self.a = None
        self.b = None
        self.grid = None
        self.grid_levels = None
        self.slice_containers = None
        self.weights = None
        self.function = None
        self.integral_approximation = None

        # Different grid versions
        self.grid_version = grid_version
        self.container_version = container_version
        self.slice_grouping = slice_grouping
        self.slice_version = slice_version
        self.force_balanced_refinement_tree = force_balanced_refinement_tree

        # Factories
        self.slice_factory = ExtrapolationGridSliceFactory(self.slice_version)
        self.container_factory = ExtrapolationGridSliceContainerFactory(self.container_version)

        # Should interpolation grid points be inserted?
        if self.grid_version == GridVersion.INTERPOLATE_FULL_GRID:
            self.max_interpolation_step_width_delta = np.infty
        elif self.grid_version == GridVersion.INTERPOLATE_SUB_GRIDS:
            self.max_interpolation_step_width_delta = 2 ** 1
        else:
            self.max_interpolation_step_width_delta = 0

        # TODO Assert right combinations of grid_version and container_version

    def interpolation_is_enabled(self) -> bool:
        return self.grid_version == GridVersion.INTERPOLATE_SUB_GRIDS or \
               self.grid_version == GridVersion.INTERPOLATE_FULL_GRID

    # Integration
    def integrate(self, function: Function = None) -> float:
        """
        This method integrates a given function over a grid that has been specified before.

        :param function: function to be integrated.
        :returns: the approximated integral.
        """

        assert self.grid is not None
        assert self.grid_levels is not None
        assert self.slice_containers is not None

        assert len(self.grid) >= 2

        # Update function and anti derivative
        if function is not None:
            self.set_function(function)

        # Pass the updated function to containers!
        self.update_function_in_containers()

        assert self.function is not None

        # Compute scalar product of weights and function values
        if self.weights is None:
            self.update_weights()

        assert len(self.weights) == len(self.grid), "Weight count and grid point count are different"

        value = 0
        for i in range(len(self.grid)):
            value += self.weights[i] * self.function.eval([self.grid[i]])

        self.integral_approximation = value

        return value

    def set_grid(self, grid: Sequence[float], grid_levels: Sequence[float]) -> None:
        """
        This method updates the grid and initializes the new slices.

        :param grid: The new grid points as array.
        :param grid_levels: The corresponding grid levels as array.
        :returns: void.
        """

        assert len(grid) == len(grid_levels) and len(grid) >= 2, "Wrong grid or grid levels provided."

        self.grid = grid
        self.grid_levels = grid_levels
        self.weights = None

        if self.force_balanced_refinement_tree:
            print("Forcing the grid to its closest full binary tree")
            tree = GridBinaryTree(print_debug=self.print_debug)
            tree.init_tree(grid, grid_levels)
            tree.force_full_tree_invariant()
            self.grid = tree.get_grid()
            self.grid_levels = tree.get_grid_levels()

        # Set boundary points
        self.a = self.grid[0]
        self.b = self.grid[-1]

        if self.print_debug:
            print("The grid is set to  \n   {}, with grid levels \n   {}".format(grid, grid_levels))

        # Initialize the new slices
        self.__init_grid_slices()

        if self.print_debug:
            print("The grid slices have been initialized \n")

    def __init_grid_slices(self) -> None:
        """
        This method partitions the integration area into slices (based on the provided grid).
        The slices are grouped into containers.

        :returns: void.
        """

        # Reset slice containers
        self.slice_containers = []

        # This buffer stores the step width of the previous grid slice
        step_width_buffer = None

        for i in range(len(self.grid) - 1):
            # Boundaries of this slice
            start_point = self.grid[i]
            start_level = self.grid_levels[i]
            end_point = self.grid[i + 1]
            end_level = self.grid_levels[i + 1]

            # Assert that the step width is h_k = 1/(2^k)
            step_width = end_point - start_point
            max_level = max(start_level, end_level)
            message = "The step width {} between x_{} and x_{} is not of the form 1/(2^k)".format(step_width, i, i + 1)
            assert step_width == self.get_step_width(max_level), message

            support_sequence = self.compute_support_sequence(i, i + 1)

            grid_slice = self.slice_factory.get_grid_slice(
                [start_point, end_point],
                [start_level, end_level],
                support_sequence,
                function=self.function
            )

            # Initialize containers based on the inherited class
            self.initialize_containers_with_slices(step_width, step_width_buffer, grid_slice)

            # Update buffer
            step_width_buffer = step_width

        # Enable possibility for inheriting classes to do a postprocessing step
        self.grid_init_post_processing()

    def compute_support_sequence(self, final_slice_start_index: int, final_slice_stop_index: int) \
            -> List[Tuple[float, float]]:
        """
        This method computes the sequence of support points for the sliced trapezoid.

        :param final_slice_start_index: index of the left boundary in the grid array.
        :param final_slice_stop_index: index of the right boundary in the grid array.
        :returns: support sequence, e.g. [(0, 1), (0, 0.5), (0.25, 0.5)].
        """

        # Init sequence with indices for level 0 (whole integration domain)
        sequence = [(0, len(self.grid) - 1)] \
                   + self.__compute_support_sequence_rec(0, len(self.grid) - 1,
                                                         final_slice_start_index, final_slice_stop_index)

        return [(self.grid[element[0]], self.grid[element[1]]) for element in sequence]

    def __compute_support_sequence_rec(self, start_index: int, stop_index: int,
                                       final_slice_start_index: int, final_slice_stop_index: int) \
            -> List[Tuple[int, int]]:
        """
        This method recursively computes the support sequence.

        :param start_index: current start index in array.
        :param stop_index: current stop index in array.
        :param final_slice_start_index: index of the left boundary in the grid array.
        :param final_slice_stop_index: index of the right boundary in the grid array.
        :returns: support sequence except the first support tuple, e.g. [(0, 0.5), (0.25, 0.5)].
        """

        if start_index >= stop_index:
            return []

        # Slice of the grid levels without the current level
        grid_levels_slice = self.grid_levels[(start_index + 1):stop_index]

        if len(grid_levels_slice) == 0:
            return []

        # Start or stop index of the next slice in the sequence
        new_boundary_index = (start_index + 1) + grid_levels_slice.index(min(grid_levels_slice))

        # Determine the boundary indices of the new slice in the sequence
        if new_boundary_index <= final_slice_start_index:
            start_index = new_boundary_index
            # stop_index does not change
        else:
            # start_index does not_change
            stop_index = new_boundary_index

        return [(start_index, stop_index)] + self.__compute_support_sequence_rec(start_index, stop_index,
                                                                                 final_slice_start_index,
                                                                                 final_slice_stop_index)

    def initialize_containers_with_slices(self, step_width: float, step_width_buffer: float, grid_slice) -> None:
        """
        Group  >= 1 slices into a container, which spans a partial full grid

        :param step_width: current step width
        :param step_width_buffer: previous step width
        :param grid_slice: current slice

        :return: void
        """

        if self.interpolation_is_enabled():
            self.__initialize_containers_with_interpolated_slices(step_width, step_width_buffer, grid_slice)
        else:
            self.__initialize_default_containers(step_width, step_width_buffer, grid_slice)

    def __initialize_containers_with_interpolated_slices(self, step_width: float, step_width_buffer: float, grid_slice)\
            -> None:
        """
        Group  >= 1 slices into a container, which spans a partial full grid

        :param step_width: current step width
        :param step_width_buffer: previous step width
        :param grid_slice: current slice

        :return: void
        """
        is_first_container = step_width_buffer is None

        # Create a new container...
        #    ... if this is the first container
        #    ... the step widths are to different (then there are to many points to interpolate)
        if is_first_container:
            container = self.container_factory.get_grid_slice_container(function=self.function)
            container.append_slice(grid_slice)
            self.slice_containers.append(container)
            return

        max_delta = self.max_interpolation_step_width_delta
        container = self.slice_containers[-1]
        min_step_width = container.minimal_step_width
        correct_step_width_delta = min_step_width/max_delta <= step_width <= min_step_width*max_delta

        # Append to previous container
        if step_width_buffer == step_width or correct_step_width_delta:
            container.append_slice(grid_slice)
            return

        # Create new container
        container = self.container_factory.get_grid_slice_container(function=self.function)
        container.append_slice(grid_slice)
        self.slice_containers.append(container)

    def __initialize_default_containers(self, step_width: float, step_width_buffer: float, grid_slice) -> None:
        """
        Group  >= 1 slices into a container, which spans a partial full grid

        :param step_width: current step width
        :param step_width_buffer: previous step width
        :param grid_slice: current slice

        :return: void
        """

        # Create a new container...
        #    ... if this is the first iteration or
        #    ... if the step width changes or
        #    ... the version of this grid is defined to be unit slices (then each container holds only one slice)
        if (step_width_buffer is None) \
                or (step_width != step_width_buffer) \
                or (self.slice_grouping == SliceGrouping.UNIT):
            container = self.container_factory.get_grid_slice_container(function=self.function)
            container.append_slice(grid_slice)

            self.slice_containers.append(container)
            return

        # Append to previous container
        if step_width == step_width_buffer:
            container = self.slice_containers[-1]
            container.append_slice(grid_slice)

    def grid_init_post_processing(self) -> None:
        """
        This method executes post processing steps

        :return: void
        """
        # Replace slices where one point ist missing with two interpolating slices
        if self.interpolation_is_enabled():
            self.insert_interpolating_slices()

        self.adjust_containers()

        # Add information about all containers to the left & right for each container
        self.set_adjacent_containers_for_all_slice_containers()

        # Add information about all slices to the left & right for each slice in each container
        self.set_adjacent_slices_for_all_slices()

    def insert_interpolating_slices(self) -> None:
        """
        This method computes missing grid points and updates the container accordingly

        :return: void
        """
        assert self.interpolation_is_enabled()

        for container in self.slice_containers:
            container.insert_interpolating_slices()

    def adjust_containers(self) -> None:
        """
        Split containers if necessary into unit slices. Each container should contain 2^k slices for k > 0.
        It updates the attributes accordingly.

        :returns: void.
        """

        new_containers = []

        for i, container in enumerate(self.slice_containers):
            size = container.size()

            # Split container if size != 2^k for k >= 0 (<=> log_2(size) isn't an integer)
            #   Split until size is the next closest power of 2
            if not (math.log(size, 2)).is_integer():
                # Optimized container splitting: Split multiple containers with power 2
                if self.slice_grouping == SliceGrouping.GROUPED_OPTIMIZED:
                    split_containers = container.split_into_containers_with_power_two_sizes()
                    new_containers.extend(split_containers)

                # Default grouping: Create new container for each slice
                else:
                    for slice in container.slices:
                        container = self.container_factory.get_grid_slice_container(function=self.function)
                        container.append_slice(slice)
                        self.assert_container_size(container)

                        new_containers.append(container)
            else:
                new_containers.append(container)
                self.assert_container_size(container)

        self.slice_containers = new_containers

    def set_adjacent_containers_for_all_slice_containers(self) -> None:
        """
        This method determines for each slice container all (indirect) adjacent containers to the left and right
        of the container.

        It reverses the list, since the containers should be provided from closest first to farthest away.

        :return: void
        """

        containers = self.slice_containers

        for i, container in enumerate(self.slice_containers):
            containers_to_left = containers[0:i]
            containers_to_right = containers[(i + 1):]

            container.set_adjacent_containers(left=list(reversed(containers_to_left)),
                                              right=containers_to_right)

    def set_adjacent_slices_for_all_slices(self) -> None:
        """
        This method initializes the adjacent slices for all slices in each container.

        :return: void
        """

        for container in self.slice_containers:
            container.initialize_adjacent_slices_for_all_slices()

    @staticmethod
    def assert_container_size(container) -> None:
        """
        This method assures that the container has a size of 2^k (Assert).

        :param container: Extrapolation container with slices.
        :returns: void.
        """

        assert (math.log(container.size(), 2)).is_integer()

    def update_weights(self):
        """
        This method computes the weights for the current grid and overrides the old weights.

        :returns: void.
        """
        self.weights = self.get_weights()

    def get_weights(self) -> Sequence[float]:
        """
        Get extrapolated weights as array (sorted by grid points in increasing order).

        :returns: weights as array.
        """
        grid_weights = []
        weight_dictionary = self.__get_final_weights_from_all_slices()

        for _, weights in sorted(weight_dictionary.items()):
            grid_weights.append(sum(weights))

        return grid_weights

    def __get_final_weights_from_all_slices(self) -> Dict[float, Sequence[float]]:
        """
        This method iterates over all containers and collects their weights in a single dictionary.

        :returns: all weights of the current grid as dictionary.
        """
        weight_dictionary = defaultdict(list)

        # Iterate over all grid slice containers and collect their weights into one dictionary
        for container in self.slice_containers:
            weight_dictionary_for_slice = container.get_final_weights()

            # Display error of this slice before and after extrapolation
            if self.print_debug and container.function is not None:
                print()
                container.print_error_evolution()
                print()

            for grid_point, weights in weight_dictionary_for_slice.items():
                weight_dictionary[grid_point].extend(weights)

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Setter

    def set_function(self, function: Function) -> None:
        self.function = function

        self.update_function_in_containers()

    def update_function_in_containers(self) -> None:
        # Update function in slices
        if self.slice_containers is not None:
            for container in self.slice_containers:
                container.set_function(self.function)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Getter

    def get_grid(self) -> Sequence[float]:
        return self.grid

    def get_grid_levels(self) -> Sequence[int]:
        return self.grid_levels

    def get_error(self) -> float:
        actual_result = self.integral_approximation

        if actual_result is None:
            actual_result = self.integrate()

        return actual_result - self.function.getAnalyticSolutionIntegral([self.a], [self.b])

    def get_absolute_error(self) -> float:
        return abs(self.get_error())

    def get_step_width(self, level: int) -> float:
        return (self.b - self.a) / (2 ** level)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Plot

    def plot_slices_with_function(self) -> None:
        """
        This method plots the function as well as the grid slices/containers.

        :returns: void.
        """
        assert self.function is not None
        grid = self.get_grid()

        x = np.array(grid)
        y = np.array([self.function.eval(xi) for xi in x])

        # X and Y values for plotting y=f(x)
        X = np.linspace(self.a, self.b, 100)
        Y = np.array([self.function.eval(xi) for xi in X])
        plt.plot(X, Y)

        for i in range(len(x) - 1):
            xs = [x[i], x[i], x[i + 1], x[i + 1]]
            plt.plot([x[i]], [self.function.eval(x[i])], marker='o', markersize=3, color="red")
            ys = [0, self.function.eval(x[i]), self.function.eval(x[i + 1]), 0]
            plt.fill(xs, ys, 'b', edgecolor='b', alpha=0.2)

        plt.plot([x[-1]], [self.function.eval(x[-1])], marker='o', markersize=3, color="red")
        # plt.xticks(list(range(len(grid))), grid)

        plt.text(x[-1] + 0.15, y[-1], "f")
        plt.title("RombergGrid Slices")

        plt.show()

    def plot_support_sequence_for_each_slice(self, filename: str = None) -> None:
        """
        This method plots the support sequence for each slice slices.

        :returns: void.
        """
        assert self.function is not None
        grid = self.get_grid()

        x = np.array(grid)
        y = np.array([self.function.eval(xi) for xi in x])

        # X and Y values for plotting y=f(x)
        X = np.linspace(self.a, self.b, 100)
        Y = np.array([self.function.eval(xi) for xi in X])

        i = 0
        for container in self.slice_containers:
            for slice in container.slices:
                for j, (left_supp, right_supp) in enumerate(slice.support_sequence):
                    plt.plot(X, Y)

                    y_left_supp = self.function.eval(left_supp)
                    y_right_supp = self.function.eval(right_supp)

                    p = interp1d([left_supp, right_supp], [y_left_supp, y_right_supp],
                                 kind="linear", fill_value="extrapolate")

                    y_left_supp = p(left_supp) if p(left_supp) > y_left_supp else y_left_supp
                    y_right_supp = p(right_supp) if p(right_supp) > y_right_supp else y_right_supp

                    # Plot interpolation through support points
                    plt.plot([left_supp, right_supp], [y_left_supp, y_right_supp], '-', color="red")
                    plt.text(left_supp + 0.05, y_left_supp + 0.5, "p", color="red")

                    # Grid lines
                    for point in grid:
                        if point == slice.left_point or point == slice.right_point:
                            y_p = p(point)
                        else:
                            y_p = self.function.eval(point)

                        plt.plot([point, point], [0, y_p], '--', color="grey")

                    xs = [slice.left_point, slice.left_point, slice.right_point, slice.right_point]
                    ys = [0, p(slice.left_point), p(slice.right_point), 0]
                    plt.fill(xs, ys, 'b', edgecolor='b', color="pink", alpha=0.6)
                    plt.text(x[-1] - 0.05, y[-1] - 0.7, "f", color="blue")
                    plt.title("Slice [{}, {}] with support ({}, {})".format(slice.left_point, slice.right_point,
                                                                            left_supp, right_supp))
                    plt.xticks(self.grid, self.grid)

                    if filename is not None:
                        plt.savefig("{}_slice_{}_support_{}".format(filename, i, j), bbox_inches='tight', dpi=300)

                    plt.show()
                i += 1

    def plot_slice_refinement_levels(self, filename: str = None) -> None:
        """
        This method plots the support sequence for each slice slices.

        :returns: void.
        """
        assert self.function is not None
        grid = self.get_grid()

        x = np.array(grid)
        y = np.array([self.function.eval(xi) for xi in x])

        # X and Y values for plotting y=f(x)
        X = np.linspace(self.a, self.b, 100)
        Y = np.array([self.function.eval(xi) for xi in X])

        for i in range(max(self.grid_levels) + 1):
            plt.plot(X, Y)

            for container in self.slice_containers:
                for slice in container.slices:
                    left_supp, right_supp = slice.support_sequence[min(i, slice.max_level)]
                    y_left_supp = self.function.eval(left_supp)
                    y_right_supp = self.function.eval(right_supp)

                    p = interp1d([left_supp, right_supp], [y_left_supp, y_right_supp],
                                 kind="linear", fill_value="extrapolate")

                    # Plot interpolation through support points
                    plt.plot([left_supp, right_supp], [y_left_supp, y_right_supp], '-', color="red")

                    # Grid lines
                    point = slice.left_point
                    y_p = p(point)
                    plt.plot([point, point], [0, y_p], '--', color="grey")

                    plt.text(x[-1] - 0.05, y[-1] - 0.7, "f", color="blue")
                    plt.text(x[0] + 0.05, y[0] + 0.3, "p", color="red")

                    # Plot interpolation point
                    plt.plot([left_supp], [y_left_supp], marker='o', markersize=5, color="green")

                    plt.title("Level {}".format(i))

            # Plot interpolation point
            plt.plot([grid[-1], grid[-1]], [0, self.function.eval(grid[-1])], '--', color="grey")
            plt.plot([grid[-1]], [self.function.eval(grid[-1])], marker='o', markersize=5, color="green")

            plt.xticks(self.grid, self.grid)

            if filename is not None:
                plt.savefig("{}_slice_refinement_level_{}".format(filename, i), bbox_inches='tight', dpi=300)

            plt.show()

    def plot_grid_with_containers(self, filename: str = None, highlight_containers: bool = True) -> None:
        markersize = 20
        fontsize = 60

        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 5))

        # Initialize total grid
        containers = self.slice_containers
        total_grid = []  # Should contain all points (interpolated ones included)
        total_grid_levels = []
        interpolated_indicator = []

        for container in containers:
            for slice in container.slices:
                total_grid.append(slice.left_point)
                total_grid_levels.append(slice.levels[0])
                interpolated_indicator.append(slice.is_left_point_interpolated())

            # Last point in container
            last_slice = container.slices[-1]
            total_grid.append(last_slice.right_point)
            total_grid_levels.append(last_slice.levels[1])
            interpolated_indicator.append(last_slice.is_right_point_interpolated())

        # Plot grid
        axis = ax
        starts = total_grid[0:len(total_grid) - 1]
        starts_levels = total_grid_levels[0:len(total_grid_levels) - 1]
        ends = total_grid[1:len(total_grid)]
        ends_levels = total_grid_levels[1:len(total_grid_levels)]

        # Container background color cycle
        container_color_cycle = ["blue", "red"]
        hatch_cycle = ["-", "/"]

        # Plot containers
        for i, container in enumerate(containers):
            axis.add_patch(
                patches.Rectangle(
                    (container.left_point, -0.1),
                    container.right_point - container.left_point,
                    0.2, linestyle='-',
                    linewidth=2,
                    fill=highlight_containers,
                    color=container_color_cycle[i % 2],
                    alpha=0.4,
                    hatch=hatch_cycle[i % 2] if highlight_containers else None
                )
            )

        # Plot grid points
        for i in range(len(starts)):
            linestyle = '-'

            if interpolated_indicator[i]:
                linestyle = '--'

            plt.plot([starts[i], starts[i]], [-0.2, 0.2], linestyle, color="black", linewidth=3)

            axis.text(starts[i] + 0.015, 0.01, str(starts_levels[i]),
                      fontsize=fontsize - 10, ha='center', color="blue")

        # Last vertical bar
        linestyle = '-'

        if interpolated_indicator[-1]:
            linestyle = '--'

        plt.plot([ends[-1], ends[-1]], [-0.2, 0.2], linestyle, color="black", linewidth=3)

        # Text and points
        axis.text(ends[-1] - 0.015, 0.01, str(ends_levels[-1]),
                  fontsize=fontsize - 10, ha='center', color="blue")

        # Interpolated points
        xValues = [point for i, point in enumerate(total_grid) if interpolated_indicator[i]]
        axis.plot(xValues, np.zeros(len(xValues)), 'o', markersize=markersize, markeredgewidth=3, fillstyle="none", color="black")

        # Grid points
        xValues = [point for i, point in enumerate(total_grid) if not interpolated_indicator[i]]
        axis.plot(xValues, np.zeros(len(xValues)), 'bo', markersize=markersize, color="black")

        start, end = total_grid[0], total_grid[-1]
        axis.set_xlim([start - 0.005, end + 0.005])
        axis.set_ylim([-0.05, 0.05])
        axis.set_yticks([])

        # Set custom ticks
        ticks = np.linspace(total_grid[0], total_grid[-1], 9)
        plt.xticks(ticks, ticks, fontsize=40)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()


class ExtrapolationGridSlice(ABC):
    """
    This abstract class provides an interface for grid slices.

    :param interval: An array that contains the two boundary points of this slice.
    :param levels: An array with the level of the left and right point.
    :param support_sequence: A sequence of refined support points for extrapolation.
    :param function: for error computation.
    """

    def __init__(self, interval: Sequence[float], levels: Sequence[int],
                 support_sequence: Sequence[Tuple[float, float]], extrapolation_version: ExtrapolationVersion = None,
                 left_point_is_interpolated: bool = False, right_point_is_interpolated: bool = False,
                 function: Function = None):
        assert interval[0] < interval[1]

        self.left_point = interval[0]
        self.right_point = interval[1]
        self.width = self.right_point - self.left_point

        self.max_level = max(levels)
        self.levels = levels

        assert support_sequence is None or self.max_level == len(support_sequence) - 1
        self.support_sequence = support_sequence
        self.extrapolation_version = extrapolation_version

        # Weights
        self.extrapolated_weights_dict = None

        # Interpolation
        self.left_point_is_interpolated = left_point_is_interpolated
        self.right_point_is_interpolated = right_point_is_interpolated

        # Analytical function for comparison
        self.function = None
        self.analytical_solution = None

        self.set_function(function)

        # Adjacent slices
        self.adjacent_slice_left = None
        self.adjacent_slice_right = None

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Weights

    @abstractmethod
    def get_weight_for_left_and_right_support_point(self, left_support_point: float, right_support_point: float)\
            -> Tuple[float, float]:
        """
        This methods computes the weights for the support points, that correspond to the area of this slice

        :param left_support_point: left point from the current element of the support sequence
        :param right_support_point: right point from the current element of the support sequence
        :returns: (left_weight, right_weight).
        """

        pass

    def get_support_points_with_their_weights(self, level: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        This method returns the left/right support point with its corresponding weight for a given level

        Example: Stripe [0, 0.5], Grid [0, 0.5, 0.625, 0.75, 1], Level 0.
        left support point is 0, right support point is 1!! (not 1/2)

        => returns [(0, weight_0), (1, weight_1)]

        :param level: This parameter determines the index of element from the support sequence.
                        Level -1 indexes the slice boundaries. Level 0 indexes the integration domain boundaries.
        :returns: (left_point, left_weight), (right_point, right_weight).
        """

        assert 0 <= level <= self.max_level

        left_point, right_point = self.support_sequence[level]
        left_weight, right_weight = self.get_weight_for_left_and_right_support_point(left_point, right_point)

        return (left_point, left_weight), (right_point, right_weight)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Extrapolation of weights

    @abstractmethod
    def get_final_weights(self) -> Dict[float, float]:
        """
        This method computes the final (extrapolated) weights.

        :returns: a dictionary that maps grid points to a list of their (extrapolated) weights.
        """

        pass

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Errors

    def get_absolute_error_at_level(self, level: int) -> float:
        """
        This method computes the absolute error at a given level

        :param level: index in the support sequence
        :returns: absolute error of this slice at a given level (in respect the the analytical solution).
                    This is not the final error.
        """
        assert self.function is not None
        point_weight_pairs = self.get_support_points_with_their_weights(level)

        slice_approximation_value = 0

        for (point, weight) in point_weight_pairs:
            slice_approximation_value += weight * self.function.eval(point)

        return abs(slice_approximation_value - self.analytical_solution)

    def get_extrapolated_error(self) -> float:
        """
        This method computes the absolute final error.

        :returns: absolute error (in respect the the analytical solution) of the extrapolated slice.
        """

        assert self.function is not None

        # Get extrapolated weights
        if self.extrapolated_weights_dict is None:
            self.extrapolated_weights_dict = self.get_final_weights()

        weight_dict = self.extrapolated_weights_dict

        # Compute value
        slice_approximation_value = 0
        for grid_point, weights in weight_dict.items():
            slice_approximation_value += sum(weights) * self.function.eval(grid_point)

        return abs(slice_approximation_value - self.analytical_solution)

    def print_error_evolution(self) -> None:
        # Print default error
        for level in range(self.max_level + 1):
            error = self.get_absolute_error_at_level(level)
            print("   {} on level {}".format(error, level))

        # Print extrapolated error
        extrapolated_error = self.get_extrapolated_error()
        print("   {} after extrapolation".format(extrapolated_error))

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Getter & Setter

    def set_function(self, function: Function) -> None:
        self.function = function

        if function is not None:
            self.analytical_solution = function.getAnalyticSolutionIntegral([self.left_point], [self.right_point])
        else:
            self.analytical_solution = None

    def is_interpolated(self) -> bool:
        return self.left_point_is_interpolated or self.right_point_is_interpolated

    def is_left_point_interpolated(self) -> bool:
        return self.left_point_is_interpolated

    def is_right_point_interpolated(self) -> bool:
        return self.right_point_is_interpolated

    def set_adjacent_slice_left(self, slice: 'ExtrapolationGridSlice') -> None:
        self.adjacent_slice_left = slice

    def set_adjacent_slice_right(self, slice: 'ExtrapolationGridSlice') -> None:
        self.adjacent_slice_right = slice

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def to_string(self, name="ExtrapolationGridSlice") -> str:
        return "{} [{}, {}]".format(name, self.left_point, self.right_point)


class ExtrapolationGridSliceContainer(ABC):
    """
    This abstract class provides an interface for grid slice containers.
    Implementing classes store >= 1 grid slices that are adjacent and together span a partial full grid
    of equidistant step width.

    :param function: for error computation.
    """

    def __init__(self, function: Function = None):
        self.slices = []
        self.left_point = None
        self.right_point = None
        self.minimal_step_width = None

        # Determination of step width by min and max level
        self.max_level = None  # This max_level is not shifted!!

        self.extrapolated_weights_dict = None

        # Store all adjacent containers of the global grid to the left and right of this container
        self.adjacent_containers_left = None
        self.adjacent_containers_right = None

        # Analytical function for comparison
        self.function = None
        self.analytical_solution = None
        self.set_function(function)

        self.max_interpolation_support_points = 0

    def append_slice(self, slice: ExtrapolationGridSlice) -> None:
        """
        This method appends a slice of arbitrary type to the container.

        :param slice: An extrapolation grid slice of arbitrary type.
        :returns: void.
        """

        # Update container interval
        if self.left_point is None:
            self.left_point = slice.left_point

        # Right bound has to be updated every time a new slice is appended
        self.right_point = slice.right_point
        self.max_level = max(self.max_level, slice.max_level) if self.max_level is not None else slice.max_level

        if self.minimal_step_width is not None:
            self.minimal_step_width = min(self.minimal_step_width, slice.right_point - slice.left_point)
        else:
            self.minimal_step_width = slice.right_point - slice.left_point

        self.slices.append(slice)

    def initialize_adjacent_slices_for_all_slices(self) -> None:
        # First slice of this container
        container_left = self.get_adjacent_container_left()
        slice_left = container_left.slices[-1] if container_left is not None else None

        # Last slice of this container
        container_right = self.get_adjacent_container_right()
        slice_right = container_right.slices[0] if container_right is not None else None

        slices = [slice_left] + self.slices + [slice_right]

        # Inner slices
        for i in range(1, len(slices) - 1):
            slices[i].set_adjacent_slice_left(slices[i-1])
            slices[i].set_adjacent_slice_right(slices[i+1])

    @abstractmethod
    def get_final_weights(self) -> Dict[float, float]:
        """
        This method computes the final (extrapolated) slices in this container.

        :returns: a dictionary of weights.
        """

        pass

    def get_grid(self) -> Sequence[float]:
        """
        This method returns all grid points in this container (interpolated ones included).

        :returns: the grid of this container as array
        """

        return list(map(lambda s: s.left_point, self.slices)) + [self.slices[-1].right_point]

    def get_grid_levels(self) -> Sequence[int]:
        """
        This method returns all grid levels in this container (interpolated ones included).

        :returns: the grid levels of this container as array
        """

        return list(map(lambda s: s.levels[0], self.slices)) + [self.slices[-1].levels[1]]

    def get_grid_without_interpolated_points(self) -> Sequence[float]:
        """
        This method returns only grid points that have not been interpolated.

        :returns: the grid of this container as array
        """
        grid = self.get_grid()
        interpolation_indicator = self.get_interpolated_grid_points_indicator()

        truncated_grid = [grid[i] for i, indicator in enumerate(interpolation_indicator) if not indicator]

        return truncated_grid

    def get_grid_levels_without_interpolated_points(self) -> Sequence[int]:
        """
        This method returns all grid levels in this container (without interpolated grid levels).

        :returns: the grid levels of this container as array
        """
        grid_levels = self.get_grid_levels()
        interpolation_indicator = self.get_interpolated_grid_points_indicator()

        truncated_grid_levels = [grid_levels[i] for i, indicator in enumerate(interpolation_indicator) if not indicator]

        return truncated_grid_levels

    def get_full_grid_levels_between(self, left_level: int, right_level: int, max_level: int) -> Sequence[int]:
        """
        This method computes the grid levels of the full grid between two given levels.

        E.g. Given two boundary levels 1, 2 and a maximal level 4
        => interpolated grid levels [1, 4, 3, 4, 2].

        :returns: the normalized grid levels of this container.
        """

        if left_level == max_level == right_level:
            return self.get_grid_levels()

        interpolated_levels = self.__get_full_grid_levels_between(max(left_level, right_level), max_level)

        return [left_level] + interpolated_levels + [right_level]

    def __get_full_grid_levels_between(self, start_level: int, max_level: int) -> Sequence[int]:
        if start_level > max_level:
            return []

        left_levels = self.__get_full_grid_levels_between(start_level + 1, max_level)
        right_levels = self.__get_full_grid_levels_between(start_level + 1, max_level)

        return left_levels + [start_level] + right_levels

    def get_normalized_grid_levels(self) -> Sequence[int]:
        """
        This method normalizes the grid levels (with interpolated levels)

        E.g. Given a full Romberg grid [0, 0.5, 0.625, 0.75, 1]
                             with levels [0, 1, 3, 2, 0]
        The container grid is [0.5, 0.625, 0.75]
             with grid levels [1, 3, 2]

        => normalized grid levels [0, 1, 0].

        :returns: the normalized grid levels of this container.
        """

        grid = self.get_grid()

        # Unit slice grid needs no normalization
        if len(grid) == 2:
            return self.get_grid_levels()

        # The container grid must contain an odd number of grid points
        assert len(grid) % 2 == 1
        assert len(grid) >= 3
        # Assert that the there are 2^k slices
        self.__assert_size()

        return [0] + self.__get_normalized_grid_levels(1, len(grid) - 2) + [0]

    def __get_normalized_grid_levels(self, start: float, stop: float, level: int = 1) -> Sequence[int]:
        middle = int((start + stop) / 2)

        if start > stop:
            return []
        elif start == stop:
            return [level]

        left_levels = self.__get_normalized_grid_levels(start, middle - 1, level + 1)
        right_levels = self.__get_normalized_grid_levels(middle + 1, stop, level + 1)

        return left_levels + [level] + right_levels

    def get_normalized_non_interpolated_grid_levels(self) -> Sequence[int]:
        """
        This method normalizes the grid levels while skipping the level of an interpolated grid point

        :returns: the normalized grid levels (without interpolated levels)
        """
        normalized_levels = self.get_normalized_grid_levels()
        interpolation_indicator = self.get_interpolated_grid_points_indicator()

        truncated_normalized_levels = [normalized_levels[i] for i, indicator in enumerate(interpolation_indicator)
                                       if not indicator]

        return truncated_normalized_levels

    def get_interpolated_grid_points_indicator(self) -> Sequence[bool]:
        """
        This method computes an indicator for interpolated grid points as array.
        If the i-th element in this array is true, the corresponding grid point at this index is interpolated

        :return: boolean array
        """
        indicator = list(map(lambda s: s.is_left_point_interpolated(), self.slices))
        indicator.append(self.slices[-1].is_right_point_interpolated())

        return indicator

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Errors

    def get_extrapolated_error(self) -> float:
        """
        Computes the absolute final error.

        :returns: absolute extrapolated error to the analytical solution
        """

        assert self.function is not None

        # Get extrapolated weights
        if self.extrapolated_weights_dict is None:
            self.extrapolated_weights_dict = self.get_final_weights()

        weight_dict = self.extrapolated_weights_dict

        # Compute value
        slice_approximation_value = 0
        for grid_point, weights in weight_dict.items():
            slice_approximation_value += sum(weights) * self.function.eval([grid_point])

        return abs(slice_approximation_value - self.analytical_solution)

    def print_error_evolution(self, name: str = "SliceContainer") -> None:
        print("The {} [{}, {}] has the following error evolution:".format(name, self.left_point, self.right_point))

        # Print unit slice error evolution
        if len(self.slices) == 1:
            self.slices[0].print_error_evolution()
            return

        print("   {} after extrapolation".format(self.get_extrapolated_error()))

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Getter & Setter

    def set_adjacent_containers(self, left: 'ExtrapolationGridSliceContainer',
                                right: 'ExtrapolationGridSliceContainer') -> None:
        self.adjacent_containers_left = left
        self.adjacent_containers_right = right

    def set_function(self, function) -> None:
        self.function = function

        if function is not None and self.left_point is not None and self.right_point is not None:
            self.analytical_solution = function.getAnalyticSolutionIntegral([self.left_point], [self.right_point])
        else:
            self.analytical_solution = None

        # Update function in all slices
        for slice in self.slices:
            slice.set_function(function)

    def get_slices(self) -> Sequence[ExtrapolationGridSlice]:
        return self.slices

    def get_adjacent_container_left(self) -> 'ExtrapolationGridSliceContainer':
        return self.adjacent_containers_left[0] if len(self.adjacent_containers_left) >= 1 else None

    def get_adjacent_container_right(self) -> 'ExtrapolationGridSliceContainer':
        return self.adjacent_containers_right[0] if len(self.adjacent_containers_right) >= 1 else None

    # Returns the amount of slices in this container
    def size(self) -> int:
        return len(self.slices)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def split_into_containers_with_power_two_sizes(self) -> Sequence['ExtrapolationGridSliceContainer']:
        """
        This method implements a optimized container splitting.
        The size of the container is considered as a sum of powers of 2.
        This implementation enables maximal partial full grid groups (of size 2^k) for the default Romberg.

        :return:
        """
        container_type = type(self)
        size = self.size()
        split_containers = []

        while size > 0:
            new_container_size = self.find_closest_power_below(size)
            new_container = container_type(self.function)

            for i in range(new_container_size):
                new_container.append_slice(self.slices.pop(0))

            self.assert_container_size(new_container)

            size -= new_container_size
            split_containers.append(new_container)

        return split_containers

    @staticmethod
    def find_closest_power_below(n: int, base: int = 2) -> int:
        """
        This method determines the closest power below a given number for a given base and returns it.

        :param n: upper bound.
        :param base: base for exponentiation.
        :returns: closest 2^k below given number n.
        """
        for i in range(n):
            left_pow = math.pow(base, i)
            right_pow = math.pow(base, i + 1)

            if n == right_pow:
                return int(right_pow)

            if left_pow < n < right_pow:
                return int(left_pow)

        return 1

    @staticmethod
    def assert_container_size(container: 'ExtrapolationGridSliceContainer') -> None:
        """
        This method assures that the container has a size of 2^k (Assert).

        :param container: Extrapolation container with slices.
        :returns: void.
        """

        assert (math.log(container.size(), 2)).is_integer()

    def update_container_information(self) -> None:
        self.left_point = self.slices[0].left_point
        self.right_point = self.slices[-1].right_point
        self.max_level = max(self.get_grid_levels())

        if self.minimal_step_width is not None:
            self.minimal_step_width = min(list([
                self.slices[i].right_point - self.slices[i].left_point for i in range(len(self.slices))
            ]))

    def to_string(self, name: str = "SliceContainer") -> str:
        str_builder = "{} [{}, {}] with the slices: \n".format(name, self.left_point, self.right_point)

        for i, slice in enumerate(self.slices):
            str_builder += "   " + slice.to_string() + "\n"

        return str_builder

    def __assert_size(self) -> None:
        assert (math.log(self.size(), 2)).is_integer()


class RombergGridSlice(ExtrapolationGridSlice):
    """
    This class store a grid slice for extrapolation.

    :param interval: An array that contains the two boundary points of this slice.
    :param levels: An array with the level of the left and right point.
    :param support_sequence: A sequence of refined support points for extrapolation.
    :param function: for error computation.
    """

    def __init__(self, interval: Tuple[float, float], levels: Tuple[float, float], support_sequence: Sequence[Tuple[float, ...]],
                 extrapolation_version: ExtrapolationVersion,
                 left_point_is_interpolated: bool = False, right_point_is_interpolated: bool = False,
                 function: Function = None):
        super(RombergGridSlice, self).__init__(interval, levels, support_sequence,
                                               extrapolation_version,
                                               left_point_is_interpolated, right_point_is_interpolated,
                                               function)

        self.extrapolation_version = extrapolation_version
        self.coefficient_factory = ExtrapolationCoefficientsFactory(self.extrapolation_version)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Weights

    # Based on generalized sliced trapezoidal rule
    # See thesis, for the derivation of this weights
    # This methods computes the weights for the support points, that correspond to the area of this slice
    def get_weight_for_left_and_right_support_point(self, left_support_point: float, right_support_point: float)\
            -> Tuple[float, float]:
        # Support points may only be located outside of this slice, or on the boundaries itself
        assert left_support_point <= self.left_point and right_support_point >= self.right_point
        assert left_support_point != right_support_point

        # Compute common terms for the left and right weight
        slice_width = self.width

        support_point_ratio = left_support_point / (left_support_point - right_support_point)  # denominator always != 0
        slice_support_ratio = (1 / 2) * (
                (self.right_point + self.left_point) / (left_support_point - right_support_point)
        )

        # Determine the weight for the left support point
        left_weight = slice_width * (1 - support_point_ratio + slice_support_ratio)

        # Determine the weight for the right support point
        right_weight = slice_width * (support_point_ratio - slice_support_ratio)

        # Return the weight for the support point
        return left_weight, right_weight

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Extrapolation of weights

    # This method returns a dictionary that maps grid points to a list of their extrapolated weights
    def get_final_weights(self) -> Dict[float, Sequence[float]]:
        # The first element of the support sequence contains the integration domain boundaries
        (a, b) = self.support_sequence[0]

        # Generate the extrapolation coefficients
        coefficient_factory = self.coefficient_factory.get(a, b, self.support_sequence)

        # Dictionary that maps grid points to their extrapolated weights
        weight_dictionary = defaultdict(list)

        for level in range(self.max_level + 1):  # 0 <= i <= max_level
            point_weight_pairs = self.get_support_points_with_their_weights(level)

            assert len(point_weight_pairs) == 2
            (left_point, left_weight) = point_weight_pairs[0]
            (right_point, right_weight) = point_weight_pairs[1]

            coefficient = coefficient_factory.get_coefficient(self.max_level, level)
            left_weight = coefficient * left_weight
            right_weight = coefficient * right_weight

            weight_dictionary[left_point].append(left_weight)
            weight_dictionary[right_point].append(right_weight)

        self.subtract_constants(weight_dictionary)

        # Update dictionary of extrapolated weights
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    def subtract_constants(self, weight_dictionary: Dict[float, Sequence[float]]) -> None:
        """
        This is an interface for constant subtraction based on the extrapolation expansion

        :param weight_dictionary: contains all weights
        :return: void
        """
        pass

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def to_string(self, name: str = None) -> str:
        return super().to_string("RombergGridSlice")


class RombergGridSliceConstantSubtraction(RombergGridSlice):
    def subtract_constants(self, weight_dictionary: Dict[float, Sequence[float]]) -> None:
        constants = SlicedRombergConstants(self)
        constant_dict = constants.get_final_extrapolation_constant_weights()

        for point, weight in sorted(constant_dict.items()):
            weight_dictionary[point].append(weight)


# This class stores >= 1 grid slices that are adjacent and together span a partial full grid
# of equidistant step width
class RombergGridSliceContainer(ExtrapolationGridSliceContainer):
    """
    This class groups >= 1 grid slices that are adjacent and span a full grid of equidistant step width together.
    If the slice count is > 1 a (default or custom) Romberg is executed in this full grid.
    Else the unit slice is extrapolated based on its own extrapolation type.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(RombergGridSliceContainer, self).__init__(function)

    def get_final_weights(self) -> Dict[float, Sequence[float]]:
        assert len(self.slices) > 0

        # This container has only one slice. => Extrapolate this unit slice
        if len(self.slices) == 1:
            return self.slices[0].get_final_weights()

        # This container has >= 2 slices
        # Execute default Romberg on this container
        factory = RombergWeightFactory.get(self.left_point, self.right_point, ExtrapolationVersion.ROMBERG_DEFAULT)
        grid = self.get_grid()
        normalized_grid_levels = self.get_normalized_grid_levels()
        normalized_max_level = max(normalized_grid_levels)

        weight_dictionary = defaultdict(list)

        # Extrapolate weights on this container
        for i, point in enumerate(grid):
            if (i == 0) or (i == len(grid) - 1):
                weight = factory.get_boundary_point_weight(normalized_max_level)
            else:
                weight = factory.get_inner_point_weight(normalized_grid_levels[i], normalized_max_level)

            weight_dictionary[point].append(weight)

        # Update pointer to dictionary
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    # See parent class
    def to_string(self, name: str = None) -> str:
        return super().to_string("RombergGridSliceContainer")


# This class stores >= 1 grid slices that are adjacent and together span a partial full grid
# of equidistant step width
class SimpsonRombergGridSliceContainer(ExtrapolationGridSliceContainer):
    """
    This class groups >= 1 grid slices that are adjacent and span a full grid of equidistant step width together.
    If the slice count is > 1 Romberg's method with Simpson sum as base rule is executed in this full grid.
    Else the unit slice is extrapolated based on its own extrapolation type.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(SimpsonRombergGridSliceContainer, self).__init__(function)

    def get_final_weights(self) -> Dict[float, Sequence[float]]:
        assert len(self.slices) > 0

        # This container has only one slice. => Extrapolate this unit slice
        if len(self.slices) == 1:
            return self.slices[0].get_final_weights()

        # This container has >= 2 slices
        # Execute default Romberg on this container
        factory = RombergWeightFactory.get(self.left_point, self.right_point, ExtrapolationVersion.ROMBERG_SIMPSON)
        grid = self.get_grid()
        normalized_grid_levels = self.get_normalized_grid_levels()
        normalized_max_level = max(normalized_grid_levels)

        weight_dictionary = defaultdict(list)

        # Extrapolate weights on this container
        for i, point in enumerate(grid):
            if (i == 0) or (i == len(grid) - 1):
                weight = factory.get_boundary_point_weight(normalized_max_level)
            else:
                weight = factory.get_inner_point_weight(normalized_grid_levels[i], normalized_max_level)

            weight_dictionary[point].append(weight)

        # Update pointer to dictionary
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    # See parent class
    def to_string(self, name: str = None) -> str:
        return super().to_string("SimpsonRombergGridSliceContainer")


class InterpolationGridSliceContainer(ExtrapolationGridSliceContainer):
    """
        This class interpolates as many missing points as possible inside the container and executes a default
        Romberg method on this interpolated partial full grid.

        For the interpolation we use the closest points to the left and right of this container.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(InterpolationGridSliceContainer, self).__init__(function)
        self.max_interpolation_support_points = 7

    def get_support_points_for_interpolation_by_levels(self, max_point_count: int) -> Sequence[float]:
        """
        This method computes support points that are used in the interpolation of grid points.
        The support points are distributed to the left containers, middle container and right containers.
        The points in the middle container a computed level wise

        :param max_point_count: determines how much interpolation support points should be used (>= 2).
        :return: get optimal distributed support points
        """
        assert max_point_count >= 2

        left_containers = self.adjacent_containers_left
        right_containers = self.adjacent_containers_right

        has_left_containers = len(left_containers) > 0
        has_right_containers = len(right_containers) > 0

        points_left = []
        points_right = []

        if has_left_containers:
            for container in left_containers:
                # At this point each container has at least two non-interpolated grid points
                points_left.extend(container.get_grid_without_interpolated_points()[0:-1])

        if has_right_containers:
            for container in right_containers:
                # At this point each container has at least two non-interpolated grid points
                points_right.extend(container.get_grid_without_interpolated_points()[1:])

        points_middle = self.get_grid_without_interpolated_points()
        levels_middle = self.get_grid_levels_without_interpolated_points()

        support_points = []

        # Determine how much points should be used to the left and right of this container
        #   as well as within the container itself.
        max_left, max_middle, max_right = self.get_interpolation_support_point_count(len(points_left),
                                                                                     len(points_middle),
                                                                                     len(points_right),
                                                                                     max_point_count)

        if max_left > 0:
            support_points.extend(points_left[-min(max_left, len(points_left)):])

        support_points.extend(self.get_grid_with_max_point_count(points_middle, levels_middle, max_middle))

        if max_right > 0:
            support_points.extend(points_right[0:min(max_right, len(points_right))])

        return support_points

    @staticmethod
    def get_interpolation_support_point_count(max_count_left: int, max_count_middle: int, max_count_right: int,
                                              max_count_total: int) -> Tuple[int, int, int]:
        max_iter = max_count_left + max_count_middle + max_count_right
        i = 0

        total_count = 0
        left_count = 0
        middle_count = 0
        right_count = 0

        if max_count_middle >= 2 and max_count_total >= 2:
            total_count = 2
            left_count = 0
            middle_count = 2

        while total_count < max_count_total and i < max_iter:
            if middle_count < max_count_middle:
                middle_count += 1
                total_count += 1

            if left_count < max_count_left and total_count < max_count_total:
                left_count += 1
                total_count += 1

            if right_count < max_count_right and total_count < max_count_total:
                right_count += 1
                total_count += 1

            i += 1

        return left_count, middle_count, right_count

    @staticmethod
    def get_grid_with_max_point_count(grid: Sequence[float], grid_levels: Sequence[int], count: int) -> Sequence[float]:
        """
        This method computes a truncated grid with <= count points. Points of a level are only added to the truncated
        grid if all available points on this level (+ previous points) fit below the point count.

        :param grid: grid points as array
        :param grid_levels: grid levels as array
        :param count: maximal point count
        :return: truncated grid with max. "count" points
        """

        assert count >= 0
        assert len(grid) == len(grid_levels)

        if count >= len(grid):
            return grid

        current_level = min(grid_levels)
        current_grid = []

        while True:
            current_grid_tmp = []
            for i in range(len(grid)):
                if grid_levels[i] <= current_level:
                    current_grid_tmp.append(grid[i])

            if len(current_grid_tmp) > count:
                return current_grid

            current_grid = current_grid_tmp
            current_level += 1

    def get_support_points_for_interpolation_geometrically(self, interp_point: float, max_point_count: int,
                                                           adaptive: bool = True) -> Sequence[float]:
        """
        This method computes support points that are used in the interpolation of grid points.
        The algorithms chooses the geometrical closest neighbours first.

        :param interp_point: the point around which the support points are selected.
        :param max_point_count: determines how much interpolation support points should be used (>= 2).
        :param adaptive: If enabled, the algorithm does increase the support points
                            when many negative weights occur while interpolating

        :return: geometrically closest support points
        """
        assert max_point_count >= 2

        # Get non interpolated global grid
        non_interpolated_global_grid = []

        for container in list(reversed(self.adjacent_containers_left)):
            non_interpolated_global_grid.extend(container.get_grid_without_interpolated_points())

        non_interpolated_global_grid.extend(self.get_grid_without_interpolated_points())

        for i, container in enumerate(self.adjacent_containers_right):
            non_interpolated_global_grid.extend(container.get_grid_without_interpolated_points())

        # Remove duplicates
        non_interpolated_global_grid = list(dict.fromkeys(non_interpolated_global_grid))

        # Find closest non-interpolated neighbours
        left_index = None
        right_index = None

        for i in range(len(non_interpolated_global_grid) - 1):
            left = non_interpolated_global_grid[i]
            right = non_interpolated_global_grid[i + 1]

            if left < interp_point < right:
                left_index = i
                right_index = i + 1  # i+1, because interp_point is not in the real grid
                break

        # Determine geometrically closest support points
        support_points = []

        points_left_exist = left_index is not None and left_index >= 0
        points_right_exist = right_index is not None and right_index < len(non_interpolated_global_grid)

        while (len(support_points) < max_point_count) and (points_left_exist or points_right_exist):
            new_left = False
            new_right = False

            if points_left_exist:
                new_support_point = non_interpolated_global_grid[left_index]
                support_points.insert(0, new_support_point)
                left_index -= 1
                new_left = True

            if points_right_exist and len(support_points) < max_point_count:
                new_support_point = non_interpolated_global_grid[right_index]
                support_points.append(new_support_point)
                right_index += 1
                new_right = True

            # Break if many negative lagrange weights occur
            negative_sum = 0

            if adaptive and len(support_points) > 2:
                for support_point in support_points:
                    weight = self.get_interpolation_weight(support_points, support_point, interp_point)

                    if weight < 0:
                        negative_sum += weight

                    if negative_sum > 1 / self.max_interpolation_support_points:
                        if new_left:
                            support_points.pop(0)

                        if new_right and len(support_points) > 2:
                            support_points.pop(-1)

                        break

            points_left_exist = left_index >= 0
            points_right_exist = right_index < len(non_interpolated_global_grid)

        if len(support_points) < 2:
            assert len(support_points) >= 2, "There are to few support points, for an interpolation."

        return support_points

    @staticmethod
    def get_levels_for_support_point_grid(support_points: Sequence[float]):
        assert len(support_points) >= 2

        inner_levels = None

        if len(support_points) == 2:
            inner_levels = []

        else:
            inner_levels = InterpolationGridSliceContainer.__get_levels_for_support_points_rec(
                support_points,
                start_index=1,
                stop_index=(len(support_points)-1)-1,  # -1 for last index and -1 for 2nd last index
                level=1
            )

        assert len(inner_levels) + 2 == len(support_points)

        return [0] + inner_levels + [0]

    @staticmethod
    def __get_levels_for_support_points_rec(support_points: Sequence[float],
                                                          start_index: float, stop_index: float,
                                                          level: int = 1) -> Sequence[int]:
        middle_index = int((start_index + stop_index) / 2)
        # middle = support_points[middle_index]

        if start_index > stop_index:
            return []
        elif start_index == stop_index:
            return [level]

        left_levels = InterpolationGridSliceContainer.__get_levels_for_support_points_rec(
            support_points,
            start_index,
            middle_index - 1,
            level + 1
        )
        right_levels = InterpolationGridSliceContainer.__get_levels_for_support_points_rec(
            support_points,
            middle_index + 1,
            stop_index,
            level + 1
        )

        return left_levels + [level] + right_levels

    def insert_interpolating_slices(self) -> None:
        """
        This method inserts interpolation slices into the container.

        :return: void.
        """
        interpolation_points = self.get_interpolation_points()

        # No interpolation points have to be inserted
        if len(interpolation_points) == 0:
            return

        container_grid = self.get_grid()
        container_grid_levels = self.get_grid_levels()

        new_slices = []

        for i in range(len(container_grid) - 1):
            # Add last slices that have no interpolation points
            if len(interpolation_points) == 0:
                new_slices.extend(self.slices[i:])
                break

            factory = ExtrapolationGridSliceFactory(ExtrapolationGridSliceFactory.get_slice_version(self.slices[i]))

            # Grid that has to be swapped in
            points_between = [p for p in interpolation_points if container_grid[i] < p < container_grid[i + 1]]
            points = [container_grid[i]] + points_between + [container_grid[i+1]]

            levels = self.get_full_grid_levels_between(container_grid_levels[i], container_grid_levels[i+1],
                                                       self.max_level)

            # Replace slice with multiple interpolating slices
            if len(points_between) > 0:
                for j in range(0, len(points)-1, 2):
                    left_index = j
                    right_index = j+1
                    left_interpolated_slice = factory.get_grid_slice(
                        [points[left_index], points[right_index]],
                        [levels[left_index], levels[right_index]],
                        left_point_is_interpolated=left_index > 0,
                        right_point_is_interpolated=right_index < len(points)-1,
                        function=self.function
                    )
                    assert left_interpolated_slice.is_right_point_interpolated() is True

                    left_index = j+1
                    right_index = j+2
                    right_interpolated_slice = factory.get_grid_slice(
                        [points[left_index], points[right_index]],
                        [levels[left_index], levels[right_index]],
                        left_point_is_interpolated=left_index > 0,
                        right_point_is_interpolated=right_index < len(points) - 1,
                        function=self.function
                    )

                    assert right_interpolated_slice.is_left_point_interpolated() is True

                    # Exchange slice with two interpolated slices
                    new_slices.append(left_interpolated_slice)
                    new_slices.append(right_interpolated_slice)

                    # Pop queue
                    interpolation_points.pop(0)
            else:  # No interpolation slices inserted
                new_slices.append(self.slices[i])

        self.slices = new_slices

    def get_interpolation_points(self) -> Sequence[float]:
        """
        This methods returns the interpolated grid points as a list of points.

        :return: list of interpolated grid points.
        """
        container_grid = self.get_grid_without_interpolated_points()
        step_width = self.minimal_step_width
        interpolation_points = []

        current_point = container_grid[0]
        while current_point < container_grid[-1]:
            current_point += step_width

            if current_point not in container_grid:
                interpolation_points.append(current_point)

        return interpolation_points

    def get_final_weights(self) -> Dict[float, Sequence[float]]:
        assert len(self.slices) > 0

        if len(self.slices) == 1 and not self.slices[0].is_interpolated():
            return self.slices[0].get_final_weights()

        # This container has >= 2 slices
        # Execute default Romberg on this container
        factory = RombergWeightFactory.get(self.left_point, self.right_point, ExtrapolationVersion.ROMBERG_DEFAULT)
        weight_dictionary = defaultdict(list)

        # Non-interpolated grid points
        self.populate_non_interpolated_point_weights(factory, weight_dictionary)

        # Interpolated grid points
        self.populate_interpolated_point_weights(factory, weight_dictionary)

        # Update pointer to dictionary
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    def populate_non_interpolated_point_weights(self, factory: RombergWeights,
                                                weight_dictionary: Dict[float, Sequence[float]]) -> None:
        grid = self.get_grid_without_interpolated_points()
        normalized_grid_levels = self.get_normalized_non_interpolated_grid_levels()
        normalized_max_level = max(normalized_grid_levels)

        # Extrapolate weights on this container
        for i, point in enumerate(grid):
            if (i == 0) or (i == len(grid) - 1):
                weight = factory.get_boundary_point_weight(normalized_max_level)
            else:
                weight = factory.get_inner_point_weight(normalized_grid_levels[i],
                                                        normalized_max_level)

            weight_dictionary[point].append(weight)

    def populate_interpolated_point_weights(self, factory: RombergWeights,
                                            weight_dictionary: Dict[float, Sequence[float]]) -> None:
        indicator = self.get_interpolated_grid_points_indicator()
        grid = [point for i, point in enumerate(self.get_grid()) if indicator[i]]
        normalized_grid_levels = [level for i, level in enumerate(self.get_normalized_grid_levels())
                                    if indicator[i]]
        normalized_max_level = max(self.get_grid_levels())  # Max. level of whole grid (with interpolated points)

        for i, interp_point in enumerate(grid):
            level = normalized_grid_levels[i]
            support_points = self.\
                get_support_points_for_interpolation_geometrically(interp_point, self.max_interpolation_support_points,
                                                                   adaptive=True)

            if level == 0:
                weight = factory.get_boundary_point_weight(normalized_max_level)
            else:
                weight = factory.get_inner_point_weight(normalized_grid_levels[i],
                                                        normalized_max_level)

            print()
            for support_point in support_points:
                interpolation_weight = self.get_interpolation_weight(support_points, support_point, interp_point)
                # print(interpolation_weight)
                interpolated_extrapolated_weight = weight * interpolation_weight
                weight_dictionary[support_point].append(interpolated_extrapolated_weight)

            # TODO add a flag for plotting this
            # Plot basis functions
            # self.plot_interpolation_basis_functions(support_points, interp_point)
            # self.plot_interpolation_basis_functions_with_evaluations(support_points, interp_point)

    def get_interpolation_weight(self, support_points: Sequence[float],
                                 basis_point: float, evaluation_point: float) -> float:
        pass

    def plot_interpolation_basis_functions(self, support_points: Sequence[float], interpolation_point: float):
        pass

    def plot_interpolation_basis_functions_with_evaluations(self, support_points: Sequence[float],
                                                            evaluation_point: float):
        pass


# This class was replaced with LagrangeRombergGridSliceContainer(GlobalBasisGridRombergGridSliceContainerAdapter)
class LagrangeRombergGridSliceContainerOld(InterpolationGridSliceContainer):
    """
        This class interpolates as many missing points as possible inside the container and executes a default
        Romberg method on this interpolated partial full grid.

        For the interpolation we use the closest points to the left and right of this container.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(LagrangeRombergGridSliceContainerOld, self).__init__(function)
        self.max_interpolation_support_points = 7

    def get_interpolation_weight(self, support_points, basis_point, evaluation_point) -> float:
        return self.get_langrange_basis(support_points, basis_point, evaluation_point)

    @staticmethod
    def get_langrange_basis(support_points: Sequence[float], basis_point: float, evaluation_point: float) -> float:
        """
        This method computes a lambda expression for the lagrange basis function at basis_point.
        Specifically: Let support_points = [x_0, x_1, ... , x_n]
        L = product( (evaluation_point - x_k) / (basis_point - x_k) )

        :param support_points: an array of support points for the interpolation
        :param basis_point: determines basis_point of this lagrange basis function
        :param evaluation_point: determines at which point the basis is evaluated
        :return: the evaluated basis function L
        """

        evaluated_basis = 1

        for point in support_points:
            if point != basis_point:
                evaluated_basis *= (evaluation_point - point) / (basis_point - point)

        return evaluated_basis

    def plot_interpolation_basis_functions_with_evaluations(self, support_points, evaluation_point):
        # Iterate over all dimensions and plot each basis
        # plt.figure()

        columns = 2
        fig, axes = plt.subplots(figsize=(10, 15), nrows=math.ceil(len(support_points) / columns), ncols=columns)
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle("Basis functions for the interpolation of the point {}".format(evaluation_point))

        axes = axes.ravel()

        a = support_points[0]
        b = support_points[-1]

        for i, basis_point in enumerate(support_points):
            if self.function is not None:
                evaluation_points = np.linspace(a, b, len(support_points) * 100)
                axes[i].plot(evaluation_points, [self.function([point]) for point in evaluation_points],
                         linestyle=':', color="black")

            # Plot basis functions
            for support_point in support_points:
                evaluation_points = np.linspace(a, b, len(support_points) * 100)
                axes[i].plot(evaluation_points, [self.get_langrange_basis(support_points, support_point, point)
                                             for point in evaluation_points])

            # Print interpolation grid
            axes[i].plot(support_points, np.zeros(len(support_points)), 'bo', markersize=4, color="black")

            # Plot evaluation point
            if evaluation_point is not None:
                evaluation_point_value = self.get_langrange_basis(support_points, basis_point, evaluation_point)
                axes[i].plot(evaluation_point, evaluation_point_value, 'bo', markersize=4, color="red")
                axes[i].plot([evaluation_point, evaluation_point], [0, evaluation_point_value], ':',
                             color="red")

            axes[i].set_title("Evaluation using basis function of point {}".format(basis_point))

            if len(support_points) < 15:
                axes[i].set_xticks(support_points)
                axes[i].set_xticklabels(support_points, fontsize=9)

        # Clear unused axes
        for i in range(len(support_points), len(axes)):
            fig.delaxes(axes[i])

        fig.show()


class GlobalBasisGridRombergGridSliceContainerAdapter(InterpolationGridSliceContainer):

    """
        This class provides an interfaces for interpolating as many missing points as possible inside the container.
        It uses the interpolation functionality from the GlobalBasisGrid classes.

        It executes a default Romberg method on this interpolated partial full grid.
        For the interpolation we use the closest points to the left and right of this container.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(GlobalBasisGridRombergGridSliceContainerAdapter, self).__init__(function)
        self.max_interpolation_support_points = 7
        self.interpolation_grid = None

    @abstractmethod
    def construct_interpolation_grid(self, support_points):
        """

        Parameters
        ----------
        support_points

        Returns GlobalBasisGrid
        -------

        """
        pass

    def get_interpolation_weight(self, support_points, basis_point, evaluation_point) -> float:
        # Interpolation grid is computed beforehand
        interpolation_grid = self.construct_interpolation_grid(support_points)
        self.interpolation_grid = interpolation_grid

        support_point_grid_levels = InterpolationGridSliceContainer.get_levels_for_support_point_grid(support_points)
        interpolation_grid.set_grid([support_points], [support_point_grid_levels])

        # Compute weight
        assert len(interpolation_grid.basis) == 1
        basis = interpolation_grid.basis[0]
        assert len(basis) == len(support_points)

        i = support_points.index(basis_point)  # Element must be found
        weight = basis[i](evaluation_point)
        # TODO which basis function should be chosen?

        return weight

    def plot_interpolation_basis_functions(self, support_points, interpolation_point):
        self.interpolation_grid.plot_basis_functions(support_points, interpolation_point, self.function)

    def plot_interpolation_basis_functions_with_evaluations(self, support_points, evaluation_point):
        self.interpolation_grid.plot_basis_functions_with_evaluations(support_points, evaluation_point, self.function)


class BSplineRombergGridSliceContainer(GlobalBasisGridRombergGridSliceContainerAdapter):

    """
        This class interpolates (BSplines) as many missing points as possible inside the container
        It  executes a default Romberg method on this interpolated partial full grid.

        For the interpolation we use the closest points to the left and right of this container.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(BSplineRombergGridSliceContainer, self).__init__(function)
        self.max_interpolation_support_points = 7

    def construct_interpolation_grid(self, support_points):
        # Prevent circular dependencies
        from sparseSpACE.Grid import GlobalBSplineGrid

        # Interpolation grid is computed beforehand
        return GlobalBSplineGrid([support_points[0]], [support_points[-1]], boundary=True)


class LagrangeRombergGridSliceContainer(GlobalBasisGridRombergGridSliceContainerAdapter):

    """
        This class interpolates (Hierarchical Lagrange) as many missing points as possible inside the container
        It  executes a default Romberg method on this interpolated partial full grid.

        For the interpolation we use the closest points to the left and right of this container.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(LagrangeRombergGridSliceContainer, self).__init__(function)
        self.max_interpolation_support_points = 7

    def construct_interpolation_grid(self, support_points):
        # Prevent circular dependencies
        from sparseSpACE.Grid import GlobalLagrangeGrid

        # Interpolation grid is computed beforehand
        return GlobalLagrangeGrid([support_points[0]], [support_points[-1]], boundary=True)


class HierarchicalLagrangeRombergGridSliceContainer(GlobalBasisGridRombergGridSliceContainerAdapter):

    """
        This class interpolates (Hierarchical Lagrange) as many missing points as possible inside the container
        It  executes a default Romberg method on this interpolated partial full grid.

        For the interpolation we use the closest points to the left and right of this container.
    """

    def __init__(self, function: Function = None):
        """
            Constructor of this class.

            :param function: for error computation.
        """
        super(HierarchicalLagrangeRombergGridSliceContainer, self).__init__(function)
        self.max_interpolation_support_points = 7

    def construct_interpolation_grid(self, support_points):
        # Prevent circular dependencies
        from sparseSpACE.Grid import GlobalHierarchicalLagrangeGrid

        # Interpolation grid is computed beforehand
        return GlobalHierarchicalLagrangeGrid([support_points[0]], [support_points[-1]], boundary=True)


class TrapezoidalGridSlice(ExtrapolationGridSlice):
    """
     This type of GridSlice executes no extrapolation. It produces default trapezoidal weights.

     :param interval: An array that contains the two boundary points of this slice.
     :param levels: An array with the level of the left and right point.
     :param support_sequence: A sequence of refined support points for extrapolation.
     :param function: for error computation.
     """

    def __init__(self, interval: Tuple[float, float], levels: Tuple[float, float], support_sequence: Sequence[Tuple[float, float]],
                 left_point_is_interpolated: bool = False, right_point_is_interpolated: bool = False,
                 function: Function = None):
        super(TrapezoidalGridSlice, self).__init__(interval, levels, support_sequence,
                                                   left_point_is_interpolated=left_point_is_interpolated,
                                                   right_point_is_interpolated=right_point_is_interpolated,
                                                   function=function)

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Weights

    # See thesis, for the derivation of this weights
    # This methods computes the weights for the support points, that correspond to the area of this slice
    def get_weight_for_left_and_right_support_point(self, left_support_point: float, right_support_point: float)\
            -> Tuple[float, float]:
        return self.width / 2, self.width / 2

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Final weights

    def get_final_weights(self) -> Dict[float, Sequence[float]]:
        # Dictionary that maps grid points to their extrapolated weights
        weight_dictionary = defaultdict(list)

        # There is no Extrapolation happening in this type of grid slice
        left_weight, right_weight = \
            self.get_weight_for_left_and_right_support_point(left_support_point=self.left_point,
                                                             right_support_point=self.right_point)

        weight_dictionary[self.left_point].append(left_weight)
        weight_dictionary[self.right_point].append(right_weight)

        # Update dictionary of weights
        self.extrapolated_weights_dict = weight_dictionary

        return weight_dictionary

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Helpers

    def to_string(self, name: str = None) -> str:
        return super().to_string("TrapezoidalGridSlice")


class ExtrapolationGridSliceFactory:
    """
     This class encapsulates the creation of different grid slice types (Factory Pattern).

     :param slice_version: Determines which type of slice has to be created.
     """

    def __init__(self, slice_version: SliceVersion):
        self.slice_version = slice_version

    def get_grid_slice(self, interval: Tuple[float, float], levels: Tuple[float, float],
                       support_sequence: Sequence[Tuple[float, float]] = None,
                       left_point_is_interpolated: bool = False, right_point_is_interpolated: bool = False,
                       function: Function = None) -> ExtrapolationGridSlice:
        if self.slice_version == SliceVersion.ROMBERG_DEFAULT:
            return RombergGridSlice(interval, levels, support_sequence,
                                    ExtrapolationVersion.ROMBERG_DEFAULT,
                                    left_point_is_interpolated, right_point_is_interpolated,
                                    function)

        elif self.slice_version == SliceVersion.ROMBERG_DEFAULT_CONST_SUBTRACTION:
            return RombergGridSliceConstantSubtraction(interval, levels, support_sequence,
                                                       ExtrapolationVersion.ROMBERG_DEFAULT,
                                                       left_point_is_interpolated, right_point_is_interpolated,
                                                       function)

        elif self.slice_version == SliceVersion.TRAPEZOID:
            return TrapezoidalGridSlice(interval, levels, support_sequence,
                                        left_point_is_interpolated, right_point_is_interpolated,
                                        function)

        else:
            raise RuntimeError("Wrong SliceVersion provided.")

    @staticmethod
    def get_slice_version(slice: ExtrapolationGridSlice) -> SliceVersion:
        if isinstance(slice, RombergGridSlice):
            if slice.extrapolation_version == ExtrapolationVersion.ROMBERG_DEFAULT:
                return SliceVersion.ROMBERG_DEFAULT
            else:
                raise RuntimeError("This slice extrapolation version does not exist")
        elif isinstance(slice, TrapezoidalGridSlice):
            return SliceVersion.TRAPEZOID

        raise RuntimeError("Cannot detect slice version")


class ExtrapolationGridSliceContainerFactory:
    """
     This class encapsulates the creation of different grid slice container types (Factory Pattern).

     :param slice_container_version: Determines which type of slice container has to be created.
     """

    def __init__(self, slice_container_version: SliceContainerVersion):
        self.slice_container_version = slice_container_version

    def get_grid_slice_container(self, function: Function = None) -> ExtrapolationGridSliceContainer:
        if self.slice_container_version == SliceContainerVersion.ROMBERG_DEFAULT:
            return RombergGridSliceContainer(function)
        elif self.slice_container_version == SliceContainerVersion.LAGRANGE_ROMBERG_OLD:
            return LagrangeRombergGridSliceContainerOld(function)
        elif self.slice_container_version == SliceContainerVersion.LAGRANGE_ROMBERG:
            return LagrangeRombergGridSliceContainer(function)
        elif self.slice_container_version == SliceContainerVersion.SIMPSON_ROMBERG:
            return SimpsonRombergGridSliceContainer(function)
        elif self.slice_container_version == SliceContainerVersion.BSPLINE_ROMBERG:
            return BSplineRombergGridSliceContainer(function)
        elif self.slice_container_version == SliceContainerVersion.HIERARCHICAL_LAGRANGE_ROMBERG:
            return HierarchicalLagrangeRombergGridSliceContainer(function)
        else:
            raise RuntimeError("Wrong ContainerVersion provided.")


class ExtrapolationConstants:
    """
    This class approximates the constants that have to be subtracted in the extrapolation process
    :param slice: the extrapolated slice
    """

    def __init__(self, slice: ExtrapolationGridSlice):
        self.slice = slice
        self.midpoint = (self.slice.left_point + self.slice.right_point) / 2

    def get_nth_derivative_approximation(self, support_points: Sequence[Tuple[float, float]], max_level: int):
        assert 0 <= max_level <= self.slice.max_level + 1

        variable = symbols('t')
        lagrange_weights = self.get_lagrange_interpolation_weights(support_points, variable)

        assert len(support_points) == len(lagrange_weights)

        # Compute (max_level)-th derivative of the lagrange weights
        lagrange_weights_derivative = list(lagrange_weights)
        for i in range(max_level):
            lagrange_weights_derivative = self.differentiate_symbolic_factors(lagrange_weights_derivative, variable)

        return lagrange_weights_derivative

    def get_interpolation_support_points(self, max_level: int) -> Sequence[float]:
        support_points = []

        # Geometrically add points left and right from the slice
        left_slice = self.slice
        right_slice = self.slice

        while len(support_points) <= max_level and (left_slice is not None or right_slice is not None):
            if left_slice is not None:
                support_points = [left_slice.left_point] + support_points
                left_slice = left_slice.adjacent_slice_left

            if right_slice is not None and len(support_points) <= max_level:
                support_points.append(right_slice.right_point)
                right_slice = right_slice.adjacent_slice_right

        return support_points

    @staticmethod
    def get_lagrange_interpolation_weights(support_points: Sequence[float], variable=None):
        """
        This methods computes the symbolic factors for each support point through lagrange interpolation.

        :param support_points: an symbolic index based array of support points for the lagrange interpolation.
        :param variable: a SymPy symbol, that is the variable of the lagrange polynomial.
        :return: an array of lagrange weight factors.
        """
        assert len(support_points)

        if variable is None:
            variable = sym.symbols('t')

        factors = []

        for k in range(len(support_points)):
            factor = 1

            for j in range(len(support_points)):
                if j != k:
                    factor = factor * ((variable - support_points[j]) / (support_points[k] - support_points[j]))

            factors.append(factor)

        return factors

    @staticmethod
    def differentiate_symbolic_factors(factors, variable):
        """
        This method partially differentiates the given symbolic factor array one time.

        :param factors: an array of symbolic factors.
        :param variable: the variable we derive from.
        :return: an array of first derivatives.
        """
        return list(map(lambda factor: sym.diff(factor, variable), factors))

    @staticmethod
    def append_constants_dict(dictionary, points, constant):
        for i, point in enumerate(points):
            dictionary[point].append(constant[i])


class SlicedRombergConstants(ExtrapolationConstants):
    """
    These constants have been formally proven by a Taylor expansion of the slice trapezoidal rule followed by
    extrapolation steps.
    """

    def __init__(self, slice: ExtrapolationGridSlice):
        super(SlicedRombergConstants, self).__init__(slice)

        self.max_constant_level = self.slice.max_level + 2

        # Initialize derivatives
        self.derivatives = []
        self.support_points_for_derivatives = []
        for k in range(1, self.max_constant_level):
            support_points = self.get_interpolation_support_points(k)
            derivative = np.array(self.get_nth_derivative_approximation(support_points, k))

            self.derivatives.append(derivative)
            self.support_points_for_derivatives.append(support_points)

    def get_final_extrapolation_constant_weights(self) -> Dict[float, float]:
        """
        This method returns a dictionary that maps grid points to their summarized constants.
        (Only constants for right boundary refinement)

        Adds new constant weights to the dictionary.
        e.g. Extrapolation with a support sequence of length 3 => max_level = 2.
        In the first extrapolation step we combine eq. 1 with 2 and eq. 2 with 3.
        These two new equations are then combined in the second extrapolation step (in this step appear
            NO new constants. Instead the constants obtain new factors).

        For each of those extrapolations we have to add/subtract new constants weights to the dictionary.

        :return: void
        """

        # Return empty extrapolation dictionary, if no extrapolation happened
        if self.slice.max_level <= 0:
            return defaultdict(list)

        dict_list = []

        # Contants from first column of the Romberg table
        for i in range(1,  self.slice.max_level + 1):
            # Left or right extrapolation?
            prev_right_support = self.slice.support_sequence[i - 1][1]
            right_support = self.slice.support_sequence[i][1]

            if prev_right_support == right_support:  # Right boundary is unchanged => extrapolation of left boundary
                dict_list.append(self.__get_constants_for_left_boundary_extrapolation(i))
            else:  # extrapolation of right boundary
                dict_list.append(self.__get_constants_for_right_boundary_extrapolation(i))

        # Sum constant weights in dictionary for each point
        for i in range(len(dict_list)):
            constant_weights = {}

            for point, weights in sorted(dict_list[i].items()):
                constant_weights[point] = sum(weights)

            dict_list[i] = constant_weights

        # Special case: only one extrapolation step
        if len(dict_list) == 1:
            return dict_list[0]

        # Combination of the constants of the first column
        for j in range(2, self.slice.max_level+1):  # Iterate of the remaining columns
            new_dict_list = []

            for i in range(0, self.slice.max_level - j + 1):
                # Combine always the constants of two adjacent cells each
                dict1 = dict_list[i]
                dict2 = dict_list[i+1]

                factor1 = (self.slice.support_sequence[j + i][1] - self.slice.support_sequence[j + i][0]) ** 2
                factor2 = (self.slice.support_sequence[i][1] - self.slice.support_sequence[i][0]) ** 2

                new_dict = self.__subtract_constant_dicts(dict1, dict2, factor1, factor2)
                new_dict_list.append(new_dict)

            # Update dict_list with new combined constants
            dict_list = new_dict_list

        assert len(dict_list) == 1

        return dict_list[0]

    @staticmethod
    def __subtract_constant_dicts(dict1: Dict[float, float], dict2: Dict[float, float], factor1: float, factor2: float) \
            -> Dict[float, float]:
        """
        This method combines two dictionaries with factors.

        :param dict1:
        :param dict2:
        :param factor1:
        :param factor2:
        :return: a new dictionary that combines the constants
        """
        combined_dict = {}

        for point, weights in sorted(dict1.items()):
            combined_dict[point] = factor1 * dict1[point] - factor2 * dict2[point]
            del dict2[point]

        # Add weight of points that are not in dict1
        for point, weights in sorted(dict2.items()):
            combined_dict[point] = factor2 * dict2[point]

        return combined_dict

    def compute_taylor_derivative_factor(self, k: int):
        """
        This method computes and returns f^{(k)}(m) / k!.

        :param k: level of derivative
        :return: f^{(k)}(m) / k!
        """
        derivative_factors = self.derivatives[k - 1]

        return derivative_factors / math.factorial(k)

    def get_integration_constant_weights(self, k: int):
        """

        :param k: extrapolation level
        :return: support_points, constant_weights
        """
        support_points = self.support_points_for_derivatives[k - 1]
        derivative_factors = self.derivatives[k - 1]

        step_width = self.slice.width

        numerator = step_width ** (k + 1)
        denominator = int(math.pow(2, k)) * int(math.factorial(k + 1))
        constant_weights = derivative_factors * (numerator / denominator)

        return support_points, constant_weights

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Right boundary extrapolation

    def __get_constants_for_right_boundary_extrapolation(self, level: int):
        """
        Adds constants for a extrapolation of the right boundary.

        :param level: new level towards one extrapolates
        :return: dictionary of constant weights
        """

        constant_weights_dict = defaultdict(list)

        for k in range(1, self.max_constant_level):
            # - C_{1,k}
            support_points, constant_1_1 = self.get_constant_1_for_right_boundary_extrapolation(k)
            self.append_constants_dict(constant_weights_dict, support_points, -constant_1_1)

            # factor2
            seq = self.slice.support_sequence
            a_max, b_max = seq[level]
            a_prev, b_prev = seq[level - 1]
            H_max = b_max - a_prev
            H_prev = b_prev - a_prev
            factor_2 = ((H_max**2) / H_prev - (H_prev**2) / H_max) / (H_max**2 - H_prev**2)

            # - factor_2 * C_{2, k}
            support_points, constant_2_1 = self.get_constant_2_for_right_boundary_extrapolation(k)
            self.append_constants_dict(constant_weights_dict, support_points, - factor_2 * constant_2_1)

            # factor3
            m = self.midpoint

            def factor_3(exp):
                numerator = (((H_max**2)/H_prev)*(b_max - m)**exp - ((H_prev**2)/H_max)*(b_prev - m)**exp)
                denominator = H_max**2 - H_prev**2

                return numerator / denominator

            # - factor_3(k) * C_{3, k}
            support_points, constant_3_1 = self.get_constant_3_for_right_boundary_extrapolation(k)
            self.append_constants_dict(constant_weights_dict, support_points, - factor_3(k) * constant_3_1)

            # + C_{k} where k is even
            if k % 2 == 0:
                support_points, integration_constant = self.get_integration_constant_weights(k)
                self.append_constants_dict(constant_weights_dict, support_points, integration_constant)

        return constant_weights_dict

    def get_constant_1_for_right_boundary_extrapolation(self, k: int):
        """
        Computes C_{1,k} for right refinement extrapolation step

        :param k: extrapolation level
        :return: support_points, constant_weights
        """
        assert k >= 1

        derivative_step = self.slice.width * self.compute_taylor_derivative_factor(k)
        constant_weights = derivative_step * ((self.slice.left_point - self.midpoint) ** k)

        support_points = self.support_points_for_derivatives[k - 1]
        assert len(support_points) == len(constant_weights)

        return support_points, constant_weights

    def get_constant_2_for_right_boundary_extrapolation(self, k: int):
        """
        Computes C_{2,k}=C_{1,k}*(a-m) for right refinement extrapolation step

        :param k: extrapolation level
        :return: support_points, constant_weights
        """
        support_points, constant_weights = self.get_constant_1_for_right_boundary_extrapolation(k)

        return support_points, constant_weights * (self.slice.left_point - self.midpoint)

    def get_constant_3_for_right_boundary_extrapolation(self, k: int):
        """
        Computes C_{3,k}=C_{1,k} for right refinement extrapolation step

        :param k: extrapolation level
        :return: support_points, constant_weights
        """
        assert k >= 1

        derivative_step = self.slice.width * self.compute_taylor_derivative_factor(k)
        constant_weights = derivative_step * (self.slice.left_point - self.midpoint)

        support_points = self.support_points_for_derivatives[k - 1]
        assert len(support_points) == len(constant_weights)

        return support_points, constant_weights

    # -----------------------------------------------------------------------------------------------------------------
    # ---  Left boundary extrapolation

    def __get_constants_for_left_boundary_extrapolation(self, level: int):
        """
        Adds constants for a extrapolation of the left boundary.

        :param level: new level towards one extrapolates
        :return: dictionary of constant weights
        """

        constant_weights_dict = defaultdict(list)

        for k in range(1, self.max_constant_level):
            # Factor 1
            seq = self.slice.support_sequence
            a_max, b_max = seq[level]
            a_prev, b_prev = seq[level - 1]
            H_max = b_prev - a_max
            H_prev = b_prev - a_prev
            m = self.midpoint

            def factor_1(exp):
                numerator = (((H_max ** 2) * (a_prev - m) ** exp) - ((H_prev ** 2) * (a_max - m) ** exp))
                denominator = H_max ** 2 - H_prev ** 2

                return numerator / denominator

            # - factor_1(k) * C_{1,k}
            support_points, constant_1_1 = self.get_constant_1_for_left_boundary_extrapolation(k)
            self.append_constants_dict(constant_weights_dict, support_points, - factor_1(k) * constant_1_1)

            # factor2
            def factor_2(exp):
                numerator = (((H_max ** 2) * (a_prev - m) ** (exp+1)) / H_prev
                             - ((H_prev ** 2) * (a_max - m) ** (exp+1)) / H_max)
                denominator = H_max ** 2 - H_prev ** 2

                return numerator / denominator

            # - factor_2 * C_{2, k}
            support_points, constant_2_1 = self.get_constant_2_for_left_boundary_extrapolation(k)
            self.append_constants_dict(constant_weights_dict, support_points, - factor_2(k) * constant_2_1)

            # factor3
            numerator = (((H_prev ** 2) / H_max) * (a_max - m)
                         - ((H_max ** 2) / H_prev) * (a_prev - m))
            denominator = H_max ** 2 - H_prev ** 2
            factor_3 = numerator / denominator

            # - factor_3(k) * C_{3, k}
            support_points, constant_3_1 = self.get_constant_3_for_left_boundary_extrapolation(k)
            self.append_constants_dict(constant_weights_dict, support_points, - factor_3 * constant_3_1)

            # + C_{k} where k is even
            if k % 2 == 0:
                support_points, integration_constant = self.get_integration_constant_weights(k)
                self.append_constants_dict(constant_weights_dict, support_points, integration_constant)

        return constant_weights_dict

    def get_constant_1_for_left_boundary_extrapolation(self, k: int):
        """
        Computes C_{1,k} for left refinement extrapolation step

        :param k: extrapolation level
        :return: support_points, constant_weights
        """
        assert k >= 1

        constant_weights = self.slice.width * self.compute_taylor_derivative_factor(k)

        support_points = self.support_points_for_derivatives[k - 1]
        assert len(support_points) == len(constant_weights)

        return support_points, constant_weights

    def get_constant_2_for_left_boundary_extrapolation(self, k: int):
        """
        Computes C_{2,k}=C_{1,k} for left refinement extrapolation step

        :param k: extrapolation level
        :return: support_points, constant_weights
        """
        support_points, constant_weights = self.get_constant_1_for_right_boundary_extrapolation(k)

        return support_points, constant_weights

    def get_constant_3_for_left_boundary_extrapolation(self, k: int):
        """
        Computes C_{3,k}=C_{1,k}*(b-m)^k for left refinement extrapolation step

        :param k: extrapolation level
        :return: support_points, constant_weights
        """
        support_points, constant_weights = self.get_constant_1_for_right_boundary_extrapolation(k)

        return support_points, constant_weights * (self.slice.right_point - self.midpoint) ** k


# -----------------------------------------------------------------------------------------------------------------
# ---  Balanced Extrapolation Grid

# Extrapolation for balanced binary trees
class BalancedExtrapolationGrid:
    def __init__(self, print_debug: bool = False):
        # Root node has level 1, since boundary points are note store in the tree
        self.root_node = None
        self.active_node_queue = []

        self.grid = None
        self.grid_levels = None
        self.max_level = None

        self.print_debug = print_debug

    def set_grid(self, grid: Sequence[float], grid_levels: Sequence[int]) -> None:
        """
        Given a grid level array, this method initializes a appropriate binary tree structure
        Example input:    grid [0, 0.125, 0.25, 0.375, 0.5, 0.75, 1]
                        levels [0, 3, 2, 3, 1, 2, 0]

        :param grid:
        :param grid_levels:
        :return:
        """
        self.grid = grid
        self.grid_levels = grid_levels
        self.max_level = max(grid_levels)

        assert grid_levels[0] == 0 and grid_levels[-1] == 0

        # build tree based on inner points (neglect index 0 and -1)
        self.init_tree_rec(None, grid, grid_levels, 1, len(grid_levels) - 2)

        assert self.root_node is not None

        # Assert that the tree is balanced
        for node in self.root_node.get_nodes_using_dfs_in_order():
            assert node.has_both_children() or node.is_leaf(), "The grid is not balanced!"

    def init_tree_rec(self, node: 'GridNode', grid: Sequence[float], grid_levels: Sequence[int],
                      start_index: int, stop_index: int, inside_left_subtree: bool = None) -> None:
        # Find index of minimal element in active interval
        current_grid_level_slice = grid_levels[start_index:stop_index + 1]
        if start_index < stop_index:
            split_index = start_index + current_grid_level_slice.index(min(current_grid_level_slice))
        else:
            split_index = start_index

        # Termination criteria
        if stop_index < start_index or split_index > stop_index or split_index < start_index:
            return
        # Root node
        elif node is None:
            self.root_node = GridNode(grid[0], grid[-1], print_debug=self.print_debug)
            self.init_tree_rec(self.root_node, grid, grid_levels,
                               start_index, split_index - 1, inside_left_subtree=True)
            self.init_tree_rec(self.root_node, grid, grid_levels,
                               split_index + 1, stop_index, inside_left_subtree=False)
        # Other nodes
        else:
            if inside_left_subtree:
                node.set_left_child(GridNode(grid[start_index - 1], grid[stop_index + 1], parent_node=node))
                node = node.get_left_child()
            else:
                node.set_right_child(GridNode(grid[start_index - 1], grid[stop_index + 1], parent_node=node))
                node = node.get_right_child()

            self.init_tree_rec(node, grid, grid_levels,
                               start_index, split_index - 1, inside_left_subtree=True)
            self.init_tree_rec(node, grid, grid_levels,
                               split_index + 1, stop_index, inside_left_subtree=False)

    def get_weights(self) -> Sequence[float]:
        assert self.grid is not None
        assert self.grid_levels is not None
        assert self.root_node is not None

        # Initialize list of weight dictionaries
        weight_dict_list = []

        # Build list of weights dictionaries for extrapolation
        for i in range(1, self.max_level + 1):
            leaf_nodes = self.root_node.get_leafs_or_max_level_nodes(max_level=i)

            weight_dict = defaultdict(np.float64)

            for leaf in leaf_nodes:
                weight_dict[leaf.grid_point] = leaf.get_step_width()

            weight_dict_list.append(weight_dict)

        # Extrapolate
        #   M_{0,0}     {}         {}
        #   M_{1,0}   M_{1,1}      {}
        #   M_{2,0}   M_{2,1}   M_{2,2}

        weight_dict_table = [weight_dict_list]  # Initialize first column of table

        for j in range(1, self.max_level):  # Iterate over columns
            # Initialize empty table entries
            weight_dict_table.append([{}] * j)
            column_list = weight_dict_table[j]

            # Extrapolate column entries row by row
            for i in range(j, self.max_level):
                left_entry = weight_dict_table[j-1][i]
                top_left_entry = weight_dict_table[j-1][i-1]

                new_entry = self.extrapolate_dicts_one_step(left_entry, top_left_entry, j)
                column_list.append(new_entry)

        # Transform dictionary to weights
        weights = []

        for grid_point in self.grid:
            weight = weight_dict_table[-1][-1][grid_point]
            weights.append(weight)

        assert len(weights) == len(self.grid)

        return weights

    @staticmethod
    def extrapolate_dicts_one_step(left_dict: Dict[float, float], top_left_dict, k) -> Dict[float, float]:
        new_dict = defaultdict(np.float64)

        # Compute a_k
        coefficient = (-1) / (4 ** k - 1)

        # Weights that exist in the left dict
        considered_grid_points = []

        for grid_point in left_dict:
            considered_grid_points.append(grid_point)
            new_dict[grid_point] = (1 - coefficient) * left_dict[grid_point] + coefficient * top_left_dict[grid_point]

        # Weight that do not exist in the left dict, but in the top left dict
        for grid_point, weight in top_left_dict.items():
            if grid_point not in considered_grid_points:
                new_dict[grid_point] = 0 + coefficient * weight

        return new_dict

    # Return grid with boundary points
    def get_grid(self, max_level: int = None) -> Sequence[float]:
        return [self.grid[0]] + self.root_node.get_grid(max_level) + [self.grid[-1]]

    # Return levels, with boundaries
    def get_grid_levels(self, max_level: int = None) -> Sequence[int]:
        return [0] + self.root_node.get_grid_levels(max_level) + [0]


class GridNode:
    # left bound, right bound of interval, root node has no parent
    def __init__(self, boundary_left: float, boundary_right: float,
                 left_child: 'GridNode' = None, right_child: 'GridNode' = None, parent_node: 'GridNode' = None,
                 start_level: int = 1, print_debug: bool = False):
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.grid_point = self.get_midpoint(boundary_left, boundary_right)

        self.left_child = left_child
        self.right_child = right_child
        self.parent_node = parent_node
        self.level = start_level if parent_node is None else parent_node.level + 1

        self.print_debug = print_debug

    def get_leafs_or_max_level_nodes(self, max_level: int = None) -> List['GridNode']:
        return self.__get_leafs_or_max_level_nodes(self, max_level)

    def __get_leafs_or_max_level_nodes(self, node: 'GridNode', max_level: int = None) -> List['GridNode']:
        """
        Assumption: Balanced tree

        :param node:
        :param max_level:
        :return:
        """
        if node is None or (max_level is not None and node.level > max_level):
            return []

        if node.level == max_level or node.is_leaf():
            return [node]

        left_subtree = self.__get_leafs_or_max_level_nodes(node.get_left_child(), max_level)
        right_subtree = self.__get_leafs_or_max_level_nodes(node.get_right_child(), max_level)

        return left_subtree + right_subtree

    def get_nodes_using_dfs_in_order(self, max_level: int = None) -> List['GridNode']:
        return self.__get_nodes_using_dfs_in_order_rec(self, max_level)

    def __get_nodes_using_dfs_in_order_rec(self, node: 'GridNode', max_level: int = None) -> List['GridNode']:
        if node is None or (max_level is not None and node.level > max_level):
            return []

        left_subtree = self.__get_nodes_using_dfs_in_order_rec(node.get_left_child(), max_level)
        right_subtree = self.__get_nodes_using_dfs_in_order_rec(node.get_right_child(), max_level)

        return left_subtree + [node] + right_subtree

    # Returns the grid of current subtree
    def get_grid(self, max_level: int = None) -> Sequence[float]:
        return list(map(lambda node: node.grid_point, self.get_nodes_using_dfs_in_order(max_level)))

    # Returns the levels of the current subtree
    def get_grid_levels(self, max_level: int = None) -> Sequence[int]:
        return list(map(lambda node: node.level, self.get_nodes_using_dfs_in_order(max_level)))

    # Setter methods for children nodes
    def set_left_child(self, left_child: 'GridNode') -> None:
        self.left_child = left_child

    def set_right_child(self, right_child: 'GridNode') -> None:
        self.right_child = right_child

    # Getter methods
    def get_step_width(self) -> float:
        return self.boundary_right - self.boundary_left

    def get_left_child(self) -> None:
        return self.left_child

    def get_right_child(self) -> None:
        return self.right_child

    def has_left_child(self) -> bool:
        return self.get_left_child() is not None

    def has_right_child(self) -> bool:
        return self.get_right_child() is not None

    def has_both_children(self) -> bool:
        return self.has_left_child() and self.has_right_child()

    def has_only_one_child(self):
        return (self.has_left_child() and (not self.has_right_child())) \
               or (self.has_right_child() and (not self.has_left_child()))

    def is_root_node(self) -> bool:
        return self.parent_node is None

    def has_parent_node(self) -> bool:
        return not self.is_root_node()

    def is_left_child(self) -> bool:
        return self.parent_node.get_left_child() is self

    def is_right_child(self) -> bool:
        return self.parent_node.get_right_child() is self

    # Return sibling of current node
    def get_sibling(self) -> 'GridNode':
        if self.is_root_node():
            return None

        parent_node = self.parent_node

        return parent_node.right_child if parent_node.left_child is self else parent_node.left_child

    def is_leaf(self) -> bool:
        return not self.has_left_child() and not self.has_right_child()

    @staticmethod
    def get_midpoint(a: float, b: float) -> float:
        return (a + b) / 2

    @staticmethod
    def node_to_string(node: 'GridNode') -> str:
        return "[{}, {}] (point {} of level {})".format(node.a, node.b, node.point, node.level)
