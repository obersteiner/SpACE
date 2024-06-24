import numpy as np
import abc, logging
from sparseSpACE.Integrator import *
import numpy.polynomial.legendre as legendre
import numpy.polynomial.hermite as hermite
import math
from math import isclose, isinf
from sparseSpACE.BasisFunctions import *
from sparseSpACE.Utils import *
from sparseSpACE.ComponentGridInfo import *
from typing import Callable, Tuple, Sequence
from sparseSpACE.Extrapolation import ExtrapolationGrid, SliceGrouping, SliceVersion, SliceContainerVersion, \
    BalancedExtrapolationGrid, GridVersion


# the grid class provides basic functionalities for an abstract grid
class Grid(object):

    def __init__(self, a, b, boundary:bool=True):
        self.boundary = boundary
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.dim = len(a)

    # integrates the grid on the specified area for function f
    def integrate(self, f: Callable[[Tuple[float, ...]], Sequence[float]], levelvec: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        if not self.is_global():
            self.setCurrentArea(start, end, levelvec)
        return self.integrator(f, self.levelToNumPoints(levelvec), start, end)

    # def integrate_point(self, f, levelvec, start, end, point):
    #    if not self.isGlobal():
    #        self.setCurrentArea(start, end, levelvec)
    #    return self.integrator.integrate_point(f, point)

    # returns if all grid components are nested
    def isNested(self) -> bool:
        return all([self.grids[d].is_nested() for d in range(len(self.grids))])

    # the default case is that a grid is not globally but only locally defined
    def is_global(self) -> bool:
        return False

    # this method can be used to generate a local grid for the given area and the specified number of points
    # typically the local grid is constructed for each area in the refinement graph during the algorithm
    @abc.abstractmethod
    def setCurrentArea(self, start: Sequence[float], end: Sequence[float], levelvec: Sequence[int]):
        pass

    # this method returns the actual coordinate of a point specified by the indexvector of the point
    # (e.g. in a 3 by 3 equdistant grid on the unit interval, the point [1,1] would have coordinate (0.5,0.5) and point [0,2] coordinate (0,1)
    # overwrite if necessary
    def getCoordinate(self, indexvector: Sequence[int]):
        position = np.empty(self.dim)
        for d in range(self.dim):
            position[d] = self.coordinate_array[d][indexvector[d]]
        return position

    # this method translates a point in an equidistant mesh of level self.levelvec in the Domain [0,1] to its corresponding index
    def getIndexTo1DCoordinate(self, coordinate: float, level: int) -> int:
        return coordinate * 2 ** level

    def point_not_zero(self, p: Sequence[float]) -> bool:
        # print(p, self.grid.boundary or not (self.point_on_boundary(p)))
        return self.boundary or not (self.point_on_boundary(p))

    def points_not_zero(self, points: Sequence[Sequence[float]]) -> Sequence[bool]:
        a = np.asarray(self.a)
        b = np.asarray(self.b)
        return np.logical_or(self.boundary*np.ones(len(points)), np.logical_not(np.logical_or(np.any(np.isclose(points, a), axis=1), np.any(np.isclose(points, b), axis=1))))

    def point_on_boundary(self, p: Sequence[float]) -> bool:
        # print("2",p, (p == self.a).any() or (p == self.b).any())
        return  any([isclose(c, self.a[d]) for d, c in enumerate(p)]) or any([isclose(c, self.b[d]) for d, c in enumerate(p)])

    def get_points_and_weights(self) -> Tuple[Sequence[Tuple[float, ...]], Sequence[float]]:
        return self.getPoints(), self.get_weights()

    def get_weights(self) -> Sequence[float]:
        #return np.asarray(list(self.getWeight(index) for index in get_cross_product_range(self.numPoints)))
        if get_cross_product_list(self.weights) == []:
            return []
        return np.asarray(np.prod(get_cross_product_list(self.weights), axis=1), dtype=np.float64)

    def get_num_points(self):
        return np.prod(self.numPoints)

    def get_mid_point(self, a: float, b: float, d: int) -> float:
        #if self.numPoints[d] == 1:
        #    return self.coordinate_array[d][0]
        #else:
        #    if self.numPoints[d] % 2 == 1:
        #        return sorted(self.coordinate_array[d])[int(self.numPoints[d]/2)]
        return self.grids[d].get_mid_point(a, b)

    # This method adjusts the grid to the given area defined by start and end. We initialize the weights and the grids
    # for the given levelvector.
    def setCurrentArea(self, start: Sequence[float], end: Sequence[float], levelvec: Sequence[int]) -> None:
        self.start = start if start is not None else self.a
        self.end = end if end is not None else self.b
        assert self.dim == len(self.start)
        assert self.dim == len(self.end)
        self.levelvec = levelvec
        for d in range(self.dim):
            self.grids[d].set_current_area(self.start[d], self.end[d], self.levelvec[d])
        self.numPoints = self.levelToNumPoints(levelvec)
        self.coordinate_array = [self.grids[d].coords for d in range(self.dim)]
        self.coordinate_array_with_boundary = [self.grids[d].coords_with_boundary for d in range(self.dim)]
        self.weights = [self.grids[d].weights for d in range(self.dim)]
        self.length = np.array(self.end) - np.array(self.start)
        self.numPointsWithBoundary = self.levelToNumPointsWithBoundary(levelvec)
        # prepare coordinates and weights

    # this method returns the number of points in the grid that correspond to the specified levelvector
    def levelToNumPoints(self, levelvec: Sequence[int]) -> Sequence[int]:
        numPoints = np.zeros(len(levelvec), dtype=int)
        for d in range(len(levelvec)):
            numPoints[d] = self.grids[d].level_to_num_points_1d(levelvec[d])
        return numPoints

    # this method returns the number of points in the grid that correspond to the specified levelvector
    def levelToNumPointsWithBoundary(self, levelvec: Sequence[int]) -> Sequence[int]:
        numPoints = np.zeros(len(levelvec), dtype=int)
        for d in range(len(levelvec)):
            boundary_save = self.grids[d].boundary
            self.grids[d].boundary = True
            numPoints[d] = self.grids[d].level_to_num_points_1d(levelvec[d])
            self.grids[d].boundary = boundary_save

        return numPoints

    # this method returns all the coordinate tuples of all points in the grid
    def getPoints(self) -> Sequence[Tuple[float, ...]]:
        return get_cross_product_list(self.coordinate_array)

    # this method returns all the index tuples of all points in the grid
    def get_indexlist(self) -> Sequence[Tuple[float, ...]]:
        return get_cross_product_list([range(1, n + 1) for n in self.levelToNumPoints(self.levelvec)])

    # this method returns the quadrature weight for the point specified by the indexvector
    def getWeight(self, indexvector: Sequence[int]) -> float:
        weight = 1
        for d in range(self.dim):
            weight *= self.weights[d][indexvector[d]]
        return weight

    # this mehtod returns if the grid is a high order (>1) grid
    def is_high_order_grid(self) -> bool:
        return False

    # this method returns the boundary flags for each dimension
    def get_boundaries(self) -> Sequence[bool]:
        boundaries = np.empty(len(self.grids), dtype=bool)
        for d, grid in enumerate(self.grids):
            boundaries[d] = grid.boundary
        return boundaries

    # this method sets the boundary flags for each dimension
    def set_boundaries(self, boundaries):
        for d, grid in enumerate(self.grids):
            grid.boundary = boundaries[d]

    # this method returns the point coordinates array for all dimension (list of vectors)
    def get_coordinates(self) -> Sequence[Sequence[float]]:
        return self.coordinate_array

    # this method returns the point coordinates in the specified dimension d
    def get_coordinates_dim(self, d) -> Sequence[float]:
        return self.coordinate_array[d]

    # this method checks if the quadrature rule is accurate up to specified degree in the interval [a,b]. the grid_1D and weights_1D
    # define the quadrature rule (i.e. point coordinates and their weights)
    def check_quality_of_quadrature_rule(self, a: float, b: float, degree: int, grid_1D: Sequence[float], weights_1D: Sequence[float]) -> bool:
        tol = 10 ** -14
        bad_approximation = False #(sum(abs(weights_1D)) - (b - a)) / (b - a) > tol  # (tol/(10**-15))**(1/self.dim)
        # if bad_approximation:
        # print("Too much negative entries, error:",
        #     (sum(abs(weights_1D)) - (b - a)) / (b - a))

        if not bad_approximation:
            for i in range(degree + 1):
                real_moment = b ** (i + 1) / (i + 1) - a ** (i + 1) / (i + 1)
                if abs(sum([grid_1D[j] ** i * weights_1D[j] for j in range(len(grid_1D))]) - real_moment) / abs(
                        real_moment) > tol:
                    print("Bad approximation for degree",i, "with error", abs(sum([grid_1D[j]**i * weights_1D[j] for j in range(len(grid_1D))]) - real_moment) / abs(real_moment) )
                    bad_approximation = True
                    break

        tolerance_lower = 0
        #print(weights_1D, all([w > tolerance_lower for w in weights_1D]), d)
        #return sum(weights_1D) < (b-a) * 2
        return bad_approximation

    # This method checks if the quadrature rule is stable (i.e. all weights are >= 0)
    def check_stability_of_quadrature_rule(self, weights_1D: Sequence[float]) -> bool:
        return not(all([w >= 0 for w in weights_1D]))

    @abc.abstractmethod
    def initialize_grid(self):
        pass

from scipy.optimize import fmin
from scipy.special import eval_hermitenorm, eval_sh_legendre


# This abstract class defines the common structure and interface of all 1D Grids
class Grid1d(object):
    def __init__(self, a: float=None, b: float=None, boundary: bool=True):
        self.boundary = boundary
        self.a = a
        self.b = b

    def set_current_area(self, start: float, end: float, level: int) -> None:
        self.start = start
        self.end = end
        self.level = level
        self.num_points = self.level_to_num_points_1d(level)
        boundary_save = self.boundary
        self.boundary = True
        self.num_points_with_boundary = self.level_to_num_points_1d(level)
        self.boundary = boundary_save
        self.length = end - start
        # coords, weights = self.get_1d_points_and_weights(level)

        self.lowerBorder = int(0)
        self.upperBorder = self.num_points
        if not self.boundary and self.num_points < self.num_points_with_boundary:
            if isclose(start, self.a):
                self.lowerBorder = 1
            if isclose(end, self.b):
                self.upperBorder = self.num_points_with_boundary - 1
            else:
                self.upperBorder = self.num_points_with_boundary
        # equidistant spacing; only valid for equidistant grids
        if self.num_points_with_boundary == 1:
            self.spacing = None
        else:
            self.spacing = (end - start) / (self.num_points_with_boundary - 1)
        coordsD, weightsD = self.get_1d_points_and_weights()
        self.coords = np.asarray(coordsD)
        self.weights = np.asarray(weightsD)
        if self.boundary == False:
            self.coords_with_boundary = np.array([self.a] + list(coordsD) + [self.b])
        else:
            self.coords_with_boundary = np.array(self.coords)

    @abc.abstractmethod
    def get_1d_points_and_weights(self) -> Sequence[float]:
        pass

    def get_1D_level_weights(self) -> Sequence[float]:
        return [self.get_1d_weight(i) for i in range(self.num_points)]

    @abc.abstractmethod
    def get_1d_weight(self, index: int) -> float:
        pass

    def get_mid_point(self, a: float, b: float) -> float:
        return (a + b) / 2.0

    # the default case is that a grid is nested; overwrite this if not nested!
    def is_nested(self) -> bool:
        return True

    def is_high_order_grid(self) -> bool:
        return False


# This grid can be used to define a grid consisting of arbitrary 1D Grids
class MixedGrid(Grid):
    def __init__(self, a: Sequence[float], b: Sequence[float], grids: Sequence[Grid1d], boundary: bool=None, integrator: IntegratorBase=None):
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.dim = len(a)
        if integrator is None:
            self.integrator = IntegratorArbitraryGridScalarProduct(self)
        else:
            if integrator == 'old':
                self.integrator = IntegratorArbitraryGrid(self)
            else:
                assert False
        assert (len(grids) == self.dim)
        self.grids = grids
        self.boundary = all([grid.boundary for grid in grids])

    def is_high_order_grid(self):
        return any([grid.is_high_order_grid() for grid in self.grids])


class BasisGrid(Grid):
    # integrates the grid on the specified area for function f
    def integrate(self, f: Callable[[Tuple[float, ...]], Sequence[float]], levelvec: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        if not self.is_global():
            self.setCurrentArea(start, end, levelvec)
        integral = self.integrator(f, self.levelToNumPoints(levelvec), start, end)
        self.surplus_values[tuple((tuple(start), tuple(end), tuple(levelvec)))] = self.integrator.get_surplusses()
        return integral

    def interpolate_grid(self, grid_points_for_dims: Sequence[Sequence[float]], start: Sequence[float], end: Sequence[float], levelvec: Sequence[int]) -> Sequence[Sequence[float]]:
        surplusses = self.surplus_values[tuple((tuple(start), tuple(end), tuple(levelvec)))]
        num_points = np.prod([len(grid_d) for grid_d in grid_points_for_dims])
        num_points_grid = [len(grid_d) for grid_d in grid_points_for_dims]
        results = np.zeros((num_points, np.shape(surplusses)[0]))
        evaluations = np.empty(self.dim, dtype=object)
        for d in range(self.dim):
            points_d = grid_points_for_dims[d]
            evaluations1D = np.zeros((len(points_d), (len(self.splines[d]))))
            for i, spline in enumerate(self.grids[d].splines):
                for j, p in enumerate(points_d):
                    evaluations1D[j, i] = spline(p)
            evaluations[d] = evaluations1D

        pointIndexList = get_cross_product_range(num_points_grid)
        for n, p in enumerate(pointIndexList):
            indexList = get_cross_product_range(self.numPoints)
            for i, index in enumerate(indexList):
                intermediate_result = np.array(surplusses[:,i])
                for d in range(self.dim):
                    for j in range(np.shape(surplusses)[0]):
                        intermediate_result *= evaluations[d][p[d], index[d]]
                results[n,:] += intermediate_result
        return results

    def interpolate(self, evaluation_points: Sequence[Tuple[float, ...]], start: Sequence[float], end: Sequence[float], levelvec: Sequence[int]) -> Sequence[Sequence[float]]:
        surplusses = self.surplus_values[tuple((tuple(start), tuple(end), tuple(levelvec)))]
        results = np.zeros((len(evaluation_points), np.shape(surplusses)[0]))
        evaluations = np.empty(self.dim, dtype=object)
        for d in range(self.dim):
            points_d = np.array([evaluation_point[d] for evaluation_point in evaluation_points])
            evaluations1D = np.zeros((len(evaluation_points), len(self.grids[d].splines)))
            for i, basis in enumerate(self.grids[d].splines):
                for j, p in enumerate(points_d):
                    evaluations1D[j, i] = basis(p)
            evaluations[d] = evaluations1D
        indexList = get_cross_product_range(self.numPoints)
        for i, index in enumerate(indexList):
            intermediate_result = np.ones((len(evaluation_points), np.shape(surplusses)[0]))
            intermediate_result *= np.array(surplusses[:,i])
            for d in range(self.dim):
                for j in range(np.shape(surplusses)[0]):
                    intermediate_result[:, j] *= evaluations[d][:, index[d]]
            results[:,:] += intermediate_result
        return results

    def get_basis(self, d: int, index: int):
        return self.grids[d].splines[index]


class LagrangeGrid(BasisGrid):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, p: int=3, modified_basis: bool=False):
        self.boundary = boundary
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.dim = len(a)
        self.integrator = IntegratorHierarchicalBasisFunctions(self)
        self.grids = [LagrangeGrid1D(a=a[d], b=b[d], boundary=self.boundary, p=p, modified_basis=modified_basis) for d in range(self.dim)]
        self.p = p
        self.modified_basis = modified_basis
        self.surplus_values = {}
        assert not boundary or not modified_basis

    def is_high_order_grid(self) -> bool:
        return self.p > 1


class LagrangeGrid1D(Grid1d):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, p: int=3, modified_basis: bool=False):
        super().__init__(a=a, b=b, boundary=boundary)
        self.p = p  # max order of lagrange polynomials
        self.coords_gauss, self.weights_gauss = legendre.leggauss(int(self.p / 2) + 1)
        self.modified_basis = modified_basis
        assert not boundary or not modified_basis

    def level_to_num_points_1d(self, level: int) -> int:
        return 2 ** level + 1 - (1 if not self.boundary else 0) * (
                int(1 if isclose(self.start, self.a) else 0) + int(1 if self.end == self.b else 0))

    def get_1d_points_and_weights(self):
        coordsD = self.get_1D_level_points(self.level, self.start, self.end)
        # print(coordsD)
        weightsD = np.array(self.compute_1D_quad_weights(coordsD))
        return coordsD, weightsD

    def get_1D_level_points(self, level: int, a: float, b: float):
        N = 2 ** level + 1  # inner points including boundary points
        # h = (b - a) / (N - 1)
        # N_knot = N + 2 * self.p
        # self.knots = np.linspace(a - self.p * h, b + self.p * h, N_knot)
        return list(np.linspace(a, b, N)[self.lowerBorder: self.upperBorder])

    def compute_1D_quad_weights(self, grid_1D):
        self.splines = np.empty(2 ** self.level + 1, dtype=object)
        grid_levels_1D = np.zeros(len(grid_1D), dtype=int)
        for l in range(1, self.level + 1):
            offset = 2**(self.level - l)
            for i in range(offset, len(grid_levels_1D), 2*offset):
                grid_levels_1D[i] = l
        #print(grid_levels_1D)
        max_level = max(grid_levels_1D)
        grid_levels_1D = np.asarray(grid_levels_1D)
        level_coordinate_array = [[] for l in range(max_level+1)]
        for i, p in enumerate(grid_1D):
            level_coordinate_array[grid_levels_1D[i]].append(p)

        #print(level_coordinate_array)
        # level = 0
        weights = np.zeros(len(grid_1D))
        starting_level = 0 if self.boundary else 1
        parents = {}
        for l in range(starting_level, max_level + 1):
            h = (self.end - self.start) / 2 ** l
            # calculate knots for spline construction
            #print(knots, "level", l, "dimension", d)
            #iterate over levels and add hierarchical B-Splines
            if l == 0:
                start = 0
                end = 2
                step = 1
            else:
                start = 1
                end = 2 ** l
                step = 2
            for i in range(len(level_coordinate_array[l])):
                # i is the index of the basis function
                # point associated to the basis (spline) with index i
                x_basis = level_coordinate_array[l][i]
                if l == 0:
                    knots = list(level_coordinate_array[l])
                elif l == 1:
                    knots = list(sorted(level_coordinate_array[0] + level_coordinate_array[1]))
                else:
                    parent = self.get_parent(x_basis, grid_1D, grid_levels_1D)
                    knots = list(sorted(parents[parent] + [x_basis]))
                parents[x_basis] = knots
                if len(knots) > self.p + 1:
                    index_x = knots.index(x_basis)
                    left = index_x
                    right = len(knots) - index_x - 1
                    if left < int((self.p+1)/2):
                        knots = knots[: self.p+1]
                    elif right < int(self.p/2):
                        knots = knots[-(self.p+1):]
                    else:
                        knots = knots[index_x - int((self.p + 1)/2): index_x + int(self.p/2) + 1]
                    assert len(knots) == self.p + 1
                # if this knot is part of the grid insert it
                #print(i, x_basis, grid_1D, grid_levels_1D, l)
                if self.modified_basis:
                    assert False
                    spline = LagrangeBasisRestricted(self.p, knots.index(x_basis), knots, a, b)
                else:
                    spline = LagrangeBasisRestricted(self.p, knots.index(x_basis), knots)
                index = grid_1D.index(x_basis)
                self.splines[index] = spline
                weights[index] = spline.get_integral(self.start, self.end, self.coords_gauss, self.weights_gauss)
        if not self.boundary:
            self.splines = self.splines[1:-1]
        for spline in self.splines:
            assert spline is not None
        return weights

    def is_high_order_grid(self) -> bool:
        return self.p > 1

    def get_parent(self, p: int, grid_1D, grid_levels):
        index_p = grid_1D.index(p)
        level_p = grid_levels[index_p]
        found_parent=False
        for i in reversed(range(index_p)):
            if grid_levels[i] == level_p - 1:
                return grid_1D[i]
            elif grid_levels[i] < level_p - 1:
                break
        for i in range(index_p+1, len(grid_1D)):
            if grid_levels[i] == level_p - 1:
                return grid_1D[i]
            elif grid_levels[i] < level_p - 1:
                break
        assert False


class BSplineGrid(BasisGrid):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, p: int=3, modified_basis: bool=False):
        self.boundary = boundary
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.dim = len(a)
        self.integrator = IntegratorHierarchicalBasisFunctions(self)
        self.grids = [BSplineGrid1D(a=a[d], b=b[d], boundary=self.boundary, p=p, modified_basis=modified_basis) for d in range(self.dim)]
        self.p = p
        self.modified_basis = modified_basis
        self.surplus_values = {}
        assert not boundary or not modified_basis

    def is_high_order_grid(self) -> bool:
        return self.p > 1

class BSplineGrid1D(Grid1d):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, p: int=3, modified_basis: bool=False):
        super().__init__(a=a, b=b, boundary=boundary)
        self.p = p #spline order
        assert p % 2 == 1
        self.coords_gauss, self.weights_gauss = legendre.leggauss(int(self.p/2) + 1)
        self.modified_basis = modified_basis
        assert not boundary or not modified_basis

    def level_to_num_points_1d(self, level: int):
        return 2 ** level + 1 - (1 if not self.boundary else 0) * (
                int(1 if isclose(self.start, self.a) else 0) + int(1 if self.end == self.b else 0))

    def get_1d_points_and_weights(self):
        coordsD = self.get_1D_level_points(self.level, self.start, self.end)
        # print(coordsD)
        weightsD = np.array(self.compute_1D_quad_weights(coordsD))
        return coordsD, weightsD

    def get_1D_level_points(self, level, a: float, b: float):
        N = 2**level + 1 #inner points including boundary points
        #h = (b - a) / (N - 1)
        #N_knot = N + 2 * self.p
        #self.knots = np.linspace(a - self.p * h, b + self.p * h, N_knot)
        return np.linspace(a, b, N)[self.lowerBorder: self.upperBorder]

    def compute_1D_quad_weights(self, grid_1D):
        self.start = np.array(self.start)
        self.end = np.array(self.end)

        # level = 0
        weights = np.zeros(2 ** self.level + 1)
        self.splines = np.empty(2 ** self.level + 1, dtype=object)
        l = self.level
        stride = 2 ** (self.level - l)
        h = (self.end - self.start) / 2 ** l
        if l < log2(self.p + 1):
            knots = np.linspace(self.start, self.end, 2 ** l + 1)
        else:
            knots = np.array([self.start + i * h for i in range(-self.p, 2 ** l + self.p + 1) if
                              i <= 0 or (self.p + 1) / 2 <= i <= 2 ** l - (self.p + 1) / 2 or i >= 2 ** l])
        #print(knots)
        for i in range(self.lowerBorder, self.upperBorder):
            if self.modified_basis:
                spline = HierarchicalNotAKnotBSplineModified(self.p, i, l, knots, self.start, self.end)
            else:
                spline = HierarchicalNotAKnotBSpline(self.p, i, l, knots)
            self.splines[i * stride] = spline
            weights[i * stride] = spline.get_integral(self.start, self.end, self.coords_gauss, self.weights_gauss)
        #print(self.get_1D_level_points(self.level, self.start, self.end), knots)
        if not self.boundary:
            self.splines = self.splines[self.lowerBorder: self.upperBorder]
        return weights[self.lowerBorder:self.upperBorder]

    def compute_1D_quad_weights(self, grid_1D):
        max_level = self.level

        # print(level_coordinate_array)
        # level = 0
        weights = np.zeros(2 ** self.level + 1)
        self.splines = np.empty(2 ** self.level + 1, dtype=object)
        starting_level = 0 if self.boundary else 1
        for l in range(starting_level, max_level + 1):
            level_coordinates = np.linspace(self.start, self.end, 2 ** l + 1)
            h = (self.end - self.start) / 2 ** l
            stride = 2 ** (self.level - l)
            # calculate knots for spline construction
            if l < log2(self.p + 1):
                knots = np.linspace(self.start, self.end, 2 ** l + 1)
            else:
                knots = np.array([self.start + i * h for i in range(-self.p, 2 ** l + self.p + 1) if
                                  i <= 0 or (self.p + 1) / 2 <= i <= 2 ** l - (self.p + 1) / 2 or i >= 2 ** l])
            # print(knots, "level", l, "dimension", d)
            # iterate over levels and add hierarchical B-Splines
            if l == 0:
                start = 0
                end = 2
                step = 1
            else:
                start = 1
                end = 2 ** l
                step = 2
            for i in range(start, end, step):
                # if this knot is part of the grid insert it
                # print(i, x_basis, grid_1D, grid_levels_1D, l)
                if self.modified_basis:
                    spline = HierarchicalNotAKnotBSplineModified(self.p, i, l, knots, self.start, self.end)
                else:
                    spline = HierarchicalNotAKnotBSpline(self.p, i, l, knots)
                self.splines[stride * i] = spline
                weights[stride * i] = spline.get_integral(self.start, self.end, self.coords_gauss, self.weights_gauss)
        if not self.boundary:
            self.splines = self.splines[self.lowerBorder: self.upperBorder]
        return weights[self.lowerBorder:self.upperBorder]

    def get_hierarchical_splines(self, grid_1D, l: int, weights):
        stride = 2**(self.level - l)
        h = (self.b - self.a) / 2**l
        if l < log2(self.p + 1):
            knots = np.linspace(self.a, self.b, 2**l + 1)
        else:
            knots = np.array([self.a + i * h for i in range(-self.p,2**l + self.p + 1) if  i <= 0 or (self.p + 1)/2 <= i <= 2**l - (self.p+1)/2 or i >= 2**l])
        #print(knots)
        if l == 0:
            start = 0
            end = 2
            step = 1
        else:
            start = 1
            end = 2**l
            step = 2
        for i in range(start, end, step):
            spline = HierarchicalNotAKnotBSpline(self.p, i, l, knots)
            self.splines[i*stride] = spline
            weights[i*stride] = spline.get_integral(self.a, self.b, self.coords_gauss, self.weights_gauss)

    def is_high_order_grid(self):
        return self.p > 1

# this class generates a Leja grid which constructs 1D Leja grid structures
# and constructs the tensorized grid according to the levelvector
class LejaGrid(Grid):
    def __init__(self, a, b, boundary=True, integrator=None):
        self.boundary = boundary
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.dim = len(a)
        if integrator is None:
            self.integrator = IntegratorArbitraryGridScalarProduct(self)
        else:
            if integrator == 'old':
                self.integrator = IntegratorArbitraryGrid(self)
            else:
                assert False
        self.linear_growth_factor = 2
        self.grids = [LejaGrid1D(a=a[d], b=b[d], boundary=self.boundary) for d in range(self.dim)]


class LejaGrid1D(Grid1d):
    def __init__(self, a, b, boundary):
        super().__init__(a=a, b=b, boundary=boundary)
        self.linear_growth_factor = 2

    def get_1d_points_and_weights(self):
        coordsD = self.get_1D_level_points(self.level, 0, 1)
        weightsD = np.array(self.compute_1D_quad_weights(coordsD)) * self.length
        assert len(coordsD) == len(weightsD)
        coordsD = np.array(coordsD)
        coordsD *= self.length
        coordsD += self.start
        return coordsD, weightsD

    def level_to_num_points_1d(self, level):
        if level == 0:
            numPoints = 2
        else:
            numPoints = self.linear_growth_factor * (level + 1) - 1
        if not self.boundary:
            numPoints -= 2
        return numPoints

    def compute_1D_quad_weights(self, grid_1D):
        N = len(grid_1D)
        V = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                V[i, j] = eval_sh_legendre(j, grid_1D[i]) * np.sqrt(2 * j + 1)

        weights = np.linalg.inv(V)[0, :]
        if len(grid_1D) == self.num_points:
            return weights
        return weights[self.lowerBorder:self.upperBorder]

    def __minimize_function(self, func, a, b):

        guess = a + (b - a) / 2.0
        f_min = fmin(func, guess, xtol=1e-14, maxiter=10000, disp=False)[-1]

        return f_min

    def __get_lleja_poly(self, x, sorted_points, a, b, weightFunction=lambda x: 1.0 / s):

        if (x < a or x > b):
            return -1

        poly = 1.0
        for i in range(len(sorted_points)):
            poly *= np.abs(x - sorted_points[i])

        poly *= weightFunction(x)

        return poly

    def __get_neg_lleja_poly(self, x, sorted_points, a, b, weightFunction=lambda x: 1.0):

        return (-1.0) * self.__get_lleja_poly(x, sorted_points, a, b, weightFunction)

    def __get_starting_point(self, a, b, weightFunction=lambda x: 1.0):

        neg_weight = lambda x: -weightFunction(x)
        starting_point = self.__minimize_function(neg_weight, a, b)

        return starting_point

    def get_1D_level_points(self, curr_level, left_bound, right_bound, weightFunction=lambda x: 1.0, eps=1e-14):

        sorted_points = []
        unsorted_points = []
        no_points = self.level_to_num_points_1d(curr_level)
        if not self.boundary:
            no_points += 2
        if no_points == 2:
            return np.array([left_bound, right_bound], dtype=np.float64)
        starting_point = self.__get_starting_point(left_bound, right_bound, weightFunction)
        sorted_points.append(starting_point)
        unsorted_points.append(starting_point)

        for point in range(1, no_points):
            x_val = []
            y_val = []

            a = 0.
            b = left_bound
            for i in range(len(sorted_points) + 1):
                sorted_points = sorted(sorted_points)

                a = b
                if i < len(sorted_points):
                    b = sorted_points[i]
                else:
                    b = right_bound

                x_min = (a + b) / 2.0
                y_min = 0.0

                if np.abs(b - a) > eps:
                    wlleja_func = lambda x: self.__get_neg_lleja_poly(x, sorted_points, a, b, weightFunction)
                    x_min = self.__minimize_function(wlleja_func, a, b)

                    x_val.append(x_min)
                    y_min = wlleja_func(x_val[-1])
                    y_val.append(y_min)
                else:
                    x_val.append(x_min)
                    y_val.append(y_min)

            wlleja_point = x_val[y_val.index(np.min(y_val))]
            sorted_points.append(wlleja_point)
            unsorted_points.append(wlleja_point)

        sorted_points = np.array(sorted(unsorted_points), dtype=np.float64)
        sorted_points[0] = left_bound
        sorted_points[-1] = right_bound

        return sorted_points[self.lowerBorder:self.upperBorder]


# this class provides an equdistant mesh and uses the trapezoidal rule compute the quadrature
class TrapezoidalGrid(Grid):
    def __init__(self, a, b, boundary=True, integrator=None, modified_basis=False):
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.dim = len(a)
        self.boundary = boundary
        self.modified_basis = modified_basis
        if integrator is None:
            self.integrator = IntegratorArbitraryGridScalarProduct(self)
        else:
            if integrator == 'old':
                self.integrator = IntegratorArbitraryGrid(self)
            else:
                assert False
        self.grids = [TrapezoidalGrid1D(a=a[d], b=b[d], boundary=self.boundary, modified_basis=modified_basis) for d in range(self.dim)]


class TrapezoidalGrid1D(Grid1d):

    def __init__(self, a=None, b=None, boundary=True, modified_basis=False):
        super().__init__(a, b, boundary)
        self.modified_basis = modified_basis
        assert (not self.boundary) or (not modified_basis)

    def level_to_num_points_1d(self, level):
        return 2 ** level + 1 - (1 if not self.boundary else 0) * (
                int(1 if isclose(self.start, self.a) else 0) + int(1 if self.end == self.b else 0))

    def get_1d_points_and_weights(self):
        coordsD = self.get_1D_level_points()
        weightsD = self.get_1D_level_weights()
        # print(coordsD, weightsD, self.lowerBorder, self.upperBorder, self.a, self.b, self.start, self.end)
        return coordsD, weightsD

    def get_1D_level_points(self):
        if not self.boundary and self.num_points == 1:
            return np.array([(self.end+self.start)/2.0])
        return np.linspace(self.start, self.end, self.num_points_with_boundary)[self.lowerBorder:self.upperBorder]

    def get_1d_weight(self, index):
        if self.modified_basis:
            if self.num_points == 1:
                return self.end - self.start
            elif self.num_points == 2:
                if self.lowerBorder == 1:
                    if index == 0:
                        return self.end - self.start
                    else:
                        return 0
                elif self.upperBorder == self.num_points_with_boundary - 1:
                    if index == 1:
                        return self.end - self.start
                    else:
                        return 0
                else:
                    return self.weight_composite_trapezoidal(index)
            else:
                if index == 0 and self.lowerBorder == 1:
                    return 2 * self.spacing
                elif index == 1 and self.lowerBorder == 1:
                    if self.num_points == 3 and self.upperBorder == self.num_points_with_boundary - 1:
                        return 0
                    else:
                        return self.weight_composite_trapezoidal(index) * 0.5
                elif index == self.num_points - 1 and self.upperBorder == self.num_points_with_boundary - 1:
                    return 2 * self.spacing
                elif index == self.num_points - 2 and self.upperBorder == self.num_points_with_boundary - 1:
                    return self.weight_composite_trapezoidal(index) * 0.5
                else:
                    return self.weight_composite_trapezoidal(index)
        else:
            return self.weight_composite_trapezoidal(index)

    def weight_composite_trapezoidal(self, index):
        if not self.boundary and self.num_points == 1:
            return self.spacing
        return self.spacing * (0.5 if index + self.lowerBorder == 0 or index + self.lowerBorder == \
                                      self.num_points_with_boundary - 1 else 1)

# this class provides an equdistant mesh and uses the trapezoidal rule compute the quadrature
class SimpsonGrid(Grid):
    def __init__(self, a, b, boundary=True, integrator=None):
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.dim = len(a)
        self.boundary = boundary
        if integrator is None:
            self.integrator = IntegratorArbitraryGridScalarProduct(self)
        else:
            if integrator == 'old':
                self.integrator = IntegratorArbitraryGrid(self)
            else:
                assert False
        self.grids = [SimpsonGrid1D(a=a[d], b=b[d], boundary=self.boundary) for d in range(self.dim)]


class SimpsonGrid1D(TrapezoidalGrid1D):

    def __init__(self, a=None, b=None, boundary=True):
        super().__init__(a, b, boundary)

    def get_1D_level_weights(self) -> Sequence[float]:
        if self.num_points_with_boundary < 3:
            return [self.weight_composite_trapezoidal(i) for i in range(self.num_points)]
        weights = np.ones(self.num_points_with_boundary) * self.spacing * 1.0/3.0
        weights[1:-1:2] *= 4
        weights[2:-1:2] *= 2
        if not self.boundary:
            weights = weights[1:-1]
        return weights

    def get_1d_weight(self, index):
        if self.num_points_with_boundary < 3:
            return self.weight_composite_trapezoidal(index)
        weight = self.spacing * 1.0/3.0
        if index % 2 == 0:
            weight *= 6
        elif index != 0 or index != self.num_points_with_boundary - 1:
            weight *= 2
        return weight

# this class generates a grid according to the roots of Chebyshev points and applies a Clenshaw Curtis quadrature
# the formulas are taken from: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.33.3141&rep=rep1&type=pdf
class ClenshawCurtisGrid(Grid):
    def __init__(self, a, b, boundary=True, integrator=None):
        self.a = a
        self.b = b
        assert len(a) == len(b)
        self.boundary = boundary
        self.dim = len(a)
        if integrator is None:
            self.integrator = IntegratorArbitraryGridScalarProduct(self)
        else:
            if integrator == 'old':
                self.integrator = IntegratorArbitraryGrid(self)
            else:
                assert False
        self.grids = [ClenshawCurtisGrid1D(a=a[d], b=b[d], boundary=self.boundary) for d in range(self.dim)]

    def is_high_order_grid(self):
        return True

class ClenshawCurtisGrid1D(Grid1d):
    def level_to_num_points_1d(self, level):
        return 2 ** level + 1 - int(not self.boundary) * (int(self.start == 0) + int(self.end == 1))

    def get_1d_points_and_weights(self):
        coordsD = self.get_1D_level_points()
        weightsD = self.get_1D_level_weights()
        return coordsD, weightsD

    def get_1D_level_points(self):
        coordinates = np.empty(self.num_points)
        for i in range(self.num_points):
            coordinates[i] = self.start + (
                    1 - math.cos(math.pi * (i + self.lowerBorder) / (self.num_points_with_boundary - 1))) * \
                             self.length / 2
        return coordinates

    def get_1d_weight(self, index):
        weight = self.length / 2.0
        if self.num_points_with_boundary > 2:
            if index == 0 or index == self.num_points_with_boundary - 1:
                weight_factor = 1.0 / ((self.num_points_with_boundary - 2) * self.num_points_with_boundary)
            else:
                weight_factor = 0.0
                for j in range(1, math.floor((self.num_points_with_boundary - 1) / 2.0) + 1):
                    term = 1.0 / (1.0 - 4 * j * j) * math.cos(
                        2 * math.pi * index * j / (self.num_points_with_boundary - 1))
                    if j == math.floor((self.num_points_with_boundary - 1) / 2.0):
                        term *= 0.5
                    weight_factor += term
                weight_factor = 2.0 / (self.num_points_with_boundary - 1) * (1 + 2 * weight_factor)
            weight *= weight_factor
        return weight

    def is_high_order_grid(self):
        return True

class GlobalGrid(Grid):

    def is_global(self) -> bool:
        return True

    def isNested(self) -> bool:
        return True

    def set_grid(self, grid_points: Sequence[Sequence[float]], grid_levels: Sequence[Sequence[int]]) -> None:
        assert len(grid_points) == len(grid_levels)
        for d in range(self.dim):
            assert len(grid_levels[d]) == len(grid_points[d])
        self.coordinate_array = []
        self.coordinate_array_with_boundary = []
        self.weights = []
        self.levels = []
        #print("Points and levels", grid_points, grid_levels)
        self.basis = [np.empty(len(grid_points[d]), dtype=object) for d in range(self.dim)]

        if self.boundary:
            self.numPoints = [len(grid_points[d]) for d in range(self.dim)]
            self.numPointsWithBoundary = [len(grid_points[d]) for d in range(self.dim)]
        else:
            self.numPoints = [len(grid_points[d]) - 2 for d in range(self.dim)]
            self.numPointsWithBoundary = [len(grid_points[d]) for d in range(self.dim)]

        for d in range(self.dim):
            # check if grid_points are sorted
            assert all(grid_points[d][i] <= grid_points[d][i + 1] for i in range(len(grid_points[d]) - 1))
            if self.boundary:
                coordsD = grid_points[d]
                weightsD = self.compute_1D_quad_weights(grid_points[d], self.a[d], self.b[d], d, grid_levels_1D=grid_levels[d])
                levelsD = grid_levels[d]
            else:
                coordsD = grid_points[d][1:-1]
                weightsD = self.compute_1D_quad_weights(grid_points[d], self.a[d], self.b[d], d, grid_levels_1D=grid_levels[d],)[1:-1]
                levelsD = grid_levels[d][1:-1]
            coords_d_with_boundary = grid_points[d]
            self.coordinate_array.append(coordsD)
            self.coordinate_array_with_boundary.append(coords_d_with_boundary)
            self.weights.append(weightsD)
            self.levels.append(levelsD)
            self.numPoints[d] = len(coordsD)
        self.coordinate_array = np.asarray(self.coordinate_array, dtype=object)
        self.coordinate_array_with_boundary = np.asarray(self.coordinate_array_with_boundary, dtype=object)
        self.weights = np.asarray(self.weights, dtype=object)
        self.levels = np.asarray(self.levels, dtype=object)

    def levelToNumPoints(self, levelvec: Sequence[int]) -> Sequence[int]:
        if hasattr(self, 'numPoints'):
            return self.numPoints
        else:
            return [2 ** levelvec[d] + 1 - int(self.boundary == False) * 2 for d in range(self.dim)]

    def getPoints(self) -> Sequence[Tuple[float, ...]]:
        return get_cross_product_list(self.coordinate_array)

    def getCoordinate(self, indexvector: Sequence[int]) -> Sequence[float]:
        position = np.zeros(self.dim)
        for d in range(self.dim):
            position[d] = self.coordinate_array[d][indexvector[d]]
        return position

    def getWeight(self, indexvector: Sequence[int]) -> float:
        weight = 1
        for d in range(self.dim):
            weight *= self.weights[d][indexvector[d]]
        return weight

    def get_coordinates(self) -> Sequence[Sequence[float]]:
        return self.coordinate_array

    def get_coordinates_dim(self, d: int) -> Sequence[float]:
        return self.coordinate_array[d]

    @abc.abstractmethod
    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        pass

    def get_mid_point(self, start: float, end: float, d: int=0) -> float:
        return 0.5 * (end + start)


class GlobalTrapezoidalGrid(GlobalGrid):
    def __init__(self, a, b, boundary=True, modified_basis=False):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGridScalarProduct(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.modified_basis = modified_basis
        assert not(modified_basis) or not(boundary)

    @staticmethod
    def compute_weights(grid_1D: Sequence[float], a: float, b: float, modified_basis: bool) -> Sequence[float]:
        weights = np.zeros(len(grid_1D))
        if modified_basis and len(grid_1D) == 3:
            weights[1] = b - a
        elif modified_basis and len(grid_1D) == 4:
            weights[2] = (b**2/2 - b *grid_1D[1] - a**2/2 + a * grid_1D[1]) / (grid_1D[2] - grid_1D[1])
            weights[1] = -1 *weights[2] + b - a
        else:
            for i in range(0, len(grid_1D)):
                if i > 0:
                    if modified_basis and i == 1:
                        #weights[i] += (grid_1D[i+1] - grid_1D[i - 1])
                        h_b = (grid_1D[i + 1] - grid_1D[i - 1])
                        h_a = (grid_1D[i + 1] - grid_1D[i])
                        #print(h_b, h_a, h_b ** 2 / (2 * h_a))
                        weights[i] += h_b ** 2 / (2 * h_a)
                    elif modified_basis and i == 2:
                        #pass
                        h_b = (grid_1D[i] - grid_1D[i - 2])
                        h_a = (grid_1D[i] - grid_1D[i - 1])
                        weights[i] += h_b - h_b ** 2 / (2 * h_a)
                    else:
                        if not (modified_basis and i == len(grid_1D) - 2):
                            weights[i] += 0.5 * (grid_1D[i] - grid_1D[i - 1])

                if i < len(grid_1D) - 1:
                    if modified_basis and i == len(grid_1D) - 2:
                        if i > 1:
                            h_b = (grid_1D[i + 1] - grid_1D[i-1])
                            h_a = (grid_1D[i] - grid_1D[i - 1])
                            #weights[i] += (grid_1D[i + 1] - grid_1D[i-1])
                            weights[i] += h_b**2 / (2*h_a)

                    elif modified_basis and i == len(grid_1D) - 3:
                        #pass
                        if i > 1:
                            h_b = (grid_1D[i + 2] - grid_1D[i])
                            h_a = (grid_1D[i + 1] - grid_1D[i])
                            weights[i] += h_b - h_b**2/(2*h_a)
                    else:
                        if not(modified_basis and i == 1):
                            weights[i] += 0.5*(grid_1D[i + 1] - grid_1D[i])
            if modified_basis:
                weights[0] = 0.0
                weights[-1] = 0.0

        if modified_basis and not (b - a) * (1 - 10 ** -12) <= sum(weights[1:-1]) <= (
                b - a) * (1 + 10 ** -12):
            print(grid_1D, weights)
        if modified_basis:
            assert (b - a) * (1 - 10 ** -12) <= sum(weights[1:-1]) <= (
                b - a) * (1 + 10 ** -12)
        return weights

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        # print("Weights of GlobalTrapezoidalGrid: {}".format(grid_1D))
        # print("Levels of GlobalTrapezoidalGrid: {}".format(grid_levels_1D))

        return GlobalTrapezoidalGrid.compute_weights(grid_1D, a, b, self.modified_basis)


class GlobalTrapezoidalGridWeighted(GlobalTrapezoidalGrid):
    def __init__(self, a, b, uq_operation, boundary=True, modified_basis=False):
        super().__init__(a, b, boundary, modified_basis)
        self.distributions = uq_operation.get_distributions()

    @staticmethod
    def get_middle_weighted(a, b, cdf, ppf):
        # This is used to split a refinement object so that the new
        # ones have the same probability
        cdf_mid = 0.5 * (cdf(a) + cdf(b))
        mid = ppf(cdf_mid)
        assert isinstance(mid, float)
        # ~ mid = optimize.toms748(lambda x: distr.cdf(x) - cdf_mid,
            # ~ self.start, self.end, rtol=0.01)
        # ~ mid = max(min(mid, self.end), self.start)
        # ~ print("a,b,mid,cdf_mid", a, b, mid, cdf_mid, cdf(a), cdf(b))
        if not a < mid < b:
            print("Could not calculate the weighted middle")
            mid = 0.5 * (a + b)
            if not a < mid < b:
                print(f"Warning: Could not calculate the middle between {a} and {b}")
                if isinf(a):
                    mid = b - 10 ** -14
                elif isinf(b):
                    mid = a + 10 ** -14
        return mid

    def get_mid_point(self, a, b, d):
        distr = self.distributions[d]
        return self.get_middle_weighted(a, b, distr.cdf, distr.ppf)

    @staticmethod
    def compute_weights(grid_1D: Sequence[float], a: float, b: float, distribution, boundary: bool, modified_basis: bool) -> Sequence[float]:
        num_points = len(grid_1D)
        if num_points == 1:
            return [1.0]
        elif not boundary and num_points == 3:
            return [0.0, 1.0, 0.0]
        assert boundary or num_points > 3
        if modified_basis:
            # This only works with the Uniform distribution.
            weights = GlobalTrapezoidalGrid.compute_weights(grid_1D, a, b, True) / (b - a)
            assert isclose(sum(weights), 1.0), sum(weights)
            return weights
        weights = np.zeros(num_points)
        for i in range(num_points-1):
            x1 = grid_1D[i]
            x2 = grid_1D[i+1]
            # Calculate weights with the method of undetermined coefficients
            moment_0 = distribution.get_zeroth_moment(x1, x2)
            # w1 + w2 = moment_0
            moment_1 = distribution.get_first_moment(x1, x2)
            # w1 * x1 + w2 * x2 = moment_1
            if isinf(x1):
                # For infinite borders, L'Hospital leads to a simple w2
                w2 = moment_0
            elif isinf(x2):
                w2 = 0
            else:
                w2 = (moment_1 - moment_0 * x1) / (x2 - x1)
            w1 = moment_0 - w2
            # ~ print("dd", w1, w2, x1, x2, moment_0, moment_1)
            # Add them to the composite quadrature weights
            weights[i] += w1
            weights[i+1] += w2

        # Sometimes very small weights are negative instead of 0 due to
        # numerical errors
        for i in range(num_points):
            if weights[i] >= 0.0:
                continue
            assert -weights[i] < 10 ** -5, "calculated negative weight"
            weights[i] = 0.0

        if not boundary:
            # Remove weights from boundary points and normalize
            weights[0] = 0.0
            weights[-1] = 0.0
            f = 1.0 / sum(weights[1:-1])
            weights[1:-1] = [f * v for v in weights[1:-1]]
        else:
            if abs(1.0 - sum(weights)) > 10 ** -4:
                print("warning: 1D weights sum is {}".format(sum(weights)))

        return weights

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        distr = self.distributions[d]
        return self.compute_weights(grid_1D, a, b, distr, self.boundary, self.modified_basis)


class GlobalRombergGrid(GlobalGrid):
    def __init__(self, a, b, boundary=True, modified_basis=False,
                 do_cache=True,
                 grid_version=GridVersion.DEFAULT,
                 container_version=SliceContainerVersion.ROMBERG_DEFAULT,
                 slice_grouping=SliceGrouping.UNIT,
                 slice_version=SliceVersion.ROMBERG_DEFAULT):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGridScalarProduct(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.modified_basis = modified_basis
        self.do_cache = do_cache

        assert not(modified_basis)
        assert boundary

        self.grid_version = grid_version
        self.container_version = container_version
        self.slice_grouping = slice_grouping
        self.slice_version = slice_version

        self.weight_cache = {}

    def initialize_grid(self):
        # Reset weight cache
        if len(self.weight_cache) > 50:
            self.weight_cache = {}

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int,
                                grid_levels_1D: Sequence[int]=None) -> Sequence[float]:

        key = tuple([tuple(grid_1D), tuple(grid_levels_1D)])

        if self.do_cache and key in self.weight_cache:
            return self.weight_cache[key]

        romberg_grid = ExtrapolationGrid(grid_version=self.grid_version,
                                         container_version=self.container_version,
                                         slice_grouping=self.slice_grouping,
                                         slice_version=self.slice_version)

        romberg_grid.set_grid(grid_1D, grid_levels_1D)

        # TODO Adapt function to one dimensional grid stripe (1D), because the function is multidimensional in the general case
        # romberg_grid.set_function(self.function)

        weights = romberg_grid.get_weights()

        if self.do_cache:
            self.weight_cache[key] = weights

        return weights


class GlobalBalancedRombergGrid(GlobalGrid):
    def __init__(self, a, b, boundary=False, modified_basis=False):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGridScalarProduct(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.modified_basis = modified_basis

        assert not modified_basis
        assert not boundary

        self.weight_cache = {}

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int,
                                grid_levels_1D: Sequence[int]=None) -> Sequence[float]:

        grid = BalancedExtrapolationGrid()

        grid.set_grid(grid_1D, grid_levels_1D)
        weights = grid.get_weights()

        return weights


class GlobalBasisGrid(GlobalGrid):
    def get_basis(self, d: int, i: int) -> BasisFunction:
        return self.basis[d][i]

    def get_surplusses(self, levelvec: Sequence[int]) -> Sequence[Sequence[float]]:
        return self.surplus_values[tuple(levelvec)]

    # integrates the grid on the specified area for function f
    def integrate(self, f: Callable[[Tuple[float, ...]], Sequence[float]], levelvec: Sequence[int], start: Sequence[float], end: Sequence[float]) -> Sequence[float]:
        integral = self.integrator(f, self.levelToNumPoints(levelvec), start, end)
        self.surplus_values[tuple(levelvec)] = self.integrator.get_surplusses()
        return integral

    def interpolate(self, evaluation_points: Sequence[Tuple[float, ...]], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        levelvec = component_grid.levelvector
        surplusses = self.surplus_values[tuple(levelvec)]
        #print(surplusses)
        results = np.zeros((len(evaluation_points), np.shape(surplusses)[0]))
        #print(np.shape(results))
        #for n, evaluation_point in enumerate(evaluation_points):
        evaluations = np.empty(self.dim, dtype=object)
        for d in range(self.dim):
            points_d = np.array([evaluation_point[d] for evaluation_point in evaluation_points])
            evaluations1D = np.zeros((len(evaluation_points), len(self.basis[d])))
            for i, basis in enumerate(self.basis[d]):
                for j, p in enumerate(points_d):
                    evaluations1D[j, i] = basis(p)
            evaluations[d] = evaluations1D
        #print(evaluations)
        indexList = get_cross_product_range(self.numPoints)
        #print(indexList)
        for i, index in enumerate(indexList):
            intermediate_result = np.ones((len(evaluation_points), np.shape(surplusses)[0]))
            intermediate_result *= np.array(surplusses[:,i])
            for d in range(self.dim):
                for j in range(np.shape(surplusses)[0]):
                    intermediate_result[:, j] *= evaluations[d][:, index[d]]
            #print(intermediate_result)
            results[:,:] += intermediate_result
            #print(results[n,:])
        #print(results)
        return results

    def interpolate_grid(self, grid_points_for_dims: Sequence[Sequence[float]], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        levelvec = component_grid.levelvector
        surplusses = self.surplus_values[tuple(levelvec)]
        num_points = np.prod([len(grid_d) for grid_d in grid_points_for_dims])
        num_points_grid = [len(grid_d) for grid_d in grid_points_for_dims]
        #print(surplusses)
        results = np.zeros((num_points, np.shape(surplusses)[0]))
        #print(np.shape(results))
        #for n, evaluation_point in enumerate(evaluation_points):
        evaluations = np.empty(self.dim, dtype=object)
        for d in range(self.dim):
            points_d = grid_points_for_dims[d]
            evaluations1D = np.zeros((len(points_d), (len(self.basis[d]))))
            for i, basis in enumerate(self.basis[d]):
                for j, p in enumerate(points_d):
                    #print(p, spline(p))
                    evaluations1D[j, i] = basis(p)
            evaluations[d] = evaluations1D
        #print(evaluations)

        #print(indexList)
        pointIndexList = get_cross_product_range(num_points_grid)
        for n, p in enumerate(pointIndexList):
            indexList = get_cross_product_range(self.numPoints)
            for i, index in enumerate(indexList):
                intermediate_result = np.array(surplusses[:,i])
                for d in range(self.dim):
                    for j in range(np.shape(surplusses)[0]):
                        intermediate_result *= evaluations[d][p[d], index[d]]
                    #print(intermediate_result, surplusses[:,i], evaluations[d][p[d],:], grid[d], grid[d][p[d]], p[d])
                #print(intermediate_result)
                results[n,:] += intermediate_result
            #print(results[n,:])
        #print(results)
        #print(list(pointIndexList))

        #print(results)
        return results

    def plot_basis_functions(self, support_points=None, interpolation_point=None, function=None, filename=None):
        if self.basis is None:
            print("No basis functions constructed")
            return

        # Iterate over all dimensions and plot each basis
        for d in range(self.dim):
            plt.figure()

            if function is not None:
                evaluation_points = np.linspace(self.a[d], self.b[d], len(support_points) * 100)
                plt.plot(evaluation_points, [function([point]) for point in evaluation_points],
                         linestyle=':', color="black")

            # Plot basis functions
            for i, basis in enumerate(self.basis[d]):
                evaluation_points = np.linspace(self.a[d], self.b[d], len(basis.knots) * 100)
                plt.plot(evaluation_points, [basis(point) for point in evaluation_points])

            # Print interpolation grid
            plt.plot(support_points, np.zeros(len(support_points)), 'bo', markersize=6, color="dimgray")
            plt.title("Basis functions for the interpolation of point {}".format(interpolation_point))

            plt.show()

    def plot_basis_functions_with_evaluations(self, support_points, evaluation_point, function=None):
        # Iterate over all dimensions and plot each basis
        # plt.figure()

        for d in range(self.dim):
            # TODO plot dimension d
            columns = 2
            fig, axes = plt.subplots(figsize=(10, 15), nrows=math.ceil(len(support_points) / columns), ncols=columns)
            fig.subplots_adjust(hspace=0.5)
            fig.suptitle("Basis functions for the interpolation of the point {}".format(evaluation_point))

            axes = axes.ravel()

            a = support_points[0]
            b = support_points[-1]

            for i, basis in enumerate(self.basis[d]):
                if function is not None:
                    evaluation_points = np.linspace(a, b, len(support_points) * 100)
                    axes[i].plot(evaluation_points, [function([point]) for point in evaluation_points],
                             linestyle=':', color="black")

                # Plot basis functions
                for j, basis_f in enumerate(self.basis[d]):
                    evaluation_points = np.linspace(self.a[d], self.b[d], len(basis.knots) * 100)
                    axes[i].plot(evaluation_points, [basis_f(point) for point in evaluation_points])

                # Print interpolation grid
                axes[i].plot(support_points, np.zeros(len(support_points)), 'bo', markersize=4, color="black")

                # Plot evaluation point
                if evaluation_point is not None:
                    evaluation_point_value = basis(evaluation_point)
                    axes[i].plot(evaluation_point, evaluation_point_value, 'bo', markersize=4, color="red")
                    axes[i].plot([evaluation_point, evaluation_point], [0, evaluation_point_value], ':',
                                 color="red")

                axes[i].set_title("Evaluation using basis function of point {}".format(support_points[i]))

                if len(support_points) < 15:
                    axes[i].set_xticks(support_points)
                    axes[i].set_xticklabels(support_points, fontsize=9)

            # Clear unused axes
            for i in range(len(support_points), len(axes)):
                fig.delaxes(axes[i])

            fig.show()


class GlobalBSplineGrid(GlobalBasisGrid):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, modified_basis: bool=False, p: int=3, chebyshev=False):
        self.boundary = boundary
        self.integrator = IntegratorHierarchicalBasisFunctions(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.modified_basis = modified_basis
        assert p % 2 == 1
        self.p = p
        self.coords_gauss, self.weights_gauss = legendre.leggauss(int(self.p/2) + 1)
        assert not(modified_basis) or not(boundary)
        self.surplus_values = {}
        self.chebyshev = chebyshev

    def get_mid_point(self, start: float, end: float, d: int=0):
        if self.chebyshev:
            coordinate_start = math.acos(1 - 2 * (start - self.a[d]) / (self.b[d] - self.a [d])) / math.pi
            coordinate_end = math.acos(1 - 2 * (end - self.a[d]) / (self.b[d] - self.a[d])) / math.pi
            coordinate = (coordinate_start + coordinate_end) / 2
            #print("Start", coordinate)
            coordinate_normalized = (coordinate - self.a[d]) / (self.b[d] - self.a[d])
            coordinate = self.a[d] + (self.b[d] - self.a[d]) * (1 - math.cos(coordinate_normalized * math.pi)) / 2
            # coordinate = (coordinate + 1 + self.a[d]) / (2 * (self.b[d] - self.a[d]))
            #print("Transformed", coordinate, coordinate_normalized)
            return coordinate
        else:
            return 0.5 * (end+start)

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        max_level = max(grid_levels_1D)
        grid_levels_1D = np.asarray(grid_levels_1D)
        level_coordinate_array = self.get_full_level_hierarchy(grid_1D, grid_levels_1D, a, b)
        #print(level_coordinate_array)
        self.start = a
        self.end = b
        # level = 0
        weights = np.zeros(len(grid_1D))
        starting_level = 0 if self.boundary else 1
        for l in range(starting_level, max_level + 1):
            h = (self.end - self.start) / 2 ** l
            # calculate knots for spline construction
            if l < log2(self.p + 1):
                knots = level_coordinate_array[l] #np.linspace(self.start, self.end, 2 ** l + 1)
            else:
                knots = np.array([self.start + i * h if i < 0 or i >= len(level_coordinate_array[l]) else level_coordinate_array[l][i] for i in range(-self.p, 2 ** l + self.p + 1) if
                                  i <= 0 or (self.p + 1) / 2 <= i <= 2 ** l - (self.p + 1) / 2 or i >= 2 ** l])
            #print(knots, "level", l, "dimension", d)
            #iterate over levels and add hierarchical B-Splines
            if l == 0:
                start = 0
                end = 2
                step = 1
            else:
                start = 1
                end = 2 ** l
                step = 2
            for i in range(start, end, step):
                # i is the index of the basis function
                # point associated to the basis (spline) with index i
                x_basis = level_coordinate_array[l][i]
                # if this knot is part of the grid insert it
                #print(i, x_basis, grid_1D, grid_levels_1D, l)
                if x_basis in grid_1D:
                    if self.modified_basis:
                        spline = HierarchicalNotAKnotBSplineModified(self.p, i, l, knots, a, b)
                    else:
                        spline = HierarchicalNotAKnotBSpline(self.p, i, l, knots)
                    index = grid_1D.index(x_basis)
                    self.basis[d][index] = spline
                    weights[index] = spline.get_integral(self.start, self.end, self.coords_gauss, self.weights_gauss)
        if not self.boundary:
            self.basis[d] = self.basis[d][1:-1]
        for spline in self.basis[d]:
            assert spline is not None
        return weights

    def get_full_level_hierarchy(self, grid_1D, grid_levels_1D, a, b):
        max_level = max(grid_levels_1D)
        grid_levels_1D = np.asarray(grid_levels_1D)
        level_coordinate_array = [[] for l in range(max_level + 1)]
        level_coordinate_array_complete = [[] for l in range(max_level + 1)]
        for i, coordinate in enumerate(grid_1D):
            level_coordinate_array[grid_levels_1D[i]].append(coordinate)
            if grid_levels_1D[i] == 0:
                level_coordinate_array_complete[grid_levels_1D[i]].append(coordinate)

        level_index = [[] for l in range(max_level + 1)]

        # fill up missing coordinates
        for l in range(1, max_level + 1):
            for i in range(2**(l-1)):
                level_coordinate_array_complete[l].append(level_coordinate_array_complete[l-1][i])
                if i == len(level_coordinate_array[l]) or level_coordinate_array[l][i] > level_coordinate_array_complete[l-1][i+1]:
                    start = level_coordinate_array_complete[l-1][i]

                    end = level_coordinate_array_complete[l-1][i+1]

                    midpoint = self.get_mid_point(start, end)
                    level_coordinate_array[l].insert(i, midpoint)
                    level_coordinate_array_complete[l].append(midpoint)
                else:
                    level_coordinate_array_complete[l].append(level_coordinate_array[l][i])
            level_coordinate_array_complete[l].append(level_coordinate_array_complete[l-1][-1])
        return level_coordinate_array_complete


class GlobalLagrangeGrid(GlobalBasisGrid):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, modified_basis: bool=False, p: int=3):
        self.boundary = boundary
        self.integrator = IntegratorHierarchicalBasisFunctions(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.modified_basis = modified_basis
        assert p >= 1
        self.p = p
        self.coords_gauss = None
        assert not(modified_basis) or not(boundary)
        self.surplus_values = {}

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int,
                                grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        max_level = max(grid_levels_1D)
        grid_1D = list(grid_1D)
        grid_levels_1D = np.asarray(grid_levels_1D)
        level_coordinate_array = [[] for l in range(max_level+1)]
        for i, p in enumerate(grid_1D):
            level_coordinate_array[grid_levels_1D[i]].append(p)

        #print(level_coordinate_array)
        self.start = a
        self.end = b
        # level = 0
        weights = np.zeros(len(grid_1D))
        starting_level = 0 if self.boundary else 1

        for l in range(starting_level, max_level + 1):
            for i in range(len(level_coordinate_array[l])):
                x_basis = level_coordinate_array[l][i]
                knots = grid_1D

                if self.modified_basis:
                    # spline = LagrangeBasisRestrictedModified(self.p, knots.index(x_basis), knots, a, b, l)
                    RuntimeError("Modified Basis is not supported")

                spline = LagrangeBasis(self.p, knots.index(x_basis), knots)
                index = grid_1D.index(x_basis)
                self.basis[d][index] = spline
                weights[index] = self._get_spline_integral(spline, d)

        if not self.boundary:
            self.basis[d] = self.basis[d][1:-1]

        for basis in self.basis[d]:
            assert basis is not None

        return weights

    def _get_spline_integral(self, spline, d=None):
        if self.coords_gauss is None:
            self.coords_gauss, self.weights_gauss = legendre.leggauss(int(self.p/2) + 1)

        return spline.get_integral(self.start, self.end, self.coords_gauss, self.weights_gauss)


class GlobalHierarchicalLagrangeGrid(GlobalBasisGrid):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, modified_basis: bool=False, p: int=3):
        self.boundary = boundary
        self.integrator = IntegratorHierarchicalBasisFunctions(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.modified_basis = modified_basis
        assert p >= 1
        self.p = p
        self.coords_gauss = None
        assert not(modified_basis) or not(boundary)
        self.surplus_values = {}

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        max_level = max(grid_levels_1D)
        grid_1D = list(grid_1D)
        grid_levels_1D = np.asarray(grid_levels_1D)
        level_coordinate_array = [[] for l in range(max_level+1)]
        for i, p in enumerate(grid_1D):
            level_coordinate_array[grid_levels_1D[i]].append(p)

        #print(level_coordinate_array)
        self.start = a
        self.end = b
        # level = 0
        weights = np.zeros(len(grid_1D))
        starting_level = 0 if self.boundary else 1
        parents = {}
        for l in range(starting_level, max_level + 1):
            #h = (self.end - self.start) / 2 ** l
            # calculate knots for spline construction
            #print(knots, "level", l, "dimension", d)
            #iterate over levels and add hierarchical B-Splines
            if l == 0:
                start = 0
                end = 2
                step = 1
            else:
                start = 1
                end = 2 ** l
                step = 2
            for i in range(len(level_coordinate_array[l])):
                # i is the index of the basis function
                # point associated to the basis (spline) with index i
                x_basis = level_coordinate_array[l][i]
                if l == 0:
                    knots = list(level_coordinate_array[l])
                elif l == 1:
                    if self.boundary:
                        knots = list(sorted(level_coordinate_array[0] + level_coordinate_array[1]))
                    else:
                        knots = list(sorted(level_coordinate_array[1]))
                else:
                    parent = self.get_parent(x_basis, grid_1D, grid_levels_1D)
                    knots = list(sorted(parents[parent] + [x_basis]))
                parents[x_basis] = knots
                if len(knots) > self.p + 1:
                    index_x = knots.index(x_basis)
                    left = index_x
                    right = len(knots) - index_x - 1
                    if left < int((self.p+1)/2):
                        knots = knots[: self.p+1]
                    elif right < int(self.p/2):
                        knots = knots[-(self.p+1):]
                    else:
                        knots = knots[index_x - int((self.p + 1)/2): index_x + int(self.p/2) + 1]
                    assert len(knots) == self.p + 1
                # if this knot is part of the grid insert it
                #print(i, x_basis, grid_1D, grid_levels_1D, l)
                if self.modified_basis:
                    spline = LagrangeBasisRestrictedModified(self.p, knots.index(x_basis), knots, a, b, l)
                else:
                    spline = LagrangeBasisRestricted(self.p, knots.index(x_basis), knots)
                index = grid_1D.index(x_basis)
                self.basis[d][index] = spline
                weights[index] = self._get_spline_integral(spline, d)
        if not self.boundary:
            self.basis[d] = self.basis[d][1:-1]
        for basis in self.basis[d]:
            assert basis is not None
        return weights

    def _get_spline_integral(self, spline, d=None):
        if self.coords_gauss is None:
            self.coords_gauss, self.weights_gauss = legendre.leggauss(int(self.p/2) + 1)
        return spline.get_integral(self.start, self.end, self.coords_gauss, self.weights_gauss)

    def get_parent(self, point_position: float, grid_1D: Sequence[float], grid_levels: Sequence[int]) -> float:
        index_p = grid_1D.index(point_position)
        level_p = grid_levels[index_p]
        found_parent=False
        for i in reversed(range(index_p)):
            if grid_levels[i] == level_p - 1:
                return grid_1D[i]
            elif grid_levels[i] < level_p - 1:
                break
        for i in range(index_p+1, len(grid_1D)):
            if grid_levels[i] == level_p - 1:
                return grid_1D[i]
            elif grid_levels[i] < level_p - 1:
                break
        assert False


class GlobalLagrangeGridWeighted(GlobalHierarchicalLagrangeGrid):
    def __init__(self, a: Sequence[float], b: Sequence[float], uq_operation, boundary: bool=True, modified_basis: bool=False, p: int=3):
        super().__init__(a, b, boundary=boundary, modified_basis=modified_basis, p=p)
        self.distributions = uq_operation.get_distributions()
        self.distributions_cp = uq_operation.get_distributions_chaospy()

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        weights = super().compute_1D_quad_weights(grid_1D, a, b, d, grid_levels_1D=grid_levels_1D)
        # ~ print(weights, grid_1D)
        return weights

    def _get_spline_integral(self, spline, d: int):
        distr = self.distributions[d]
        a, b = spline.get_boundaries()
        return distr.calculate_integral(spline, a, b)

    def get_mid_point(self, a: float, b: float, d: int):
        distr = self.distributions[d]
        return GlobalTrapezoidalGridWeighted.get_middle_weighted(a, b, distr.cdf, distr.ppf)


class GlobalSimpsonGrid(GlobalGrid):
    def __init__(self, a: Sequence[float], b: Sequence[float], boundary: bool=True, modified_basis: bool=False):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.modified_basis = False #modified_basis
        self.high_order_grid = GlobalHighOrderGrid(a, b, boundary=boundary, do_nnls=False, max_degree=2, split_up=False, modified_basis=False)
        assert not(modified_basis) or not(boundary)

    # Weights computed using https://www.wolframalpha.com/input/?i=Solve+x%2By%2Bz+%3D+c+-+a,+a*x+%2B+b+*y+%2B+c+*+z+%3D+c%5E2+%2F+2+-+a%5E2+%2F+2,+++a%5E2*x+%2B+b%5E2+*y+%2B+c%5E2+*+z+%3D+c%5E3+%2F3+-+a%5E3+%2F+3++++for+++x,+y,+z
    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        if len(grid_1D) == 1:
            return np.array([b - a])
        elif len(grid_1D) == 2:
            return super().compute_1D_quad_weights(grid_1D, a, b)
        elif len(grid_1D) % 2 != 1:
            weights_left = self.high_order_grid.compute_1D_quad_weights(grid_1D[:4], a, grid_1D[3])
            weight_right = self.compute_1D_quad_weights(grid_1D[3:], grid_1D[3], b)
            weights = np.zeros(len(grid_1D))
            weights[:4] += weights_left
            weights[3:] += weight_right
            return weights
        weights = np.zeros(len(grid_1D))
        assert len(grid_1D) >= 3

        for i in range(len(grid_1D)):

            if i % 2 == 0:
                if i > 0:
                    x_1 = grid_1D[i - 2]
                    x_2 = grid_1D[i - 1]
                    x_3 = grid_1D[i]
                    weights[i] += (x_1 - x_3) * (x_1 - 3*x_2 + 2*x_3) / (6 * (x_2 - x_3))
                if i < len(grid_1D) - 1:
                    x_1 = grid_1D[i]
                    x_2 = grid_1D[i + 1]
                    x_3 = grid_1D[i + 2]
                    weights[i] += (x_3 - x_1) * (2*x_1 - 3*x_2 + x_3) / (6 *(x_1 - x_2))
            else:
                x_1 = grid_1D[i-1]
                x_2 = grid_1D[i]
                x_3 = grid_1D[i+1]
                weights[i] += (x_3 - x_1)**3 / (6*(x_3 - x_2) * (x_2 - x_1))

        if self.modified_basis and not (b - a) * (1 - 10 ** -12) <= sum(weights) <= (
                b - a) * (1 + 10 ** -12):
            print(grid_1D, weights)
        if self.modified_basis:
            assert (b - a) * (1 - 10 ** -12) <= sum(weights) <= (
                b - a) * (1 + 10 ** -12)
        bad_approximation = self.check_quality_of_quadrature_rule(a, b, min(len(grid_1D), 2 if len(
            grid_1D) % 2 == 1 else 1), grid_1D, weights)
        if bad_approximation:
            print(grid_1D, weights)
        assert not bad_approximation
        return weights


from scipy.optimize import nnls
import matplotlib.pyplot as plt

class GlobalHighOrderGrid(GlobalGrid):
    def __init__(self, a, b, boundary=True, do_nnls=False, max_degree=5, split_up=True, modified_basis=False):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        self.do_nnls = do_nnls
        self.max_degree = max_degree
        self.split_up = split_up
        self.modified_basis = modified_basis
        self.current_dimension = -1
        self.trapezoidal_grid = GlobalTrapezoidalGrid(a,b, boundary, modified_basis)
        assert not(modified_basis) or not(boundary)

    def compute_1D_quad_weights(self, grid_1D: Sequence[float], a: float, b: float, d: int, grid_levels_1D: Sequence[int]=None) -> Sequence[float]:
        '''
        weights = np.zeros(len(grid_1D))
        for i in range(len(grid_1D)):
            if i > 0:
                weights[i] += 0.5*(grid_1D[i] - grid_1D[i-1])
            elif grid_1D[0] > a:
                weights[i] += 0.5 * (grid_1D[i] - a)
            if i < len(grid_1D) - 1:
                weights[i] += 0.5*(grid_1D[i + 1] - grid_1D[i])
            elif grid_1D[len(grid_1D) - 1] < b:
                weights[i] += 0.5 * (b - grid_1D[i])
        return weights
        '''
        self.set_current_dimension(d)
        #if self.max_degree == 2:

        weights, degree = self.get_1D_weights_and_order(grid_1D, a, b, grid_levels_1D)
        #print("Degree of quadrature", degree, "Number of points", len(grid_1D))
        if self.split_up:
            #print(grid, degree)
            if len(grid_1D) > 1 and (len(grid_1D) > 3 or self.boundary):
                weights, degree = self.recursive_splitting3(grid_1D, a, b, degree, grid_levels_1D)
        #print(weights, grid_1D, degree)
        #print("Degree", degree, len(grid_1D))
        #print(weights_1D, sum(abs(weights_1D)), sum(weights_1D), sum(weightsD), d_old)
        #print("Order", d)
        #for i in range(d):
        #    print(abs(sum([grid_1D[j]**i * weights_1D_old[j] for j in range(len(grid_1D))]) - 1 / (i+1)))
        #print("Degree of quadrature", degree, "Number of points", len(grid_1D))
        #if degree == 1:
            #print(grid_1D)
        #print(grid_1D, weights, degree)
        self.set_current_dimension(-1)
        return np.array(weights)

    def recursive_splitting2(self, grid_1D, a, b, d):
        weights_1, d_1 = self.get_1D_weights_and_order(grid_1D[ : int(len(grid_1D)/2) + 1], a, grid_1D[int(len(grid_1D)/2)])
        weights_2, d_2 = self.get_1D_weights_and_order(grid_1D[int(len(grid_1D)/2): ], grid_1D[int(len(grid_1D)/2)], b)
        #print("found order", d, "new orders", d_1, d_2)
        #if d_1 > d/2 and d_2 > d/2:
        if d_1 >= d and d_2 >= d:
            #print("Successfull splitting")
            weights_1, d_1 = self.recursive_splitting2(grid_1D[ : int(len(grid_1D)/2) + 1], a, grid_1D[int(len(grid_1D)/2)],  d)
            weights_2, d_2 = self.recursive_splitting2(grid_1D[int(len(grid_1D)/2): ], grid_1D[int(len(grid_1D)/2)], b, d)
            weights_1[-1] += weights_2[0]
            combined_weights = np.append(weights_1, weights_2[1:])
            assert len(combined_weights) == len(grid_1D)
            return combined_weights, min(d_1,d_2)

        else:
            return self.get_1D_weights_and_order(grid_1D, a, b)

    def recursive_splitting3(self, grid_1D, a, b, d, grid_levels, factor=1):
        weights_1, d_1 = self.get_1D_weights_and_order(grid_1D[ : int(len(grid_1D)/2) + 1], a, grid_1D[int(len(grid_1D)/2)], grid_levels)
        weights_2, d_2 = self.get_1D_weights_and_order(grid_1D[int(len(grid_1D)/2): ], grid_1D[int(len(grid_1D)/2)], b, grid_levels)
        d_self = min(d_1, d_2)
        max_degree = max(d_self,d)
        #print(d_self, max_degree)
        if len(grid_1D) > max(d_1,d) + max(d_2, d)+1:
            weights_1_rec, d_1 = self.recursive_splitting3(grid_1D[ : int(len(grid_1D)/2) + 1], a, grid_1D[int(len(grid_1D)/2)],  max(d_1, d)*factor, grid_levels)
            weights_2_rec, d_2 = self.recursive_splitting3(grid_1D[int(len(grid_1D)/2): ], grid_1D[int(len(grid_1D)/2)], b, max(d_2, d)*factor, grid_levels)
            if d_1 >= max(d_1, d) * factor and d_2 >= max(d_2, d) * factor:
                weights_1_rec[-1] += weights_2_rec[0]
                combined_weights = np.append(weights_1_rec, weights_2_rec[1:])
                bad_approximation = self.check_quality_of_quadrature_rule(a, b, d, grid_1D, combined_weights)
                if not bad_approximation:
                    assert len(combined_weights) == len(grid_1D)
                    return combined_weights, min(d_1, d_2)
        if d_self >= d * factor:
            weights_1[-1] += weights_2[0]
            combined_weights = np.append(weights_1, weights_2[1:])
            bad_approximation = self.check_quality_of_quadrature_rule(a, b, d, grid_1D, combined_weights)
            if not bad_approximation:
                assert len(combined_weights) == len(grid_1D)
                return combined_weights, min(d_1, d_2)
        return self.get_1D_weights_and_order(grid_1D, a, b, grid_levels)

    def recursive_splitting(self, grid_1D, a, b, d):
        middle = (b+a)/2
        if middle in grid_1D:
            middle_index = grid_1D.index(middle)
        else:
            return self.get_1D_weights_and_order(grid_1D, a, b)
        weights_1, d_1 = self.get_1D_weights_and_order(grid_1D[ : middle_index + 1], a, middle)
        weights_2, d_2 = self.get_1D_weights_and_order(grid_1D[middle_index: ], middle, b)
        #print("found order", d, "new orders", d_1, d_2)
        #if d_1 > d/2 and d_2 > d/2:
        if d_1 > d/2 and d_2 > d/2:
            #print("Successfull splitting")
            weights_1, d_1 = self.recursive_splitting(grid_1D[ : middle_index + 1], a, middle,  d)
            weights_2, d_2 = self.recursive_splitting(grid_1D[middle_index: ], middle, b, d)
            weights_1[-1] += weights_2[0]
            combined_weights = np.append(weights_1, weights_2[1:])
            assert len(combined_weights) == len(grid_1D)
            return combined_weights, min(d_1,d_2)

        else:
            return self.get_1D_weights_and_order(grid_1D, a, b)

    def get_composite_quad_weights(self, grid_1D, a, b):
        return GlobalTrapezoidalGrid.compute_weights(grid_1D, a, b, self.modified_basis)

    def get_1D_weights_and_order(self, grid_1D, a, b, grid_levels, improve_weight=True, reduce_max_order_for_length=False):
        #if len(grid_1D) == 3 and grid_1D[1] - grid_1D[0] == grid_1D[2] - grid_1D[1]:
        #    return np.array([1/6, 4/6, 1/6]) * (b-a), 2
        #if len(grid_1D) == 2:
        #    return np.array([0.5, 0.5]) * (b-a), 1
        d = d_old = 1
        weights_1D_old = np.zeros(len(grid_1D))
        grid_1D_normalized = 2 * (np.array(grid_1D) - a) / (b - a) - 1
        if self.boundary:
            trapezoidal_weights = self.get_composite_quad_weights(grid_1D_normalized, a, b)
        else:
            trapezoidal_weights = np.array(self.get_composite_quad_weights(grid_1D, a, b)[1:-1])
            trapezoidal_weights = trapezoidal_weights * 2 / sum(trapezoidal_weights)
            grid_1D_normalized = grid_1D_normalized[1:-1]
            #print(grid_1D, trapezoidal_weights, grid_1D_normalized)
        if len(grid_1D_normalized) == 1:
            if self.boundary:
                return np.array([1]) * (b-a), 0
            else:
                return np.array([0,1,0]) * (b-a), 0

        if len(grid_1D_normalized) == 0:
            return np.array([0]) * (b - a), -100
        #print(trapezoidal_weights, grid_1D_normalized)
        weights_1D_old = np.array(trapezoidal_weights)

        while(d < len(grid_1D) - int(reduce_max_order_for_length) and d <= self.max_degree):

            #print("Trapezoidal weights", trapezoidal_weights)
            #print("Grid points", grid_1D_normalized)
            evaluations, alphas, betas, lambdas = self.get_polynomials_and_evaluations(grid_1D_normalized, d, trapezoidal_weights)
            coordsD, weightsD = legendre.leggauss(int((d+2)/2))
            #coordsD = np.array(coordsD)
            #coordsD += 1
            #coordsD *= (b-a) / 2.0
            #coordsD += a
            #print("Gauss Points", coordsD, alphas, betas, lambdas)
            #weightsD = np.array(weightsD) * (b-a) / 2
            evaluations_moments = self.evaluate_polynomials(coordsD, alphas, betas, lambdas)
            moments = np.array([sum(evaluations_moments[i] * weightsD) for i in range(d+1)])
            #print("Moments for interval", a, b, "are", moments)
            #print("Evaluation", evaluations)
            #print("Evaluation gauss", evaluations_moments)

            #print(evaluations, d, moments, trapezoidal_weights)
            if not self.do_nnls:
                weights_1D = trapezoidal_weights * np.inner(evaluations.T, moments)
            else:
                weights_1D, error = nnls(evaluations, moments)
                if error > 0:
                    break
            weights_1D = (b-a) * weights_1D / 2
            #AR = np.array(evaluations)
            #for i in range(len(AR)):
            #    print(np.sqrt(trapezoidal_weights))
            #    plt.plot(grid_1D_normalized, AR[i])
            #    plt.show()
            #    AR[i] *= trapezoidal_weights
            #print("AR", AR, moments)

            #print(nnls(evaluations, moments))
            #print(grid_1D, weights_1D)
            bad_approximation = self.check_quality_of_quadrature_rule(a, b, d, grid_1D, weights_1D)
            d += 1
            if bad_approximation:
                break
            d_old = d - 1
            weights_1D_old = weights_1D
            #if improve_weight and all([w > 0 for w in weights_1D]):
            #    trapezoidal_weights = weights_1D
        if not self.boundary:
            weights_1D_old = np.array([0] + list(weights_1D_old) + [0])
        return weights_1D_old, d_old

    @staticmethod
    def check_quality_of_quadrature_rule(a, b, degree, grid_1D, weights_1D):
        tol = 10 ** -14
        bad_approximation = (sum(abs(weights_1D)) - (b - a)) / (b - a) > tol  # (tol/(10**-15))**(1/self.dim)
        # if bad_approximation:
        # print("Too much negative entries, error:",
        #     (sum(abs(weights_1D)) - (b - a)) / (b - a))
        '''
        if not bad_approximation:
            for i in range(d + 1):
                real_moment = b ** (i + 1) / (i + 1) - a ** (i + 1) / (i + 1)
                if abs(sum([grid_1D[j] ** i * weights_1D[j] for j in range(len(grid_1D))]) - real_moment) / abs(
                        real_moment) > tol:
                    # print("Bad approximation for degree",i, "with error", abs(sum([grid_1D[j]**i * weights_1D[j] for j in range(len(grid_1D))]) - real_moment) / abs(real_moment) )
                    bad_approximation = True
        '''
        tolerance_lower = 0
        #print(weights_1D, all([w > tolerance_lower for w in weights_1D]), d)
        return not(all([w >= tolerance_lower for w in weights_1D]))
        #return sum(weights_1D) < (b-a) * 2
        return bad_approximation

    @staticmethod
    def get_polynomials_and_evaluations(x_array, d, trapezoidal_weights):
        evaluations = np.ones((d+1,len(x_array)))
        x_array = np.array(x_array)
        alphas = np.zeros(d+1)
        betas = np.zeros(d+1)
        lambdas = np.ones(d+1)
        lambdas[0] = 1.0 / math.sqrt(sum(trapezoidal_weights))
        evaluations[0] *= lambdas[0]
        for i in range(1, d+1):
            if i == 1:
                evaluation_2 = np.zeros(len(x_array))
                evaluation_1 = evaluations[0]
            else:
                evaluation_2 = evaluations[i-2]
                evaluation_1 = evaluations[i-1]
            alpha = np.inner(x_array * evaluation_1 * trapezoidal_weights, evaluation_1) / np.inner(evaluation_1 * trapezoidal_weights, evaluation_1)
            #print(np.inner(evaluation_1 * trapezoidal_weights, evaluation_1))
            alphas[i] = alpha
            #print(x_array, evaluation_2)
            if i == 1:
                beta = 2
            else:
                beta = np.inner(x_array * evaluation_1 * trapezoidal_weights, evaluation_2) / np.inner(evaluation_2 * trapezoidal_weights, evaluation_2)
                #beta = np.inner(evaluation_1 * trapezoidal_weights, evaluation_1) / np.inner(evaluation_2 * trapezoidal_weights, evaluation_2)

            betas[i] = beta
            evaluations[i] = (x_array - alpha) * evaluation_1 - beta * evaluation_2
            #print(i, np.inner(evaluations[i] * trapezoidal_weights, evaluations[i]), evaluations[i], trapezoidal_weights, x_array, alpha, beta, evaluation_1, evaluation_2)
            if sum(abs(evaluations[i] )) != 0:
                lambdas[i] = 1.0 / math.sqrt(np.inner(evaluations[i]*trapezoidal_weights, evaluations[i]))
                evaluations[i] *= lambdas[i]
        return evaluations, alphas, betas, lambdas

    @staticmethod
    def evaluate_polynomials(points, alphas, betas, lambdas):
        evaluations = np.zeros((len(alphas), len(points)))
        evaluations[0] = np.ones(len(points)) * lambdas[0]
        for i in range(1, len(alphas)):
            if i == 1:
                evaluation_2 = np.zeros(len(points))
                evaluation_1 = evaluations[0]
            else:
                evaluation_2 = evaluations[i-2]
                evaluation_1 = evaluations[i-1]
            evaluations[i] = lambdas[i] * ((points - alphas[i]) * evaluation_1 - betas[i] * evaluation_2)
        return evaluations

    def set_current_dimension(self, d): self.current_dimension = d

    def get_current_dimension(self):
        assert self.current_dimension >= 0, "Dimension not saved"
        return self.current_dimension

    @staticmethod
    def normalize_coords(coordinates, a, b):
        assert a < b
        return 2.0 * (np.array(coordinates) - a) / (b - a) - 1.0

    @staticmethod
    def denormalize_coords(coordinates, a, b):
        return a + (np.array(coordinates) + 1.0) * 0.5 * (b - a)


class GlobalHighOrderGridWeighted(GlobalHighOrderGrid):
    def __init__(self, a, b, uq_operation, boundary=True, do_nnls=False, max_degree=5, split_up=True, modified_basis=False):
        # split_up is not yet supported
        split_up = False
        super().__init__(a, b, boundary=boundary, do_nnls=do_nnls, max_degree=max_degree, split_up=split_up, modified_basis=modified_basis)
        self.distributions = uq_operation.get_distributions()
        self.distributions_cp = uq_operation.get_distributions_chaospy()

    def get_composite_quad_weights(self, grid_1D, a, b):
        d = self.get_current_dimension()
        distr = self.distributions[d]
        return GlobalTrapezoidalGridWeighted.compute_weights(grid_1D, a, b, distr, self.boundary, self.modified_basis)

    def get_mid_point(self, a, b, d):
        distr = self.distributions[d]
        return GlobalTrapezoidalGridWeighted.get_middle_weighted(a, b, distr.cdf, distr.ppf)

    def get_1D_weights_and_order(self, grid_1D, a, b, improve_weight=True, reduce_max_order_for_length=False, trapezoidal_weights=None):
        # Corner cases, they are wrong if split_up is used.
        if len(grid_1D) == 1:
            return np.array([1]), -100
        if len(grid_1D) == 0:
            return np.array([0]), -100

        search_for_zeros = trapezoidal_weights is None
        if trapezoidal_weights is None:
            trapezoidal_weights = self.get_composite_quad_weights(grid_1D, a, b)

        if search_for_zeros:
            # Ignore zero weights at boundaries
            tol = 10 ** -14
            zeros_left = 0
            zeros_right = 0
            for i in range(len(grid_1D)):
                if trapezoidal_weights[i] > tol:
                    break
                zeros_left += 1
            for i in reversed(range(len(grid_1D))):
                if trapezoidal_weights[i] > tol:
                    break
                zeros_right += 1
            if zeros_left > 0 or zeros_right > 0:
                a_nonzero = grid_1D[zeros_left]
                b_nonzero = grid_1D[-zeros_right-1]
                grid_1D_nonzero = grid_1D[zeros_left:-zeros_right]
                trapezoidal_weights_nonzero = trapezoidal_weights[zeros_left:-zeros_right]
                weights_nonzero, order = self.get_1D_weights_and_order(
                    grid_1D_nonzero, a_nonzero,
                    b_nonzero, improve_weight=improve_weight,
                    reduce_max_order_for_length=reduce_max_order_for_length,
                    trapezoidal_weights=trapezoidal_weights_nonzero)
                weights = np.concatenate([np.zeros(zeros_left), weights_nonzero, np.zeros(zeros_right)])
                return weights, order

        if any([w == 0 for w in trapezoidal_weights]):
            print("inner trapezoidal weight is zero: ", trapezoidal_weights)
            return trapezoidal_weights, 1
        # ~ if len(grid_1D) <= 3:
            # ~ print("grid is small, trapezoidal weights: ", trapezoidal_weights)
            # ~ return trapezoidal_weights, 1

        assert all([w > 0 for w in trapezoidal_weights])

        # Current and previous degree
        d = d_old = 1
        assert a <= grid_1D[0] <= grid_1D[-1] <= b
        grid_1D_normalized = self.normalize_coords(grid_1D, a, b)
        # Previous weights
        weights_1D_old = np.array(trapezoidal_weights)
        # The weights which are used to increase the order
        weights_lower_order = trapezoidal_weights

        while(d < len(grid_1D) - int(reduce_max_order_for_length) and d <= self.max_degree):

            #print("Trapezoidal weights", trapezoidal_weights)
            #print("Grid points", grid_1D_normalized)
            evaluations, alphas, betas, lambdas = self.get_polynomials_and_evaluations(grid_1D_normalized,d, weights_lower_order)
            # Use at least 5 Gauss points because they can be cached and generating them is slow
            coordsD_normalized, weightsD = self.do_gauss_quad_normalized(max(int((d+2)/2), 5), a, b)
            evaluations_moments = self.evaluate_polynomials(coordsD_normalized, alphas, betas, lambdas)
            moments = np.array([sum(evaluations_moments[i] * weightsD) for i in range(d+1)])
            #print("Moments for interval", a, b, "are", moments)
            #print("Evaluation", evaluations)
            #print("Evaluation gauss", evaluations_moments)

            #print(evaluations, d, moments, trapezoidal_weights)
            if not self.do_nnls:
                weights_1D = weights_lower_order * np.inner(evaluations.T, moments)
            else:
                weights_1D, error = nnls(evaluations, moments)
                if error > 0:
                    break
            #print(grid_1D, weights_1D)
            bad_approximation = self.check_quality_of_quadrature_rule(weights_1D)
            d += 1
            if bad_approximation:
                break
            d_old = d - 1
            weights_1D_old = weights_1D
            if improve_weight and all([w > 0 for w in weights_1D]):
                weights_lower_order = weights_1D
        # ~ print("trapezoidal weights:", weights_1D_old)
        # ~ print("weights:", weights_1D_old)
        # ~ print("weights_lower_order:", weights_lower_order)
        assert all([w >= 0 for w in weights_1D_old])
        return weights_1D_old, d_old

    @staticmethod
    def check_quality_of_quadrature_rule(weights_1D):
        # ~ is_bad = not(all([w >= 0.0 for w in weights_1D])) and sum([abs(w) for w in weights_1D]) > 1.0
        is_bad = not(all([w >= 0.0 for w in weights_1D]))
        return is_bad

    def do_gauss_quad_normalized(self, num_quad_points: int, a: float, b: float):
        d = self.get_current_dimension()
        cp_distr = self.distributions_cp[d]
        distr = self.distributions[d]
        coords, weights = distr.get_quad_points_weights(num_quad_points, cp_distr)
        coords = self.normalize_coords(coords, a, b)
        return coords, weights

'''
    # unused
    def get_weight_function_normalized(self, a, b):
        d = self.get_current_dimension()
        distr = self.distributions[d]
        return lambda x: distr.pdf(a + (x + 1.0) * 0.5 * (b - a))

    # unused
    @staticmethod
    def calculate_moment(k, x1, x2, weight_func, weight_func_integrated=None):
        if k == 0:
            return weight_func_integrated(x2) - weight_func_integrated(x1)
        return integrate.quad(lambda x: x ** k * weight_func(x), x1, x2,
            epsrel=10 ** -2, epsabs=10 ** -3)[0]
#'''


# this class is outdated
class EquidistantGridGlobal(Grid):
    def __init__(self, a, b, boundary=True):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        if (self.boundary == False):
            self.lowerBorder = np.ones(self.dim, dtype=int)
        else:
            self.lowerBorder = np.zeros(self.dim, dtype=int)

    def is_global(self):
        return True

    def isNested(self):
        return True

    def set_grid(self, grid_points, levelvec):
        self.coords = []
        self.weights = []
        self.numPoints = np.zeros(self.dim, dtype=int)
        for d in range(self.dim):
            refinementDim = grid_points[d]
            coordsD = self.mapPoints(
                refinementDim.getPointsLine()[self.lowerBorder[d]: -1 * int(self.boundary == False)], self.levelvec[d],
                d)
            weightsD = self.compute_1D_quad_weights(coordsD)
            self.coords.append(coordsD)
            self.weights.append(weightsD)
            self.numPoints[d] = len(coordsD)
        self.numPointsWithoutCoarsening = self.level_to_num_points(levelvec)

    def compute_1D_quad_weights(self, grid_1D):
        N = len(grid_1D)
        V = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                V[i, j] = eval_sh_legendre(j, grid_1D[i]) * np.sqrt(2 * j + 1)

        weights = np.linalg.inv(V)[0, :]

        return weights

    def levelToNumPoints(self, levelvec):
        if hasattr(self, 'numPoints'):
            return self.numPoints
        else:
            return [2 ** levelvec[d] + 1 - int(self.boundary == False) * 2 for d in range(self.dim)]

    def getPoints(self):
        return list(zip(*[g.ravel() for g in np.meshgrid(*self.coordinate_array)]))

    def getCoordinate(self, indexvector):
        position = np.zeros(self.dim)
        for d in range(self.dim):
            position[d] = self.coordinate_array[d][indexvector[d]]
        return position

    def mapPoints(self, equidistantAdaptivePoints, level, d):
        return equidistantAdaptivePoints


class GlobalHierarchizationGrid(EquidistantGridGlobal):
    def __init__(self, a, b, boundary=True):
        EquidistantGridGlobal.__init__(self, a, b, boundary=True)
        self.doHierarchize = True
        self.integrator = IntegratorHierarchical()


# this class is outdated
class ClenshawCurtisGridGlobal(EquidistantGridGlobal):
    def __init__(self, a, b, boundary=True):
        self.boundary = boundary
        self.integrator = IntegratorArbitraryGrid(self)
        self.a = a
        self.b = b
        self.dim = len(a)
        self.length = np.array(b) - np.array(a)
        if (self.boundary == False):
            self.lowerBorder = np.ones(self.dim, dtype=int)
        else:
            self.lowerBorder = np.zeros(self.dim, dtype=int)

    '''
    def setCurrentArea(self,start,end,levelvec):
        self.start = start
        self.end = end
        self.levelvec = levelvec
        self.dim = len(self.levelvec)
        self.numPoints = self.levelToNumPoints(levelvec)
        self.numPointsWithBoundary = list(self.numPoints)
        self.length = np.array(end) - np.array(start)
        self.lowerBorder = np.zeros(self.dim,dtype=int)
        self.upperBorder = np.array(self.numPoints, dtype=int)
        if(self.boundary == False):
            for i in range(self.dim):
                if start[i]==0 :
                    self.numPointsWithBoundary[i] += 1
                    self.lowerBorder[i] = 1
                if end[i] == 1:
                    self.numPointsWithBoundary[i] += 1
                    self.upperBorder[i] = self.numPoints[i]-1
                else:
                    self.upperBorder[i] = self.numPoints[i]
    '''

    def mapPoints(self, equidistantAdaptivePoints, level, d):
        coords = np.zeros(len(equidistantAdaptivePoints))
        for i, p in enumerate(equidistantAdaptivePoints):
            index = self.getIndexTo1DCoordinate(p, level)
            coords[i] = self.get1DCoordinate(index, d)
        return coords

    def get1DCoordinate(self, index, d):
        return self.a[d] + (
                1 - math.cos(math.pi * (index + self.lowerBorder[d]) / (self.numPointsWithoutCoarsening[d] - 1))) * \
               self.length[d] / 2


class GaussGrid(Grid):
    def __init__(self, a, b, integrator=None):
        self.dim = len(a)
        assert len(a) == len(b)
        self.a = a
        self.b = b
        self.boundary = False  # never points on boundary
        if integrator is None:
            self.integrator = IntegratorArbitraryGridScalarProduct(self)
        else:
            if integrator == 'old':
                self.integrator = IntegratorArbitraryGrid(self)
            else:
                assert False
        self._initialize_grids1D()

    def is_high_order_grid(self):
        return True

    @abc.abstractmethod
    def _initialize_grids1D(self): pass


class GaussGrid1D(Grid1d):
    def level_to_num_points_1d(self, level):
        return 2 ** level + 1

    def is_nested(self):
        return False

    @abc.abstractmethod
    def get_1d_points_and_weights(self) -> Tuple[Sequence[float], Sequence[float]]:
            pass

    def is_high_order_grid(self):
        return True


# this class generates a grid according to the Gauss-Legendre quadrature
class GaussLegendreGrid(GaussGrid):
    def __init__(self,a, b, normalize=False):
        self.normalize=normalize
        super().__init__(a, b)

    def _initialize_grids1D(self):
        self.grids = [GaussLegendreGrid1D(a=self.a[d], b=self.b[d], boundary=self.boundary, normalize=self.normalize) for d in range(self.dim)]


class GaussLegendreGrid1D(GaussGrid1D):
    def __init__(self,a, b, boundary=False, normalize=False):
        self.normalize=normalize
        super().__init__(a, b, boundary)

    def get_1d_points_and_weights(self) -> Tuple[Sequence[float], Sequence[float]]:
        coordsD, weightsD = legendre.leggauss(int(self.num_points))
        coordsD = np.array(coordsD)
        coordsD += np.ones(int(self.num_points))
        coordsD *= self.length / 2.0
        coordsD += self.start
        weightsD = np.array(weightsD) * self.length / 2
        if self.normalize:
            weightsD *= 1/(self.end - self.start)
        return coordsD, weightsD


# Tailored for a normal distribution
class GaussHermiteGrid(GaussGrid):
    def __init__(self, expectations, standard_deviations, integrator=None):
        assert len(expectations) == len(standard_deviations)
        dim = len(expectations)
        self.expectations = expectations
        self.standard_deviations = standard_deviations
        a = np.array([-np.inf] * dim)
        b = np.array([np.inf] * dim)
        super().__init__(a, b, integrator=integrator)

    def _initialize_grids1D(self):
        self.grids = [GaussHermiteGrid1D(loc=self.expectations[d], scale=self.standard_deviations[d], boundary=self.boundary) for d in range(self.dim)]


class GaussHermiteGrid1D(GaussGrid1D):
    def __init__(self, loc: float=None, scale: float=None, boundary: bool=True):
        super().__init__(a=-np.inf, b=np.inf, boundary=boundary)
        self.loc = loc
        self.scale = scale

    def get_1d_points_and_weights(self) -> Tuple[Sequence[float], Sequence[float]]:
        coordsD, weightsD = hermite.hermgauss(int(self.num_points))
        coordsD = np.array(coordsD)
        coordsD = coordsD * self.scale * math.sqrt(2) + self.loc
        weightsD = np.array(weightsD)
        weightsD = weightsD / math.sqrt(math.pi)
        return coordsD, weightsD


from scipy.stats import norm
from scipy.linalg import cholesky
from scipy.linalg import ldl

from scipy.sparse import diags
from scipy.sparse.linalg import lobpcg, LinearOperator
from scipy import integrate
from scipy.stats import truncnorm


# this class generates a grid according to the quadrature rule of a truncated normal distribution
# We basically compute: N * \int_a^b f(x) e^(-(x-mean)^2/(2 stddev)) dx. Where N is a normalization factor.
# The code is based on the work in "The Truncated Normal Distribution" by John Burkhardt
class TruncatedNormalDistributionGrid(Grid):
    def __init__(self, a, b, mean, std_dev, integrator=None):
        # we assume here mean = 0 and std_dev = 1 for every dimension
        self.boundary = False  # never points on boundary
        self.dim = len(a)
        assert len(a) == len(b)
        self.a = a
        self.b = b
        if integrator is None:
            self.integrator = IntegratorArbitraryGridScalarProduct(self)
        else:
            if integrator == 'old':
                self.integrator = IntegratorArbitraryGrid(self)
            else:
                assert False
        self.grids = [
            TruncatedNormalDistributionGrid1D(a=a[d], b=b[d], mean=mean[d], std_dev=std_dev[d], boundary=self.boundary)
            for d in range(self.dim)]
        # print(self.normalization, global_a, global_b)

    def is_high_order_grid(self):
        return True

class TruncatedNormalDistributionGrid1D(Grid1d):
    def __init__(self, a, b, mean, std_dev, boundary=False):
        self.shift = lambda x: x * std_dev + mean
        self.shift_back = lambda x: (x - mean) / std_dev
        self.a = self.shift_back(a)
        self.b = self.shift_back(b)
        self.normalization = 1.0 / (norm.cdf(self.b) - norm.cdf(self.a))
        self.boundary = boundary

    def get_mid_point(self, a, b):
        middle_cdf = (norm.cdf(b) + norm.cdf(a)) / 2.0
        return norm.ppf(middle_cdf)

    def level_to_num_points_1d(self, level):
        return 2 ** level

    # This grid is not nested!
    def is_nested(self):
        return False

    # this method implements the moment method for calculation the quadrature points and weights
    # a description can be found in "The Truncated Normal Distribution" from John Burkardt and
    # in "Gene Golub, John Welsch: Calculation of Gaussian Quadrature Rules"
    def get_1d_points_and_weights(self):
        num_points = int(self.num_points)
        self.L_i_dict = {}
        a = self.shift_back(self.start)
        b = self.shift_back(self.end)
        M = self.calculate_moment_matrix(num_points, a, b)

        # the stability can be improved by adding small delta to diagonal -> shifting eigenvalues away from 0
        # currently not needed
        # for i in range(num_points + 1):
        # M[i, i] += 10**-15

        # compute ldl decomposition (a bit more stable than cholesky)
        LU, diag = self.ldl_decomp(M)

        # calculate cholesky factorization based on ldl decomposition -> apply sqrt(diagonal element) to both matrices
        for i in range(num_points + 1):
            for j in range(num_points + 1):
                if (diag[j] < 0):  # in case we encounter a negative diagonal value due to instability / conditioning
                    # fill up with small value > 0
                    LU[i, j] *= 10 ** -15
                else:
                    LU[i, j] *= math.sqrt(diag[j])

        R = LU.T  # we need the upper not the lower triangular part

        # other version using scipy cholesky factorization; tends to crash earlier
        # R = scipy.linalg.cholesky(M)

        alpha_vec = np.empty(num_points)
        alpha_vec[0] = R[0, 1] / R[0, 0]
        for i in range(1, num_points):
            alpha_vec[i] = float(R[i, i + 1]) / R[i, i] - float(R[i - 1, i]) / R[i - 1, i - 1]
        beta_vec = np.empty(num_points - 1)
        for i in range(num_points - 1):
            beta_vec[i] = R[i + 1, i + 1] / R[i, i]

        # fill tridiagonal matrix

        J = np.zeros((num_points, num_points))
        for i in range(num_points):
            J[i, i] = alpha_vec[i]

        for i in range(num_points - 1):
            J[i, i + 1] = beta_vec[i]
            J[i + 1, i] = beta_vec[i]
        evals, evecs = scipy.linalg.eig(J)
        '''
        J = scipy.sparse.diags([alpha_vec, beta_vec, beta_vec], [0,-1,1])

        # get eigenvectors and eigenvalues of J
        X = np.random.rand(num_points, num_points)
        evals, evecs = scipy.sparse.linalg.lobpcg(J, X)
        '''
        points = [self.shift(ev.real) for ev in evals]
        # print(a, b, self.normalization[d])
        mu0 = self.get_moment_normalized(0, a, b)
        weights = [mu0 * value.real ** 2 for value in evecs[0]]
        # print("points and weights", num_points, a, b, points, weights)
        return points, weights

    def ldl_decomp(self, M):
        diag = np.zeros(len(M))
        L = np.zeros((len(M), len(M)))
        for i in range(len(diag)):
            L[i, i] = 1
            for j in range(i):
                summation = 0
                for k in range(j):
                    summation -= L[i, k] * L[j, k] * diag[k]
                L[i, j] = 1.0 / diag[j] * (M[i, j] + summation)
            summation = 0

            for k in range(i):
                summation -= L[i, k] ** 2 * diag[k]
            diag[i] = M[i, i] + summation
        return L, diag

    def calculate_moment_matrix(self, num_points, a, b):
        M = np.empty((num_points + 1, num_points + 1))
        for i in range(num_points + 1):
            for j in range(num_points + 1):
                M[i, j] = self.get_moment_normalized(i + j, a, b)
        return M

    # Calculation of the moments of the truncated normal distribution according to "The Truncated Normal Distribution" from John Burkardt
    # It is slightly simplified as we assume mean=0 and std_dev=1 here.
    def get_moment(self, index, a, b):
        return self.get_L_i(index, a, b)

    def get_L_i(self, index, a, b):
        if index == 0:
            return 1.0 * (norm.cdf(b) - norm.cdf(a))
        if index == 1:
            return - float((norm.pdf(b) - norm.pdf(a)))
        L_i = self.L_i_dict.get(index, None)
        if L_i is None:
            moment_m2 = self.get_L_i(index - 2, a, b)  # recursive search
            L_i = -(b ** (index - 1) * norm.pdf(b) - a ** (index - 1) * norm.pdf(a)) + (index - 1) * moment_m2
            self.L_i_dict[index] = L_i
        # print(index,L_i)
        return L_i

    # Different version of calculating moments according to: "A Recursive Formula for the Moments of a Truncated
    # Univariate Normal Distribution" by Eric Orjebin
    def get_moment2(self, index, a, b):
        if index == -1:
            return 0.0
        if index == 0:
            return 1.0 * (norm.cdf(b) - norm.cdf(a))
        moment = self.moment_dict.get(index, None)
        if moment is None:
            m_m2 = self.get_moment2(index - 2, a, b)
            moment = (index - 1) * m_m2
            moment -= b ** (index - 1) * norm.pdf(b) - a ** (index - 1) * norm.pdf(a)
            self.moment_dict[index] = moment
        return moment

    # Slight modification of 2nd version by restructuring computation; tends to be less stable
    def get_moment5(self, index, a, b):
        moment = self.moment_dict.get(index, None)
        if moment is None:
            moment = self.get_moment5_fac(index, b) * norm.pdf(b) - self.get_moment5_fac(index, a) * norm.pdf(a)
            if index % 2 == 0:
                moment += (norm.cdf(b) - norm.cdf(a))
            print(moment, index)
            self.moment_dict[index] = moment
        return moment

    def get_moment5_fac(self, index, boundary):
        if index == -1:
            return 0
        if index == 0:
            return 0
        return - (boundary ** (index - 1)) + (index - 1) * self.get_moment5_fac(index - 2, boundary)

    # Calculating moments by numerical quadrature; tends to be inaccurate
    def get_moment3(self, index, a, b):
        moment = self.moment_dict.get(index, None)
        if moment is None:
            def integrant(x):
                return x ** index * norm.pdf(x)

            moment = integrate.quad(func=integrant, a=a, b=b, epsrel=10 ** -15, epsabs=10 ** -15)[0]
            '''
            if alpha < -1 and beta > 1:
                moment = integrate.quad(func=integrant, a=alpha, b=-1, epsrel=10**-15)[0]
                moment += integrate.quad(func=integrant, a=1, b=beta, epsrel=10**-15)[0]
                moment += integrate.quad(func=integrant, a=-1, b=-0, epsrel=10**-15)[0]
                moment += integrate.quad(func=integrant, a=0, b=1, epsrel=10**-15)[0]
            '''
            # self.moment_dict[index] = moment
        return moment

    # Calculating moments using scipy
    def get_moment4(self, index, a, b):
        moment = self.moment_dict.get(index, None)
        if moment is None:
            moment = truncnorm.moment(index, a=a, b=b)
            normalization_local = 1.0 / (norm.cdf(b) - norm.cdf(a))
            moment /= normalization_local  # denormalize
            self.moment_dict[index] = moment
        return moment

    def get_moment_normalized(self, index, a, b):
        # print("Moment:", index, " variants:", self.get_moment(index,alpha,beta,mean,std_dev), self.get_moment2(index,alpha,beta,mean,std_dev), self.get_moment4(index,alpha,beta,mean,std_dev) )
        return self.get_moment(index, a, b) * self.normalization

    def is_high_order_grid(self):
        return True


