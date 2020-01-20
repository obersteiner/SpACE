from numpy import linalg as LA
from math import isclose, isinf
from Grid import *
from BasisFunctions import *

class GridOperation(object):
    def is_area_operation(self):
        return False
    @abc.abstractmethod
    def perform_operation(self, levelvector, refinementObjects=None):
        pass

    @abc.abstractmethod
    def set_errors(self, refinementObjects=None):
        pass

    # This method indicates whether we should only count unique points for the cost estimate (return True) or
    # if we should count points multiple times if they are contained in different component grids
    @abc.abstractmethod
    def count_unique_points(self):
        return False

    def area_preprocessing(self, area):
        pass

    def area_postprocessing(self, area):
        pass

    def post_processing(self):
        pass

    # This method can be used if there is a better global error estimate than the summation of the local surplusses
    def get_global_error_estimate(self, refinement):
        return None

    def get_grid(self) -> Grid:
        return self.grid

    def get_reference_solution(self) -> Sequence[float]:
        return self.reference_solution

    def initialize(self):
        pass

    # This method calculates the error between the combi result and the reference solution. Can be changed by Operation.
    def get_error(self, combivalue, reference_solution):
        return max(abs(combivalue - reference_solution))

    # This method is called after the combination and should return the combination result
    def get_result(self):
        pass

    # Returns the size of the model evaluations of each point; in case of scalar values
    # (e.g. of a scalar-valued function) it is 1, otherwise the vector length of the vector output
    def point_output_length(self):
        return 1

    # interpolates mesh_points_grid at the given  evaluation_points using bilinear interpolation
    def interpolate_points(self, mesh_points_grid, evaluation_points):
        return Interpolation.interpolate_points(self.f, self.dim, self.grid, mesh_points_grid, evaluation_points)

    @abc.abstractmethod
    def eval_analytic(self, coordinate: Tuple[float, ...]) -> Sequence[float]:
        pass

    @abc.abstractmethod
    def get_distinct_points(self):
        pass
        f.get_f_dict_size()

class AreaOperation(GridOperation):
    def is_area_operation(self):
        return True

    def perform_operation(self, levelvector, refinementObjects=None):
        for area in enumerate(refinementObjects):
            self.evaluate_area(area, levelvector)

    @abc.abstractmethod
    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container):
        pass

from numpy.linalg import solve
from scipy.integrate import nquad
from sklearn import datasets, preprocessing
from matplotlib import cm

class DensityEstimation(AreaOperation):
    def __init__(self, grid, data, dim, level):
        self.data = data
        self.dim = dim
        self.grid = grid
        self.level = level

    def perform_operation(self, level):
        self.calcAlphas(level, self.data)

    def getIndexList(self, levelvec):
        lengths = [2 ** l - 1 for l in levelvec]
        dim_lists = [range(1, n + 1) for n in lengths]
        return get_cross_product_list(dim_lists)

    def checkIfAdjacent(self, ivec, jvec):
        for i in range(len(ivec)):
            if abs(ivec[i] - jvec[i]) > 1:
                return False
        return True

    def getL2ScalarProduct(self, ivec, jvec, lvec):
        if self.checkIfAdjacent(ivec, jvec):
            dim = len(ivec)
            f = lambda x, y: (self.basisFunction(ivec, lvec, [x, y]) * self.basisFunction(jvec, lvec, [x, y]))
            print("-------------")
            print("calc")
            start = [(min(ivec[d], jvec[d]) - 1) * 2 ** (-lvec[d]) for d in range(dim)]
            end = [(max(ivec[d], jvec[d]) + 1) * 2 ** (-lvec[d]) for d in range(dim)]
            print(ivec, jvec)
            print(start, end)
            return nquad(f, [[start[d], end[d]] for d in range(dim)], opts={"epsabs": 10 ** (-15),
                                                                            "epsrel": 1 ** (
                                                                                -15)})
        else:
            print("-------------")
            print("skipped")
            return (0, 0)

    def calcDensityEstimation(self, levelvec, data, lambd=0):
        R = self.constructR(levelvec)
        R[np.diag_indices_from(R)] += lambd
        b = self.calcB(data, levelvec)
        return solve(R, b)

    def calcB(self, data, levelvec):
        M = len(data)
        N = np.prod(self.grid.levelToNumPoints(levelvec))
        b = np.empty(N)

        indexList = self.getIndexList(levelvec)
        for i in range(N):
            sum = 0
            for j in range(M):
                sum += self.basisFunction(indexList[i], levelvec, data[j])
            b[i] = ((1 / M) * sum)
            print((1 / M) * sum)
        return b

    def constructR(self, levelvec, lumping=False):
        numberPoints = np.prod(self.grid.levelToNumPoints(levelvec))
        R = np.zeros((numberPoints, numberPoints))
        indexList = self.getIndexList(levelvec)
        print(indexList, levelvec)
        diagVal, err = self.getL2ScalarProduct(indexList[0], indexList[0], levelvec)
        R[np.diag_indices_from(R)] += diagVal
        if lumping == False:
            for i in range(0, len(indexList) - 1):
                for j in range(i + 1, len(indexList)):
                    temp, err = self.getL2ScalarProduct(indexList[i], indexList[j], levelvec)
                    print(i, j)
                    print(temp)

                    if temp != 0:
                        R[i, j] = temp
                        R[j, i] = temp
        return R

    def calcAlphas(self, level, data, lambd=0):
        alphas = {}
        for i in range(level, 0, -1):
            for j in range(1, level + 1, 1):
                if i + j == level + 1:
                    print("Calculating grid add", (i, j), sep=" ")
                    alphas.update({(i, j): self.calcDensityEstimation((i, j), data, lambd)})
        for k in range(level - 1, 0, -1):
            for l in range(1, level, 1):
                if k + l == level:
                    print("Calculating grid sub", (k, l), sep=" ")
                    alphas.update({(k, l): self.calcDensityEstimation((k, l), data, lambd)})
        return alphas

    def preCalcAlphas(self, upToLevel, data, lambd=0):
        alphas = {}
        # alphas.update({1: {(1, 1): calcDensityEstimation((1, 1), data, lambd)}})
        for i in range(2, upToLevel + 1):
            alphas.update({i: self.calcAlphas(i, data, lambd)})
        return alphas

    def combiDensityEstimation(self, level, alphas, x):
        ret = 0
        for i in range(level, 0, -1):
            for j in range(1, level + 1, 1):
                if i + j == level + 1:
                    ret += self.weightedBasisFunction((i, j), alphas.get((i, j)), x)
        for k in range(level - 1, 0, -1):
            for l in range(1, level, 1):
                if k + l == level:
                    ret -= self.weightedBasisFunction((k, l), alphas.get((k, l)), x)
        return ret

    def basisFunction(self, ivec, lvec, x):
        dim = len(ivec)
        result = 1
        for d in range(dim):
            result *= max((1 - abs(2 ** lvec[d] * x[d] - ivec[d])), 0)
        return result

    def weightedBasisFunction(self, levelvec, alphas, x):
        indices = self.getIndexList(levelvec)
        sum = 0
        for i, index in enumerate(indices):
            # TODO test if i is in support of x, if not --> dont calculate
            sum += self.basisFunction(index, levelvec, x) * alphas[i]
        return sum

    def plot(self, pointsPerDim=100):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.data[0])
        dataTrans = scaler.transform(self.data[0])
        alphas = self.calcAlphas(self.level, dataTrans, 0)

        X = np.linspace(0.0, 1.0, pointsPerDim)
        Y = np.linspace(0.0, 1.0, pointsPerDim)
        X, Y = np.meshgrid(X, Y)

        Z = np.zeros_like(X)
        for i in range(pointsPerDim):
            for j in range(pointsPerDim):
                Z[i][j] = self.combiDensityEstimation(self.level, alphas, [X[i, j], Y[i, j]])

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(bottom=0.0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()
        plt.close(fig)

    def plotData(self):
        points = self.data[0]
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(points)
        dataTrans = scaler.transform(points)
        x, y = zip(*dataTrans)
        plt.scatter(x, y)
        plt.title("#points = %d" % len(points))
        plt.show()

from scipy.interpolate import interpn

class Integration(AreaOperation):
    def __init__(self, f: Function, grid: Grid, dim: int,  reference_solution: Sequence[float]=None):
        self.f = f
        self.f_actual = None
        self.grid = grid
        self.reference_solution = reference_solution
        self.dim = dim
        self.dict_integral = {}
        self.dict_points = {}
        self.integral = np.zeros(f.output_length())

    def get_distinct_points(self):
        return self.f.get_f_dict_size()

    def get_result(self):
        return self.integral

    def point_output_length(self):
        return self.f.output_length()

    def initialize(self):
        self.f.reset_dictionary()
        self.integral = np.zeros(self.f.output_length())

    def eval_analytic(self, coordinate: Tuple[float, ...]) -> Sequence[float]:
        return self.f.eval(coordinate)

    def add_value(self, combined_solution: Sequence[float], new_solution: Sequence[float], component_grid_info: ComponentGridInfo):
        return combined_solution + component_grid_info.coefficient * new_solution

    def evaluate_area(self, area, levelvector, componentgrid_info, refinement_container, additional_info):
        partial_integral = componentgrid_info.coefficient * self.grid.integrate(self.f, levelvector, area.start, area.end)
        if area.integral is None:
            area.integral = partial_integral
        else:
            area.integral += partial_integral
        evaluations = np.prod(self.grid.levelToNumPoints(levelvector))

        evaluations = np.prod(self.grid.levelToNumPoints(levelvector))
        if refinement_container is not None:
            refinement_container.integral += partial_integral
        return evaluations

    def evaluate_levelvec(self, component_grid):
        levelvector = component_grid.levelvector
        partial_integral = self.grid.integrate(self.f, levelvector, self.grid.a, self.grid.b)
        self.integral += partial_integral * component_grid.coefficient

    def evaluate_area_for_error_estimates(self, area, levelvector, componentgrid_info, refinement_container, additional_info):
        if additional_info.error_name == "extend_parent":
            assert additional_info.filter_area is None
            extend_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
            if area.parent_info.extend_parent_integral is None:
                area.parent_info.extend_parent_integral = extend_parent_new * componentgrid_info.coefficient
            else:
                area.parent_info.extend_parent_integral += extend_parent_new * componentgrid_info.coefficient
            return np.prod(self.grid.levelToNumPoints(levelvector))
        elif additional_info.error_name == "extend_error_correction":
            assert additional_info.filter_area is None
            if area.switch_to_parent_estimation:
                extend_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
                if area.parent_info.extend_error_correction is None:
                    area.parent_info.extend_error_correction = np.array(area.integral)
                area.parent_info.extend_error_correction -= extend_parent_new * componentgrid_info.coefficient
                return np.prod(self.grid.levelToNumPoints(levelvector))
            else:
                return 0
        elif additional_info.error_name == "split_no_filter":
                assert additional_info.filter_area is None
                split_parent_new = self.grid.integrate(self.f, levelvector, area.start, area.end)
                if additional_info.target_area.parent_info.split_parent_integral is None:
                    additional_info.target_area.parent_info.split_parent_integral = split_parent_new * componentgrid_info.coefficient
                else:
                    additional_info.target_area.parent_info.split_parent_integral += split_parent_new * componentgrid_info.coefficient
                return np.prod( self.grid.levelToNumPoints(levelvector))
        else:
            assert additional_info.filter_area is not None
            if not additional_info.interpolate:  # use filtering approach
                self.grid.setCurrentArea(area.start, area.end, levelvector)
                points, weights = self.grid.get_points_and_weights()
                integral = 0.0
                num_points = 0
                for i, (p, weight) in enumerate(zip(points, weights)):
                    if self.point_in_area(p, additional_info.filter_area):
                        integral += self.f(p) * weight * self.get_point_factor(p, additional_info.filter_area, area)
                        num_points += 1
            else:  # use bilinear interpolation to get function values in filter_area
                integral = 0.0
                num_points = 0
                # create grid with boundaries; if 0 boudnary condition we will fill boundary points with 0's
                boundary_save = self.grid.get_boundaries()
                self.grid.set_boundaries([True] * self.dim)
                self.grid.setCurrentArea(area.start, area.end, levelvector)
                self.grid.set_boundaries(boundary_save)

                # get corner grid in scipy data structure
                mesh_points_grid = [self.grid.coordinate_array[d] for d in range(self.dim)]

                # get points of filter area for which we want interpolated values
                self.grid.setCurrentArea(additional_info.filter_area.start, additional_info.filter_area.end, levelvector)
                points, weights = self.grid.get_points_and_weights()

                # bilinear interpolation
                interpolated_values = self.interpolate_points(mesh_points_grid, points)

                integral += np.inner(interpolated_values.T, weights)

                # calculate all mesh points
                mesh_points = list(
                    zip(*[g.ravel() for g in np.meshgrid(*[mesh_points_grid[d] for d in range(self.dim)])]))

                # count the number of mesh points that fall into the filter area
                for p in mesh_points:
                    if self.point_in_area(p, additional_info.filter_area) and self.grid.point_not_zero(p):
                        num_points += 1
            if additional_info.error_name == "split_parent":
                child_area = additional_info.filter_area
                if child_area.parent_info.split_parent_integral is None:
                    child_area.parent_info.split_parent_integral = 0.0
                child_area.parent_info.split_parent_integral += integral * componentgrid_info.coefficient
            else:
                if additional_info.error_name == "split_parent2":
                    child_area = additional_info.filter_area
                    if child_area.parent_info.split_parent_integral2 is None:
                        child_area.parent_info.split_parent_integral2 = 0.0
                    child_area.parent_info.split_parent_integral2 += integral * componentgrid_info.coefficient

                else:
                    assert additional_info.error_name == "reference"
            return num_points


    def get_best_fit(self, area, norm):
        old_value = area.parent_info.split_parent_integral
        new_value = area.parent_info.split_parent_integral2
        if old_value is None:
            area.parent_info.split_parent_integral = np.array(new_value)
        else:
            if new_value is None or LA.norm(abs(area.integral - old_value), norm) < LA.norm(abs(area.integral - new_value), norm):
                pass
            else:
                area.parent_info.split_parent_integral = np.array(area.parent_info.split_parent_integral2)

    def initialize_error_estimates(self,area):
        area.parent_info.split_parent_integral = None
        area.parent_info.split_parent_integral2 = None
        area.parent_info.extend_parent_integral = None
        if area.parent_info.benefit_split is None:
            area.parent_info.num_points_extend_parent = None
        if area.parent_info.benefit_extend is None:
            area.parent_info.num_points_split_parent = None
        area.parent_info.num_points_reference = None
        area.parent_info.extend_error_correction = None

    def initialize_split_error(self, area):
        area.parent_info.split_parent_integral = None
        area.parent_info.split_parent_integral2 = None
        if area.parent_info.benefit_extend is None:
            area.parent_info.num_points_split_parent = None

    def initialize_error(self, area, error_name):
        if error_name == "split_parent":
            area.parent_info.split_parent_integral = None
        if error_name == "split_parent2":
            area.parent_info.split_parent_integral2 = None
        if error_name == "extend_parent":
            area.parent_info.extend_parent_integral = None

    def initialize_point_numbers(self, area, error_name):
        if error_name == "split_parent":
            area.parent_info.num_points_split_parent = None
        if error_name == "split_parent2":
            area.parent_info.num_points_split_parent = None
        if error_name == "extend_parent":
            area.parent_info.num_points_extend_parent = None

    def get_previous_value_from_split_parent(self, area):
        area.parent_info.previous_value = area.parent_info.split_parent_integral

    def count_unique_points(self):
        return True

    def area_preprocessing(self, area):
        area.set_integral(0.0)
        #area.evaluations = 0
        #area.levelvec_dict = {}
        #area.error = None

    def get_global_error_estimate(self, refinement_container, norm):
        if self.reference_solution is None:
            return None
        elif LA.norm(self.reference_solution) == 0.0:
            return LA.norm(abs(refinement_container.integral), norm)
        else:
            return LA.norm(abs((self.reference_solution - refinement_container.integral)/self.reference_solution), norm)

    def area_postprocessing(self, area):
        area.value = np.array(area.integral)

    def get_sum_sibling_value(self, area):
        area.sum_siblings = 0.0
        i = 0
        for child in area.parent_info.parent.children:
            if child.integral is not None:
                area.sum_siblings += child.integral
                i += 1
        assert i == 2 ** self.dim or i == 2  # we always have 2**dim children

    def set_extend_benefit(self, area, norm):
        if area.parent_info.benefit_extend is not None:
            return
        if area.switch_to_parent_estimation:
            comparison = area.sum_siblings
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            comparison = area.integral
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_split_parent or area.switch_to_parent_estimation

        error_extend = LA.norm(abs((area.parent_info.split_parent_integral - comparison) / (abs(comparison) + 10 ** -100)), norm)
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_extend = error_extend * (area.parent_info.num_points_split_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_extend = error_extend * area.parent_info.num_points_split_parent

    def set_split_benefit(self, area, norm):
        if area.parent_info.benefit_split is not None:
            return
        if area.switch_to_parent_estimation:
            num_comparison = area.evaluations * 2 ** self.dim
        else:
            num_comparison = area.evaluations
        assert num_comparison > area.parent_info.num_points_extend_parent

        if self.grid.boundary:
            assert area.parent_info.num_points_split_parent > 0
        error_split = LA.norm(abs((area.parent_info.extend_parent_integral - area.integral) / (abs(area.integral) + 10 ** -100)), norm)
        if not self.grid.is_high_order_grid():
            area.parent_info.benefit_split = error_split * (area.parent_info.num_points_extend_parent - area.parent_info.num_points_reference)
        else:
            area.parent_info.benefit_split = error_split * area.parent_info.num_points_extend_parent

    def set_extend_error_correction(self, area, norm):
        if area.switch_to_parent_estimation:
            area.parent_info.extend_error_correction = LA.norm(area.parent_info.extend_error_correction, norm) * area.parent_info.num_points_split_parent

    def point_in_area(self, point, area):
        for d in range(self.dim):
            if point[d] < area.start[d] or point[d] > area.end[d]:
                return False
        return True

    def get_point_factor(self, point, area, area_parent):
        factor = 1.0
        for d in range(self.dim):
            if (point[d] == area.start[d] or point[d] == area.end[d]) and not (
                    point[d] == area_parent.start[d] or point[d] == area_parent.end[d]):
                factor /= 2.0
        return factor

    def initialize_global_value(self, refinement):
        refinement.integral = 0.0

    # interpolates the cell at the subcell edge points and evaluates the integral based on the trapezoidal rule
    def compute_subcell_with_interpolation(self, cell, subcell, coefficient, refinement_container):
        start_subcell = subcell.start
        end_subcell = subcell.end
        start_cell = cell.start
        end_cell = cell.end
        subcell_points = list(zip(*[g.ravel() for g in np.meshgrid(*[[start_subcell[d], end_subcell[d]] for d in range(self.dim)])]))
        corner_points_grid = [[start_cell[d], end_cell[d]] for d in range(self.dim)]
        interpolated_values = self.interpolate_points(corner_points_grid, subcell_points)
        width = np.prod(np.array(end_subcell) - np.array(start_subcell))
        factor = 0.5**self.dim * width
        integral = 0.0
        for p in interpolated_values:
            integral += p * factor
        subcell.cell_dict[subcell.get_key()].sub_integrals.append((integral, coefficient))
        subcell.integral += integral * coefficient
        if refinement_container is not None:
            refinement_container.integral += integral * coefficient

    def print_evaluation_output(self, refinement):
        combi_integral = refinement.integral
        if len(combi_integral) == 1:
            combi_integral = combi_integral[0]
        print("combiintegral:", combi_integral)

    def get_twin_error(self, d, area, norm):
        return LA.norm(abs(area.parent_info.parent.integral - (area.integral + area.twins[d].integral)))
        #return LA.norm(abs(area.integral - area.twins[d].integral), norm)

    def calculate_operation_dimension_wise(self, gridPointCoordsAsStripes, grid_point_levels, component_grid, start, end, reuse_old_values):
        reuse_old_values = False
        if reuse_old_values:
            previous_integral, previous_points = self.get_previous_integral_and_points(component_grid.levelvector)
            integral = np.array(previous_integral)
            previous_points_coarsened = list(previous_points)
            modification_points, modification_points_coarsen = self.get_modification_points(previous_points,
                                                                                            gridPointCoordsAsStripes)
            if modification_points_coarsen is not None:
                for d in range(self.dim):
                    previous_points_coarsened[d] = list(previous_points[d])
                    for mod_point in modification_points_coarsen[d]:
                        for removal_point in mod_point[1]:
                            previous_points_coarsened[d].remove(removal_point)
                integral += self.subtract_contributions(modification_points_coarsen, previous_points_coarsened,
                                                        previous_points)
                integral -= self.get_new_contributions(modification_points_coarsen, previous_points)
            if modification_points is not None:
                # ~ integral -= self.subtract_contributions(modification_points, previous_points_coarsened,
                                                        # ~ gridPointCoordsAsStripes)
                v = self.subtract_contributions(modification_points, previous_points_coarsened,
                                                        gridPointCoordsAsStripes)
                assert len(v) == len(integral)
                integral -= v
                integral += self.get_new_contributions(modification_points, gridPointCoordsAsStripes)
        else:
            self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            integral = self.grid.integrate(self.f, component_grid.levelvector, start, end)
        self.refinement_container.integral += integral * component_grid.coefficient
        self.dict_integral[tuple(component_grid.levelvector)] = np.array(integral)
        self.dict_points[tuple(component_grid.levelvector)] = np.array(gridPointCoordsAsStripes)
        return integral

    def compute_error_estimates_dimension_wise(self, gridPointCoordsAsStripes, grid_point_levels, children_indices, component_grid):
        self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        self.calculate_surplusses(gridPointCoordsAsStripes, children_indices, component_grid)

    def set_function(self, f=None):
        assert f is None or f == self.f, "Integration and the refinement should use the same function"

    def init_dimension_wise(self, grid, grid_surplusses, refinement_container, lmin, lmax, a, b, version = 2):
        self.grid = grid
        self.grid_surplusses = grid_surplusses
        self.refinement_container = refinement_container
        self.version = version
        self.lmin = lmin
        self.lmax = lmax
        self.a = a
        self.b = b

    def initialize_refinement_container_dimension_wise(self, refinement_container):
        refinement_container.integral = 0.0
    # This method calculates the surplus error estimates for a point by calculating dim-1 dimensional slices
    # through the domain along the child coordinates. We always calculate the 1-dimensional surplus for every point
    # on this slice.
    def calculate_surplusses(self, grid_points, children_indices, component_grid):
        tol = 10**-84
        if isinstance(self.grid_surplusses, GlobalBSplineGrid) or isinstance(self.grid_surplusses, GlobalLagrangeGrid):
            grid_values = np.empty((self.f.output_length(), np.prod(self.grid.numPoints)))
            points = self.grid.getPoints()
            for i, point in enumerate(points):
                grid_values[:, i] = self.f(point)
        for d in range(0, self.dim):
            k=0
            refinement_dim = self.refinement_container.get_refinement_container_for_dim(d)
            if isinstance(self.grid_surplusses, GlobalBSplineGrid) or isinstance(self.grid_surplusses, GlobalLagrangeGrid):
                hierarchization_operator = HierarchizationLSG(self.grid)
                surplusses_1d = hierarchization_operator.hierarchize_poles_for_dim(np.array(grid_values), self.grid.numPoints, self.f, d)
                surplus_pole = np.zeros((self.f.output_length(), self.grid.numPoints[d]))
                stride = int(np.prod(self.grid.numPoints[d+1:]))
                for j in range(self.grid.numPoints[d]):
                    i = j * stride
                    while i < np.prod(self.grid.numPoints):
                        surplus_pole[:,j] += np.sum(abs(surplusses_1d[:,i:i+stride])) #* weights[i:i+stride]))
                        i += stride * self.grid.numPoints[d]
                #print("surplus pole", surplus_pole, surplusses_1d, stride)
            for child_info in children_indices[d]:
                left_parent = child_info.left_parent
                right_parent = child_info.right_parent
                child = child_info.child
                if isinstance(self.grid_surplusses, GlobalBSplineGrid) or isinstance(self.grid_surplusses, GlobalLagrangeGrid):
                    index_child = grid_points[d].index(child) - int(not(self.grid.boundary))
                    volume = surplus_pole[:, index_child] / np.prod(self.grid.numPoints) * self.grid.numPoints[d] * self.grid.weights[d][index_child]
                    evaluations = np.prod(self.grid.numPoints) / self.grid.numPoints[d]
                else:
                    volume, evaluations = self.sum_up_volumes_for_point_vectorized(child_info=child_info, grid_points=grid_points, d=d)
                #print("surplus pole", surplus_pole[1], grid_values[d], child)
                #print(surplus_pole[grid_points[d].index(child)])


                k_old = 0
                if left_parent < 0:
                    factor_left = (1 + tol)
                else:
                    factor_left = (1 - tol)
                for i in range(refinement_dim.size() ):
                    if refinement_dim.get_object(i).start >= left_parent * factor_left:
                        k_old = i
                        break
                k = k_old
                refine_obj = refinement_dim.get_object(k)
                if right_parent < 0:
                    factor_right = (1 - tol)
                else:
                    factor_right = (1 + tol)
                if not (refine_obj.start >= left_parent * factor_left and refine_obj.end <= right_parent * factor_right):
                    for child_info in children_indices[d]:
                        print(child_info.left_parent, child_info.child, child_info.right_parent)
                #print(refine_obj.start, refine_obj.end, left_parent, right_parent)
                assert refine_obj.start >= left_parent * factor_left and refine_obj.end <= right_parent * factor_right
                max_level = 1
                while k < refinement_dim.size():
                    refine_obj = refinement_dim.get_object(k)
                    factor = 1 - tol if right_parent >= 0 else 1 + tol
                    if refine_obj.start >= right_parent * factor:
                        break
                    #refine_obj.print()
                    #print("Right parent", right_parent)
                    assert refine_obj.end <= right_parent * factor_right
                    k += 1
                    max_level = max(max_level, max(refine_obj.levels))
                for i in range(k_old, k):
                    refine_obj = refinement_dim.get_object(i)
                    num_area_in_support = (k-k_old)
                    # ~ fraction_of_support = (refine_obj.end - refine_obj.start)/(right_parent - left_parent)
                    modified_volume = volume/num_area_in_support ** 2 #/ 2**(max_level - log2((self.b[d] - self.a[d])/(right_parent - left_parent))) #/  (num_area_in_support)**2
                    # ~ assert fraction_of_support <= 1
                    #print(modified_volume, left_parent, child, right_parent, refine_obj.start, refine_obj.end, num_area_in_support, evaluations)
                    #if not self.combischeme.has_forward_neighbour(component_grid.levelvector):
                    #print(volume)
                    refine_obj.add_volume(modified_volume * component_grid.coefficient)
                    #refine_obj.add_evaluations(evaluations / num_area_in_support * component_grid.coefficient)
                    #assert component_grid.coefficient == 1
                     #* component_grid.coefficient)
                    #print("Dim:", d, refine_obj.start, refine_obj.end, refine_obj.volume, refine_obj.evaluations, child, left_parent, right_parent, volume, modified_volume)

                    '''
                    if refine_obj.start >= left_parent and refine_obj.end <= child: #and not child_info.has_left_child:
                        width_refinement = refine_obj.end - refine_obj.start
                        width_basis = right_parent - left_parent
                        refine_obj.add_volume(modified_volume)  # * width_refinement/ width_basis)

                    elif refine_obj.start >= child and refine_obj.end <= right_parent: #and not child_info.has_right_child:
                        width_refinement = refine_obj.end - refine_obj.start
                        width_basis = right_parent - left_parent
                        refine_obj.add_volume(modified_volume)  # * width_refinement/ width_basis)
                    else:
                        break
                    '''
                k = min(k, refinement_dim.size() - 1)

                '''
                if not child_info.has_right_child:
                    child_info.right_refinement_object.add_volume(volume / 2.0)
                    child_info.right_refinement_object.add_evaluations(evaluations / 2.0)
                if not child_info.has_left_child:
                    child_info.left_refinement_object.add_volume(volume/2.0)
                    child_info.left_refinement_object.add_evaluations(evaluations / 2.0)
                '''

    def get_surplus_width(self, d: int, right_parent: float, left_parent: float) -> float:
        return right_parent - left_parent

    # Sum up the 1-d surplusses along the dim-1 dimensional slice through the point child in dimension d.
    #  The surplusses are calculated based on the left and right parents.
    def sum_up_volumes_for_point(self, child_info, grid_points, d):
        #print(grid_points)
        child = child_info.child
        left_parent = child_info.left_parent
        right_parent = child_info.right_parent
        left_parent_of_left_parent = child_info.left_parent_of_left_parent
        right_parent_of_right_parent = child_info.right_parent_of_right_parent
        volume = 0.0
        assert right_parent > child > left_parent

        #npt.assert_almost_equal(right_parent - child, child - left_parent, decimal=12)

        # Searching close points does not work right when points have
        # low distance to each other.
        # ~ for p in grid_points[d]:
            # ~ if isclose(p, left_parent):
                # ~ left_parent = p
            # ~ if isclose(p, right_parent):
                # ~ right_parent = p
        index_left_parent = grid_points[d].index(left_parent) - 1 * int(not self.grid.boundary)
        index_child = grid_points[d].index(child) - 1 * int(not self.grid.boundary)
        index_right_parent = grid_points[d].index(right_parent) - 1 * int(not self.grid.boundary)

        left_parent_in_grid = self.grid_surplusses.boundary or not(isclose(left_parent, self.a[d]))
        right_parent_in_grid = self.grid_surplusses.boundary or not(isclose(right_parent, self.b[d]))
        # avoid evaluating on boundary points if grids has none
        if left_parent_in_grid:
            if isinf(right_parent):
                factor_left_parent = 1.0
            else:
                factor_left_parent = (right_parent - child)/(right_parent - left_parent)
            #points_left_parent = get_cross_product([self.grid_surplusses.coords[d2]if d != d2 else [left_parent] for d2 in range(self.dim)])
            #points_left_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2]if d != d2 else [left_parent] for d2 in range(self.dim)])]))
        if right_parent_in_grid:
            if isinf(left_parent):
                factor_right_parent= 1.0
            else:
                factor_right_parent = (child - left_parent)/(right_parent - left_parent)
            #points_right_parent = get_cross_product([self.grid_surplusses.coords[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])
            #points_right_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])]))
        points_children = get_cross_product([self.grid_surplusses.coords[d2] if d != d2 else [child] for d2 in range(self.dim)])
        #points_children = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [child] for d2 in range(self.dim)])]))
        indices = get_cross_product([range(len(self.grid_surplusses.coords[d2])) if d != d2 else [1] for d2 in range(self.dim)])
        #indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(self.grid_surplusses.coords[d2])) if d != d2 else None for d2 in range(self.dim)])]))
        for (point_child, index) in zip(points_children, indices):
            #index = indices[i]
            factor = np.prod([self.grid_surplusses.weights[d2][index[d2]] if d2 != d else 1 for d2 in range(self.dim)])
            #factor2 = np.prod([self.grid.weights[d2][index[d2]]  if d2 != d else self.grid.weights[d2][index_child] for d2 in range(self.dim)])
            if factor != 0:
                exponent = 1# if not self.do_high_order else 2
                #if factor2 != 0:
                value = self.f(point_child)
                #print(points_children[i], self.f.f_dict.keys())
                # avoid evaluating on boundary points if grids has none
                assert (tuple(point_child) in self.f.f_dict)

                if left_parent_in_grid:
                    point_left_parent = tuple(point_child[:d] + tuple([left_parent]) + point_child[d+1:])
                    if self.grid_surplusses.modified_basis and not right_parent_in_grid:
                        assert point_left_parent in self.f.f_dict or self.grid.weights[d][index_left_parent] == 0

                        left_of_left_parent = list(point_left_parent)
                        left_of_left_parent[d] = left_parent_of_left_parent
                        #print("Left of left:", left_of_left_parent, points_left_parent[i])
                        #value = (2 * self.f(points_children[i]) - self.f(points_left_parent[i]))/2
                        #assert (tuple(points_left_parent[i]) in self.f.f_dict)

                        if isclose(left_of_left_parent[d], self.a[d]):
                            value = self.f(point_child) - self.f(point_left_parent)
                        else:
                            m = (self.f(tuple(left_of_left_parent)) - self.f(point_left_parent)) / (
                                        left_parent_of_left_parent - left_parent)
                            previous_value_at_child = m * (child - left_parent) + self.f(point_left_parent)
                            value = self.f(point_child) - previous_value_at_child
                            #print("Hey", m, previous_value_at_child, value, (self.f(tuple(left_of_left_parent)) - self.f(points_left_parent[i])), (left_of_left_parent - left_parent))

                            assert(tuple(left_of_left_parent) in self.f.f_dict)

                    else:
                        assert point_left_parent in self.f.f_dict or self.grid.weights[d][index_left_parent] == 0

                        value -= factor_left_parent * self.f(point_left_parent)
                if right_parent_in_grid:
                    point_right_parent = tuple(point_child[:d] + tuple([right_parent]) + point_child[d+1:])
                    if self.grid_surplusses.modified_basis and not left_parent_in_grid:
                        assert point_right_parent in self.f.f_dict or self.grid.weights[d][index_right_parent] == 0

                        right_of_right_parent = list(point_right_parent)
                        right_of_right_parent[d] = right_parent_of_right_parent
                        #print("Right of right:", right_of_right_parent, points_right_parent[i])
                        #value = (2 * self.f(points_children[i]) - self.f(points_right_parent[i]))/2
                        #assert (tuple(points_right_parent[i]) in self.f.f_dict)
                        if isclose(right_of_right_parent[d], self.b[d]):
                            value = self.f(point_child) - self.f(point_right_parent)
                        else:
                            m = (self.f(tuple(right_of_right_parent)) - self.f(point_right_parent))  / (right_parent_of_right_parent - right_parent)
                            previous_value_at_child = m * (child - right_parent) + self.f(point_right_parent)
                            value = self.f(point_child) - previous_value_at_child
                            #print("Hey", m, previous_value_at_child, value, (self.f(tuple(right_of_right_parent)) - self.f(points_right_parent[i])), (right_of_right_parent - right_parent))
                            assert(tuple(right_of_right_parent) in self.f.f_dict)
                    else:
                        #print(points_right_parent[i], self.f.f_dict.keys())
                        assert point_right_parent in self.f.f_dict or self.grid.weights[d][index_right_parent] == 0
                        value -= factor_right_parent * self.f(point_right_parent)
                volume += factor * abs(value) * (self.get_surplus_width(d, right_parent, left_parent))**exponent
        if self.version == 0 or self.version == 2:
            evaluations = len(points_children) #* (1 + int(left_parent_in_grid) + int(right_parent_in_grid))
        else:
            evaluations = 0
        return abs(volume), evaluations

    # Sum up the 1-d surplusses along the dim-1 dimensional slice through the point child in dimension d.
    #  The surplusses are calculated based on the left and right parents.
    def sum_up_volumes_for_point_vectorized(self, child_info, grid_points, d):
        #print(grid_points)
        child = child_info.child
        left_parent = child_info.left_parent
        right_parent = child_info.right_parent
        left_parent_of_left_parent = child_info.left_parent_of_left_parent
        right_parent_of_right_parent = child_info.right_parent_of_right_parent
        assert right_parent > child > left_parent

        #npt.assert_almost_equal(right_parent - child, child - left_parent, decimal=12)

        for p in grid_points[d]:
            if isclose(p, left_parent):
                left_parent = p
            if isclose(p, right_parent):
                right_parent = p
        index_right_parent = grid_points[d].index(right_parent) - 1 * int(not self.grid.boundary)
        index_left_parent = grid_points[d].index(left_parent) - 1 * int(not self.grid.boundary)

        left_parent_in_grid = self.grid_surplusses.boundary or not(isclose(left_parent, self.a[d]))
        right_parent_in_grid = self.grid_surplusses.boundary or not(isclose(right_parent, self.b[d]))

        size_slize = np.prod([self.grid.numPoints[d2] if d2 != d else 1 for d2 in range(self.dim)])
        # avoid evaluating on boundary points if grids has none
        if left_parent_in_grid:
            if isinf(right_parent):
                factor_left_parent = 1.0
            else:
                factor_left_parent = (right_parent - child)/(right_parent - left_parent)
            points_left_parent = get_cross_product([self.grid_surplusses.coords[d2]if d != d2 else [left_parent] for d2 in range(self.dim)])
            #points_left_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2]if d != d2 else [left_parent] for d2 in range(self.dim)])]))
        if right_parent_in_grid:
            if isinf(left_parent):
                factor_right_parent= 1.0
            else:
                factor_right_parent = (child - left_parent)/(right_parent - left_parent)
            points_right_parent = get_cross_product([self.grid_surplusses.coords[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])
            #points_right_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])]))
        if self.grid_surplusses.modified_basis and not right_parent_in_grid:
            left_of_left_parents = get_cross_product([self.grid_surplusses.coords[d2] if d != d2 else [left_parent_of_left_parent] for d2 in range(self.dim)])
        if self.grid_surplusses.modified_basis and not left_parent_in_grid:
            right_of_right_parents = get_cross_product([self.grid_surplusses.coords[d2] if d != d2 else [right_parent_of_right_parent] for d2 in range(self.dim)])

        points_children = get_cross_product([self.grid_surplusses.coords[d2] if d != d2 else [child] for d2 in range(self.dim)])
        #points_children = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [child] for d2 in range(self.dim)])]))
        indices = get_cross_product([range(len(self.grid_surplusses.coords[d2])) if d != d2 else [1] for d2 in range(self.dim)])
        #indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(self.grid_surplusses.coords[d2])) if d != d2 else None for d2 in range(self.dim)])]))
        #index = indices[i]
        factors = np.asarray([np.prod([self.grid_surplusses.weights[d2][index[d2]] if d2 != d else 1 for d2 in range(self.dim)]) for index in indices]).reshape((size_slize, 1))
        #factor2 = np.prod([self.grid.weights[d2][index[d2]]  if d2 != d else self.grid.weights[d2][index_child] for d2 in range(self.dim)])
        exponent = 1# if not self.do_high_order else 2
        #if factor2 != 0:
        #print(points_children[i], self.f.f_dict.keys())
        # avoid evaluating on boundary points if grids has none
        #assert (tuple(point_child) in self.f.f_dict)
        point_values = np.asarray([self.f(p) for p in points_children])
        values = point_values
        if left_parent_in_grid:
            point_values_left_parent = np.asarray([self.f(p) for p in points_left_parent])
            #point_left_parent = tuple(point_child[:d] + tuple([left_parent]) + point_child[d+1:])
            if self.grid_surplusses.modified_basis and not right_parent_in_grid:
                #assert point_left_parent in self.f.f_dict or self.grid.weights[d][index_left_parent] == 0

                #left_of_left_parent = list(point_left_parent)
                #left_of_left_parent[d] = left_parent_of_left_parent
                #print("Left of left:", left_of_left_parent, points_left_parent[i])
                #value = (2 * self.f(points_children[i]) - self.f(points_left_parent[i]))/2
                #assert (tuple(points_left_parent[i]) in self.f.f_dict)

                if isclose(left_parent_of_left_parent, self.a[d]):
                    values = point_values - point_values_left_parent
                else:
                    point_values_left_of_left_parent = np.asarray([self.f(p) for p in left_of_left_parents])

                    m = (point_values_left_of_left_parent - point_values_left_parent) / (
                                left_parent_of_left_parent - left_parent)
                    previous_value_at_child = m * (child - left_parent) + point_values_left_parent
                    values = point_values - previous_value_at_child
                    #print("Hey", m, previous_value_at_child, value, (self.f(tuple(left_of_left_parent)) - self.f(points_left_parent[i])), (left_of_left_parent - left_parent))

                    #assert(tuple(left_of_left_parent) in self.f.f_dict)

            else:
                #assert point_left_parent in self.f.f_dict or self.grid.weights[d][index_left_parent] == 0

                values -= factor_left_parent * point_values_left_parent
        if right_parent_in_grid:
            point_values_right_parent = np.asarray([self.f(p) for p in points_right_parent])
            if self.grid_surplusses.modified_basis and not left_parent_in_grid:
                #assert point_right_parent in self.f.f_dict or self.grid.weights[d][index_right_parent] == 0

                #right_of_right_parent = list(point_right_parent)
                #right_of_right_parent[d] = right_parent_of_right_parent
                #print("Right of right:", right_of_right_parent, points_right_parent[i])
                #value = (2 * self.f(points_children[i]) - self.f(points_right_parent[i]))/2
                #assert (tuple(points_right_parent[i]) in self.f.f_dict)
                if isclose(right_parent_of_right_parent, self.b[d]):
                    values = point_values - point_values_right_parent
                else:
                    point_values_right_of_right_parent = np.asarray([self.f(p) for p in right_of_right_parents])
                    m = (point_values_right_of_right_parent - point_values_right_parent)  / (right_parent_of_right_parent - right_parent)
                    previous_value_at_child = m * (child - right_parent) + point_values_right_parent
                    values = point_values - previous_value_at_child
                    #print("Hey", m, previous_value_at_child, value, (self.f(tuple(right_of_right_parent)) - self.f(points_right_parent[i])), (right_of_right_parent - right_parent))
                    #assert(tuple(right_of_right_parent) in self.f.f_dict)
            else:
                #print(points_right_parent[i], self.f.f_dict.keys())
                #assert point_right_parent in self.f.f_dict or self.grid.weights[d][index_right_parent] == 0
                values -= factor_right_parent * point_values_right_parent
        #print("Values", values, np.sum(factors*abs(values), axis=0), factors * abs(values), np.shape(values), np.shape(factors))
        volume = np.sum(factors * abs(values), axis=0) * (self.get_surplus_width(d, right_parent, left_parent))**exponent
        #print("Volume", volume)
        if self.version == 0 or self.version == 2:
            evaluations = size_slize #* (1 + int(left_parent_in_grid) + int(right_parent_in_grid))
        else:
            evaluations = 0
        return abs(volume), evaluations

    # This method returns the previous integral approximation + the points contained in this grid for the given
    # component grid identified by the levelvector. In case the component grid is new, we search for a close component
    # grid with levelvector2 <= levelvector and return the respective previous integral and the points of the
    # previous grid.
    def get_previous_integral_and_points(self, levelvector):
        if tuple(levelvector) in self.dict_integral:
            return np.array(self.dict_integral[tuple(levelvector)]), np.array(self.dict_points[tuple(levelvector)])
        else:
            k = 1
            dimensions = []
            for d in range(self.dim):
                if self.lmax[d] - k > 0:
                    dimensions.append(d)
            while k < max(self.lmax):
                reduction_values = list(zip(*[g.ravel() for g in np.meshgrid(
                    *[range(0, min(k, self.lmax[d] - self.lmin[d])) for d in range(self.dim)])]))

                for value in reduction_values:
                    levelvec_temp = np.array(levelvector) - np.array(list(value))
                    if tuple(levelvec_temp) in self.dict_integral:
                        return np.array(self.dict_integral[tuple(levelvec_temp)]), np.array(self.dict_points[tuple(levelvec_temp)])
                k += 1
        assert False

    # This method checks if there are new points in the grid new_points compared to the old grid old_points
    # We then return a suited data structure containing the newly added points and the points that were removed.
    def get_modification_points(self, old_points, new_points):
        found_modification = found_modification2 = False
        # storage for newly added points per dimension
        modification_array_added = [[] for d in range(self.dim)]
        # storage for removed points per dimension
        modification_arra_removed = [[] for d in range(self.dim)]

        for d in range(self.dim):
            # get newly added points for dimension d
            modifications = sorted(list(set(new_points[d]) - set(old_points[d])))
            if len(modifications) != 0:
                found_modification = True
                modification_1D = self.get_modification_objects(modifications, new_points[d])
                modification_array_added[d].extend(list(modification_1D))
            #get removed points for dimension d
            modifications_coarsen = sorted(list(set(old_points[d]) - set(new_points[d])))
            if len(modifications_coarsen) != 0:
                found_modification2 = True
                modification_1D = self.get_modification_objects(modifications_coarsen, old_points[d])
                modification_arra_removed[d].extend(list(modification_1D))
        return modification_array_added if found_modification else None, modification_arra_removed if found_modification2 else None

    # Construct the data structures for the newly added points listed in modifications. The complete grid is given in
    # grid_points.
    def get_modification_objects(self, modifications, grid_points):
        modification_1D = []
        k = 0
        for i in range(len(grid_points)):
            if grid_points[i] == modifications[k]:
                j = 1
                # get consecutive list of points that are newly added
                while k + j < len(modifications) and grid_points[i + j] == modifications[k + j]:
                    j += 1
                # store left and right neighbour in addition to the newly added points list(grid_points[i:i + j])
                modification_1D.append((grid_points[i - 1], list(grid_points[i:i + j]), grid_points[i + j]))
                k += j
                if k == len(modifications):
                    break
        return modification_1D

    # This method calculates the change of the integral contribution of the neighbouring points of newly added points.
    # We assume here a trapezoidal rule. The newly added points are contained in new_points but not in old_points.
    def subtract_contributions(self, modification_points, old_points, new_points):
        # calculate weights of point in new grid
        self.grid_surplusses.set_grid(new_points)
        weights = self.grid_surplusses.weights
        # save weights in dictionary for fast access via coordinate
        dict_weights_fine = [{} for d in range(self.dim)]
        for d in range(self.dim):
            for p, w in zip(self.grid_surplusses.coords[d], weights[d]):
                dict_weights_fine[d][p] = w
        # reset grid to old grid
        self.grid_surplusses.set_grid(old_points)
        # sum up the changes in contributions
        integral = 0.0
        for d in range(self.dim):
            for point in modification_points[d]:
                # calculate the changes in contributions for all points that contain the neighbouring points point[0]
                # and point[2] in dimension d
                points_for_slice = list([point[0],point[2]])
                # remove boundary points if contained if grid has no boundary points
                if not self.grid.boundary:
                    points_for_slice = [p for p in points_for_slice if not(isclose(p, self.a[d]) or isclose(p, self.b[d]))]
                integral += self.calc_slice_through_points(points_for_slice, old_points, d, modification_points, subtract_contribution=True, dict=dict_weights_fine)
        return integral

    # This method calculates the new contributions of the points specified in modification_points to the grid new_points
    # The new_points grid contains the newly added points.
    def get_new_contributions(self, modification_points, new_points):
        self.grid_surplusses.set_grid(new_points)
        # sum up all new contributions
        integral = 0.0
        for d in range(self.dim):
            for point in modification_points[d]:
                # calculate the new contribution of the points with the new coordinates points[1] (a list of one or
                # multiple new coordinates) in dimension d
                integral += self.calc_slice_through_points(point[1], new_points, d, modification_points)
        return integral

    # This method computes the integral of the dim-1 dimensional slice through the points_for_slice of dimension d.
    # We also account for the fact that some points might be traversed by multiple of these slice calculations and
    # reduce the factors accordingly. If subtract_contribution is set we calculate the difference of the
    # new contribution from previously existing points to the new points.
    def calc_slice_through_points(self, points_for_slice, grid_points, d, modification_points, subtract_contribution=False, dict=None):
        integral = 0.0
        positions = [list(self.grid_surplusses.coords[d]).index(point) for point in points_for_slice]
        points = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else points_for_slice for d2 in range(self.dim)])]))
        indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(self.grid_surplusses.coords[d2])) if d != d2 else positions for d2 in range(self.dim)])]))
        for i in range(len(points)):
            # index of current point in grid_points grid
            index = indices[i]
            #point coordinates of current point
            current_point = points[i]
            # old weight of current point in coarser grid
            weight = self.grid_surplusses.getWeight(index)
            if subtract_contribution:
                # weight of current point in new finer grid
                weight_fine = 1
                for d in range(self.dim):
                    weight_fine *= dict[d][current_point[d]]
                number_of_dimensions_that_intersect = 0
                # calculate if other slices also contain this point
                for d2 in range(self.dim):
                    for mod_point in modification_points[d2]:
                        if current_point[d2] == mod_point[0] or current_point[d2] == mod_point[2]:
                            number_of_dimensions_that_intersect += 1
                # calculate the weight difference from the old to the new grid
                factor = (weight - weight_fine)/number_of_dimensions_that_intersect
            else:
                number_of_dimensions_that_intersect = 1
                # calculate if other slices also contain this point
                for d2 in range(self.dim):
                    if d2 == d:
                        continue
                    for mod_point in modification_points[d2]:
                        if current_point[d2] in mod_point[1]:
                            number_of_dimensions_that_intersect += 1
                # calculate the new weight contribution of newly added point
                factor = weight / number_of_dimensions_that_intersect
            assert(factor >= 0)
            integral += self.f(current_point) * factor
        return integral

class Interpolation(Integration):
    # interpolates mesh_points_grid at the given  evaluation_points using bilinear interpolation
    @staticmethod
    def interpolate_points(f: Function, dim: int, grid: Grid, mesh_points_grid: Sequence[Sequence[float]], evaluation_points: Sequence[Tuple[float,...]]):
        # constructing all points from mesh definition
        mesh_points = get_cross_product(mesh_points_grid)
        function_value_dim = f.output_length()
        # calculate function values at mesh points and transform  correct data structure for scipy
        values = np.array([f(p) if grid.point_not_zero(p) else np.zeros(function_value_dim) for p in mesh_points])
        interpolated_values_array = []
        for d in range(function_value_dim):
            values_1D = np.asarray([value[d] for value in values])

            values_1D = values_1D.reshape(*[len(mesh_points_grid[d]) for d in (range(dim))])

            # interpolate evaluation points from mesh points with bilinear interpolation
            interpolated_values = interpn(mesh_points_grid, values_1D, evaluation_points, method='linear')

            interpolated_values = np.asarray([[value] for value in interpolated_values])
            interpolated_values_array.append(interpolated_values)
        return np.hstack(interpolated_values_array)


import chaospy as cp
import scipy.stats as sps
from Function import *
from StandardCombi import *  # For reference solution calculation

class UncertaintyQuantification(Integration):
    # The constructor resembles Integration's constructor;
    # it has an additional parameter:
    # distributions can be a list, tuple or string
    def __init__(self, f, distributions, a: Sequence[float], b: Sequence[float],
            dim: int=None, grid=None, reference_solution=None):
        dim = dim or len(a)
        super().__init__(f, grid, dim, reference_solution)
        self.f_model = f
        # If distributions is not a list, it specifies the same distribution
        # for every dimension
        if not isinstance(distributions, list):
            distributions = [distributions for _ in range(dim)]

        # Setting the distribution to a string is a short form when
        # no parameters are given
        for d in range(dim):
            if isinstance(distributions[d], str):
                distributions[d] = (distributions[d],)

        self._prepare_distributions(distributions, a, b)
        self.f_evals = None
        self.gPCE = None
        self.pce_polys = None

    def set_grid(self, grid): self.grid = grid

    def set_reference_solution(self, reference_solution):
        self.reference_solution = reference_solution

    # From the user provided information about distributions, this function
    # creates the distributions list which contains Chaospy distributions
    def _prepare_distributions(self, distris, a: Sequence[float],
            b: Sequence[float]):
        self.distributions = []
        self.distribution_infos = distris
        chaospy_distributions = []
        known_distributions = dict()
        for d in range(self.dim):
            distr_info = distris[d]
            distr_known = distr_info in known_distributions
            if distr_known:
                # Reuse the same distribution objects for multiple dimensions
                d_prev = known_distributions[distr_info]
                self.distributions.append(self.distributions[d_prev])
            else:
                known_distributions[distr_info] = d

            distr_type = distr_info[0]
            if distr_type == "Uniform":
                distr = cp.Uniform(a[d], b[d])
                chaospy_distributions.append(distr)
                if not distr_known:
                    self.distributions.append(UQDistribution.from_chaospy(distr))
            elif distr_type == "Triangle":
                midpoint = distr_info[1]
                assert isinstance(midpoint, float), "invalid midpoint"
                distr = cp.Triangle(a[d], midpoint, b[d])
                chaospy_distributions.append(distr)
                if not distr_known:
                    self.distributions.append(UQDistribution.from_chaospy(distr))
            elif distr_type == "Normal":
                mu = distr_info[1]
                sigma = distr_info[2]
                cp_distr = cp.Normal(mu=mu, sigma=sigma)
                chaospy_distributions.append(cp_distr)
                if not distr_known:
                    # The chaospy normal distribution does not work with big values
                    def pdf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.pdf(x, loc=_mu, scale=_sigma)
                    def cdf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.cdf(x, loc=_mu, scale=_sigma)
                    def ppf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.ppf(x, loc=_mu, scale=_sigma)
                    self.distributions.append(UQDistribution(pdf, cdf, ppf))
            elif distr_type == "Laplace":
                mu = distr_info[1]
                scale = distr_info[2]
                cp_distr = cp.Laplace(mu=mu, scale=scale)
                chaospy_distributions.append(cp_distr)
                if not distr_known:
                    def pdf(x, _mu=mu, _scale=scale):
                        return sps.laplace.pdf(x, loc=_mu, scale=_scale)
                    def cdf(x, _mu=mu, _scale=scale):
                        return sps.laplace.cdf(x, loc=_mu, scale=_scale)
                    def ppf(x, _mu=mu, _scale=scale):
                        return sps.laplace.ppf(x, loc=_mu, scale=_scale)
                    self.distributions.append(UQDistribution(pdf, cdf, ppf))
            else:
                assert False, "Distribution not implemented: " + distr_type
        self.distributions_chaospy = chaospy_distributions
        self.distributions_joint = cp.J(*chaospy_distributions)
        self.all_uniform = all(k[0] == "Uniform" for k in known_distributions)
        self.a = a
        self.b = b

    def get_surplus_width(self, d: int, right_parent: float, left_parent: float) -> float:
        # Approximate the width with the probability
        cdf = self.distributions[d].cdf
        return cdf(right_parent) - cdf(left_parent)

    # reuse_old_values does not work for multidimensional function output,
    # so this method is overridden here
    def calculate_operation_dimension_wise(self, gridPointCoordsAsStripes, grid_point_levels, component_grid, start, end, _reuse_old_values=False):
        self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        integral = self.grid.integrator(self.f, self.grid.numPoints, start, end)
        self.refinement_container.integral += integral * component_grid.coefficient
        return integral

    # This function exchanges the operation's function so that the adaptive
    # refinement can use a different function than the operation's function
    def set_function(self, f=None):
        if f is None:
            self.f = self.f_actual
            self.f_actual = None
        else:
            assert self.f_actual is None
            self.f_actual = self.f
            self.f = f

    def get_distributions(self): return self.distributions
    def get_distributions_chaospy(self): return self.distributions_chaospy

    # This function returns boundaries for distributions which have an infinite
    # domain, such as normal distribution
    def get_boundaries(self, tol: float) -> Tuple[Sequence[float], Sequence[float]]:
        assert 1.0 - tol < 1.0, "Tolerance is too small"
        a = []
        b = []
        for d in range(self.dim):
            dist = self.distributions[d]
            a.append(dist.ppf(tol))
            b.append(dist.ppf(1.0 - tol))
        return a, b

    def _set_pce_polys(self, polynomial_degrees):
        if self.pce_polys is not None and self.polynomial_degrees == polynomial_degrees:
            return
        self.polynomial_degrees = polynomial_degrees
        if not hasattr(polynomial_degrees, "__iter__"):
            self.pce_polys, self.pce_polys_norms = cp.orth_ttr(polynomial_degrees, self.distributions_joint, retall=True)
            return

        # Chaospy does not support different degrees for each dimension, so
        # the higher degree polynomials are removed afterwards
        polys, norms = cp.orth_ttr(max(polynomial_degrees), self.distributions_joint, retall=True)
        polys_filtered, norms_filtered = [], []
        for i,poly in enumerate(polys):
            max_exponents = [max(exps) for exps in poly.exponents.T]
            if any([max_exponents[d] > deg_max for d, deg_max in enumerate(polynomial_degrees)]):
                continue
            polys_filtered.append(poly)
            norms_filtered.append(norms[i])
        self.pce_polys = cp.Poly(polys_filtered)
        self.pce_polys_norms = norms_filtered

    def _scale_values(self, values):
        assert self.all_uniform, "Division by the domain volume should be used for uniform distributions only"
        div = 1.0 / np.prod([self.b[i] - v_a for i,v_a in enumerate(self.a)])
        return values * div

    def _set_nodes_weights_evals(self, combiinstance, scale_weights=False):
        self.nodes, self.weights = combiinstance.get_points_and_weights()
        assert len(self.nodes) == len(self.weights)
        if scale_weights:
            assert combiinstance.has_basis_grid(), "scale_weights should only be needed for basis grids"
            self.weights = self._scale_values(self.weights)
            # ~ self.f_evals = combiinstance.get_surplusses()
            # Surpluses are required here..
            self.f_evals = [self.f(coord) for coord in self.nodes]
        else:
            self.f_evals = [self.f(coord) for coord in self.nodes]

    def _get_combiintegral(self, combiinstance, scale_weights=False):
        integral = combiinstance.get_calculated_solution()
        if scale_weights:
            assert combiinstance.has_basis_grid(), "scale_weights should only be needed for basis grids"
            return self._scale_values(integral)
        return integral

    def calculate_moment(self, combiinstance, k: int=None,
            use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            mom = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            assert len(mom) == self.f.output_length()
            return mom
        self._set_nodes_weights_evals(combiinstance)
        vals = [self.f_evals[i] ** k * self.weights[i] for i in range(len(self.f_evals))]
        return sum(vals)

    def calculate_expectation(self, combiinstance, use_combiinstance_solution=True):
        return self.calculate_moment(combiinstance, k=1, use_combiinstance_solution=use_combiinstance_solution)

    @staticmethod
    def moments_to_expectation_variance(mom1: Sequence[float],
            mom2: Sequence[float]) -> Tuple[Sequence[float], Sequence[float]]:
        expectation = mom1
        variance = [mom2[i] - ex * ex for i, ex in enumerate(expectation)]
        for i, v in enumerate(variance):
            if v < 0.0:
                # When the variance is zero, it can be set to something negative
                # because of numerical errors
                variance[i] = -v
        return expectation, variance

    def calculate_expectation_and_variance(self, combiinstance, use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            integral = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            output_dim = len(integral) // 2
            expectation = integral[:output_dim]
            expectation_of_squared = integral[output_dim:]
        else:
            expectation = self.calculate_moment(combiinstance, k=1, use_combiinstance_solution=False)
            expectation_of_squared = self.calculate_moment(combiinstance, k=2, use_combiinstance_solution=False)
        return self.moments_to_expectation_variance(expectation, expectation_of_squared)

    def calculate_PCE(self, polynomial_degrees, combiinstance, restrict_degrees=False, use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            assert self.pce_polys is not None
            assert not restrict_degrees
            integral = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            num_polys = len(self.pce_polys)
            output_dim = len(integral) // num_polys
            coefficients = integral.reshape((num_polys, output_dim))
            self.gPCE = np.transpose(cp.poly.sum(self.pce_polys * coefficients.T, -1))
            return

        self._set_nodes_weights_evals(combiinstance)

        if restrict_degrees:
            # Restrict the polynomial degrees if in some dimension not enough points
            # are available
            # For degree deg, deg+(deg-1)+1 points should be available
            num_points = combiinstance.get_num_points_each_dim()
            polynomial_degrees = [min(polynomial_degrees, num_points[d] // 2) for d in range(self.dim)]

        self._set_pce_polys(polynomial_degrees)
        self.gPCE = cp.fit_quadrature(self.pce_polys, list(zip(*self.nodes)),
            self.weights, np.asarray(self.f_evals), norms=self.pce_polys_norms)

    def get_gPCE(self): return self.gPCE

    def get_expectation_PCE(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.E(self.gPCE, self.distributions_joint)

    def get_variance_PCE(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Var(self.gPCE, self.distributions_joint)

    def get_expectation_and_variance_PCE(self):
        return self.get_expectation_PCE(), self.get_variance_PCE()

    def get_Percentile_PCE(self, q: float, sample: int=10000):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Perc(self.gPCE, q, self.distributions_joint, sample)

    def get_first_order_sobol_indices(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Sens_m(self.gPCE, self.distributions_joint)

    def get_total_order_sobol_indices(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Sens_t(self.gPCE, self.distributions_joint)

    # Returns a Function which can be passed to performSpatiallyAdaptiv
    # so that adapting is optimized for the k-th moment
    def get_moment_Function(self, k: int):
        if k == 1:
            return self.f
        return FunctionPower(self.f, k)

    def set_moment_Function(self, k: int):
        self.f = self.get_moment_Function(k)

    # Optimizes adapting for multiple moments at once
    def get_moments_Function(self, ks: Sequence[int]):
        return FunctionConcatenate([self.get_moment_Function(k) for k in ks])

    def set_moments_Function(self, ks: Sequence[int]):
        self.f = self.get_moments_Function(ks)

    def get_expectation_variance_Function(self):
        return self.get_moments_Function([1, 2])

    def set_expectation_variance_Function(self):
        self.f = self.get_expectation_variance_Function()

    # Returns a Function which can be passed to performSpatiallyAdaptiv
    # so that adapting is optimized for the PCE
    def get_PCE_Function(self, polynomial_degrees):
        self._set_pce_polys(polynomial_degrees)
        # self.f can change, so putting it to a local variable is important
        # ~ f = self.f
        # ~ polys = self.pce_polys
        # ~ funcs = [(lambda coords: f(coords) * polys[i](coords)) for i in range(len(polys))]
        # ~ return FunctionCustom(funcs)
        return FunctionPolysPCE(self.f, self.pce_polys, self.pce_polys_norms)

    def set_PCE_Function(self, polynomial_degrees):
        self.f = self.get_PCE_Function(polynomial_degrees)

    def get_pdf_Function(self):
        pdf = self.distributions_joint.pdf
        return FunctionCustom(lambda coords: float(pdf(coords)))

    def set_pdf_Function(self):
        self.f = self.get_pdf_Function()

    # Returns a Function which applies the PPF functions before evaluating
    # the problem function; it can be integrated without weighting
    def get_inverse_transform_Function(self, func=None):
        return FunctionInverseTransform(func or self.f, self.distributions)

    def set_inverse_transform_Function(self, func=None):
        self.f = self.get_inverse_transform_Function(func or self.f, self.distributions)

# UncertaintyQuantification extended for testing purposes
class UncertaintyQuantificationTesting(UncertaintyQuantification):
    # This function uses the quadrature provided by Chaospy.
    def calculate_PCE_chaospy(self, polynomial_degrees, num_quad_points):
        self._set_pce_polys(polynomial_degrees)
        nodes, weights = cp.generate_quadrature(num_quad_points,
            self.distributions_joint, rule="G")
        f_evals = [self.f(c) for c in zip(*nodes)]
        self.gPCE = cp.fit_quadrature(self.pce_polys, nodes, weights, np.asarray(f_evals), norms=self.pce_polys_norms)

    def calculate_expectation_and_variance_for_weights(self, nodes, weights):
        f_evals = np.array([self.f(c) for c in zip(*nodes)])
        f_evals_squared = np.array([v ** 2 for v in f_evals])
        expectation = np.inner(f_evals.T, weights)
        expectation_of_squared = np.inner(f_evals_squared.T, weights)
        return self.moments_to_expectation_variance(expectation, expectation_of_squared)

    def calculate_expectation_and_variance_reference(self, mode="ChaospyHalton", modeparams=None):
        if mode == "ChaospyHalton":
            num_points = modeparams or 2**14
            nodes = self.distributions_joint.sample(num_points, rule="H")
            num_samples = len(nodes[0])
            assert num_points == num_samples
            w = 1.0 / num_samples
            weights = np.array([w for _ in range(num_samples)])
        elif mode == "ChaospyGauss":
            nodes, weights = cp.generate_quadrature(29,
                self.distributions_joint, rule="G")
        elif mode == "StandardcombiGauss":
            if all([distr[0] == "Normal" for distr in self.distribution_infos]):
                expectations = [distr[1] for distr in self.distribution_infos]
                standard_deviations = [distr[2] for distr in self.distribution_infos]
                grid = GaussHermiteGrid(expectations, standard_deviations)
                # ~ combiinstance = StandardCombi(self.a, self.b, grid=grid, operation=self)
                combiinstance = StandardCombi(self.a, self.b, grid=grid)
                combiinstance.perform_combi(1, 4, self.get_expectation_variance_Function())
                combiinstance.print_resulting_combi_scheme(markersize=5)
                combiinstance.print_resulting_sparsegrid(markersize=10)
            elif self.all_uniform:

                grid = GaussLegendreGrid(self.a, self.b, self.dim)
                # ~ combiinstance = StandardCombi(self.a, self.b, grid=grid, operation=self)
                combiinstance = StandardCombi(self.a, self.b, grid=grid)
                combiinstance.perform_combi(1, 4, self.get_expectation_variance_Function())
                combiinstance.print_resulting_combi_scheme(markersize=5)
                combiinstance.print_resulting_sparsegrid(markersize=10)
            else:
                assert False, "Not implemented"
        else:
            assert False, mode
        return self.calculate_expectation_and_variance_for_weights(nodes, weights)

    def calculate_multiple_expectation_and_variance(self, solutions):
        evals = sorted(solutions)
        expectation_variances = []
        for k in evals:
            integral = solutions[k]
            output_dim = len(integral) // 2
            mom1 = integral[:output_dim]
            mom2 = integral[output_dim:]
            expectation, variance = self.moments_to_expectation_variance(mom1, mom2)
            expectation_variances.append((k, expectation, variance))
        return expectation_variances

    @staticmethod
    def sort_multiple_solutions(solutions):
        return [(num_evals, solutions[num_evals]) for num_evals in sorted(solutions)]

    def calculate_multiple_expectation_and_variance(self, solutions):
        expectation_variances = []
        for num_evals, integral in self.sort_multiple_solutions(solutions):
            output_dim = len(integral) // 2
            mom1 = integral[:output_dim]
            mom2 = integral[output_dim:]
            expectation, variance = self.moments_to_expectation_variance(mom1, mom2)
            expectation_variances.append((num_evals, expectation, variance))
        return expectation_variances

    def calculate_PCE_from_multiple(self, combiinstance, integrals):
        combiinstance.calculated_solution = integrals
        return self.calculate_PCE(None, combiinstance)


from scipy import integrate

class UQDistribution:
    def __init__(self, pdf, cdf, ppf):
        self.pdf = pdf
        self.cdf = cdf
        self.ppf = ppf
        self.cached_moments = [dict() for _ in range(2)]
        self.cache_gauss_quad = dict()
        # ~ self.cache_integrals = dict()

    @staticmethod
    def from_chaospy(cp_distr):
        # The inverse Rosenblatt transformation is the inverse cdf here
        return UQDistribution(cp_distr.pdf, cp_distr.cdf,
            lambda x: float(cp_distr.inv(x)))

    def get_zeroth_moment(self, x1: float, x2: float):
        cache = self.cached_moments[0]
        if (x1, x2) in cache:
            return cache[(x1, x2)]
        moment_0 = self.cdf(x2) - self.cdf(x1)
        cache[(x1, x2)] = moment_0
        return moment_0

    def get_first_moment(self, x1: float, x2: float):
        cache = self.cached_moments[1]
        if (x1, x2) in cache:
            return cache[(x1, x2)]
        moment_1 = integrate.quad(lambda x: x * self.pdf(x), x1, x2,
            epsrel=10 ** -2, epsabs=np.inf)[0]
        cache[(x1, x2)] = moment_1
        return moment_1

    # Returns single-dimensional quadrature points and weights
    # for the high order grid
    def get_quad_points_weights(self, num_quad_points: int, cp_distribution) -> Tuple[Sequence[float], Sequence[float]]:
        cache = self.cache_gauss_quad
        if num_quad_points in cache:
            return cache[num_quad_points]
        (coords,), weights = cp.generate_quadrature(num_quad_points, cp_distribution, rule="G")
        cache[num_quad_points] = (coords, weights)
        return coords, weights

    # Calculates the weighted integral of an arbitrary function
    # between x1 and x2
    def calculate_integral(self, func, x1: float, x2: float):
        # ~ k = (func, x1, x2)
        # ~ cache = self.cache_integrals
        # ~ if k in cache:
            # ~ print("Cache match")
            # ~ return cache[k]
        integral = integrate.quad(lambda x: func(x) * self.pdf(x), x1, x2)[0]
        # ~ cache[k] = integral
        return integral
