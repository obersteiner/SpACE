import abc, logging
# Python modules
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import numpy as np
import scipy as sp
import scipy.integrate
from scipy.interpolate import interpn
import scipy.special
import math
import time
from RefinementContainer import *
from RefinementObject import *
from combiScheme import *
from Grid import *
from ErrorCalculator import *
from Function import *
from StandardCombi import *
from numpy import linalg as LA

# This class defines the general interface and functionalties of all spatially adaptive refinement strategies
class SpatiallyAdaptivBase(StandardCombi):
    def __init__(self, a, b, grid=None, operation=None, norm=np.inf):
        self.log = logging.getLogger(__name__)
        self.dim = len(a)
        self.a = a
        self.b = b
        self.grid = grid
        self.refinements_for_recalculate = 100
        self.operation = operation
        self.norm = norm
        self.margin = 0.9
        assert (len(a) == len(b))

    # returns the number of points in a single component grid with refinement
    def get_num_points_component_grid(self, levelvec, do_naive, num_sub_diagonal):
        array2 = self.get_points_component_grid(levelvec, num_sub_diagonal)
        if do_naive:
            array2new = array2
        else:  # remove points that appear in the list multiple times
            array2new = list(set(array2))
        # print(len(array2new))
        return len(array2new)

    def evaluate_final_combi(self):
        combiintegral = 0
        dim = self.dim
        # print "Dim:",dim
        num_evaluations = 0
        for component_grid in self.scheme:
            integral = 0
            for area in self.get_areas():
                area_integral, partial_integrals, evaluations = self.evaluate_area(self.f, area, component_grid.levelvector)
                if area_integral != -2 ** 30:
                    num_evaluations += evaluations
                    integral += area_integral
            integral *= component_grid.coefficient
            combiintegral += integral
        return combiintegral, num_evaluations

    def init_adaptive_combi(self, f, minv, maxv, refinement_container, tol):
        self.tolerance = tol
        self.f = f
        if self.realIntegral is not None:
            print("Reference solution:", self.realIntegral)
        else:
            print("No reference solution present. Working purely on surplus error estimates.")
        if (refinement_container == []):  # initialize refinement
            self.lmin = [minv for i in range(self.dim)]
            self.lmax = [maxv for i in range(self.dim)]
            # calculate the combination scheme
            self.combischeme = CombiScheme(self.dim)
            self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0])
            self.initialize_refinement()
            self.f.reset_dictionary()
        else:  # use the given refinement; in this case reuse old lmin and lmax and finestWidth; works only if there was no other run in between on same object
            self.refinement = refinement_container
            self.refinement.reinit_new_objects()
        # initialize values
        self.refinements = 0
        # self.combiintegral = 0
        # self.subAreaIntegrals = []
        self.counter = 1
        # self.evaluationsTotal = 0 #number of evaluations in current grid
        # self.evaluationPerArea = [] #number of evaluations per area

    def evaluate_integral(self):
        if self.operation is not None:
            return self.evaluate_operation()
        # initialize values
        # number_of_evaluations = 0
        # get tuples of all the combinations of refinement to access each subarea (this is the same for each component grid)
        areas = self.get_new_areas()
        integralarrayComplete = np.zeros((len(areas), len(self.f((self.b+self.a)*0.5))))
        evaluation_array = np.zeros(len(areas))
        # calculate integrals
        for component_grid in self.scheme:  # iterate over component grids
            # iterate over all areas and calculate the integral
            for k, area in enumerate(areas):
                # print(component_grid)
                area_integral, partial_integrals, evaluations = self.evaluate_area(self.f, area, component_grid)
                if area_integral is not None and area_integral[0] != -2 ** 30:
                    if partial_integrals is not None:  # outdated
                        pass
                        # integralArrayIndividual.extend(partial_integrals)
                    else:
                        integralarrayComplete[k] += component_grid.coefficient * area_integral
                        # self.combiintegral += area_integral * component_grid[1]
                        factor = component_grid.coefficient if self.grid.isNested() else 1
                        evaluation_array[k] += evaluations * factor
        self.finalize_evaluation()
        for k in range(len(integralarrayComplete)):
            i = k + self.refinement.size() - self.refinement.new_objects_size()
            self.refinement.set_integral(i, integralarrayComplete[k])
            self.refinement.set_evaluations(i, evaluation_array[k])
        for k in range(len(integralarrayComplete)):
            i = k + self.refinement.size() - self.refinement.new_objects_size()
            self.calc_error(i, self.f)
            self.refinement.set_benefit(i)

        # getArea with maximal error
        self.benefit_max = self.refinement.get_max_benefit()
        self.total_error = self.refinement.get_total_error()
        if self.print_output:
            print("max surplus error:", self.benefit_max, "total surplus error:", self.total_error)
            print("combiintegral:", self.refinement.integral[0] if len(self.refinement.integral) == 1 else self.refinement.integral)
        if self.realIntegral is not None:
            return LA.norm(abs(self.refinement.integral - self.realIntegral) / abs(self.realIntegral), self.norm), self.total_error
        else:
            return self.total_error, self.total_error

    def evaluate_operation(self):
        # get tuples of all the combinations of refinement to access each subarea (this is the same for each component grid)
        areas = self.get_new_areas()
        evaluation_array = np.zeros(len(areas))
        for area in areas:
            self.operation.area_preprocessing(area)
        self.compute_solutions(areas,evaluation_array)
        self.finalize_evaluation()
        for area in areas:
            self.operation.area_postprocessing(area)
        for k in range(len(areas)):
            i = k + self.refinement.size() - self.refinement.new_objects_size()
            self.refinement.set_evaluations(i, evaluation_array[k])
        for k in range(len(areas)):
            i = k + self.refinement.size() - self.refinement.new_objects_size()
            self.calc_error(i, self.f)
            self.refinement.set_benefit(i)

        # getArea with maximal error
        self.benefit_max = self.refinement.get_max_benefit()
        self.total_error = self.refinement.get_total_error()
        if self.print_output:
            print("max surplus error:", self.benefit_max, "total surplus error:", self.total_error)
            self.operation.print_evaluation_output(self.refinement)
        global_error_estimate = self.operation.get_global_error_estimate(self.refinement, self.norm)
        if global_error_estimate is not None:
            return global_error_estimate, self.total_error
        else:
            return self.total_error, self.total_error

    def compute_solutions(self, areas, evaluation_array):
        # calculate integrals
        for component_grid in self.scheme:  # iterate over component grids
            if self.operation.is_area_operation():
                for k, area in enumerate(areas):
                    evaluations = self.evaluate_operation_area(component_grid, area)
                    if self.grid.isNested() and self.operation.count_unique_points():
                        evaluations *= component_grid.coefficient
                    evaluation_array[k] += evaluations
            else:
                assert(False) # not implemented yet
                points = self.get_points_component_grid(component_grid.levelvector, num_sub_diagonal)
                self.operation.perform_operation(points)
                self.compute_evaluations(evaluation_array, points)

    def evaluate_operation_area(self, component_grid, area, additional_info=None):
        num_sub_diagonal = (self.lmax[0] + self.dim - 1) - np.sum(component_grid.levelvector)
        modified_levelvec, do_compute = self.coarsen_grid(component_grid.levelvector, area, num_sub_diagonal)
        if do_compute:
            evaluations = self.operation.evaluate_area(area, modified_levelvec, component_grid, self.refinement, additional_info)
            return evaluations
        else:
            return 0

    def refine(self):
        # split all cells that have an error close to the max error
        self.prepare_refinement()
        self.refinement.clear_new_objects()
        margin = self.margin
        quit_refinement = False
        num_refinements = 0
        while True:  # refine all areas for which area is within margin
            # get next area that should be refined
            found_object, position, refine_object = self.refinement.get_next_object_for_refinement(
                tolerance=self.benefit_max * margin)
            if found_object and not quit_refinement:  # new area found for refinement
                self.refinements += 1
                num_refinements += 1
                # print("Refining position", position)
                quit_refinement = self.do_refinement(refine_object, position)

            else:  # all refinements done for this iteration -> reevaluate integral and check if further refinements necessary
                if self.print_output:
                    print("Finished refinement")
                    print("Refined ", num_refinements, " times")
                self.refinement_postprocessing()
                break

        if self.recalculate_frequently and self.refinements / self.refinements_for_recalculate > self.counter:
            self.refinement.reinit_new_objects()
            self.combiintegral = 0
            self.subAreaIntegrals = []
            self.evaluationPerArea = []
            self.evaluationsTotal = 0
            self.counter += 1
            if self.print_output:
                print("recalculating errors")

    # optimized adaptive refinement refine multiple cells in close range around max variance (here set to 10%)
    def performSpatiallyAdaptiv(self, minv=1, maxv=2, f=FunctionGriebel(), errorOperator=None, tol=10 ** -2,
                                refinement_container=[], do_plot=False, recalculate_frequently=False, test_scheme=False,
                                reevaluate_at_end=False, reference_solution=None, max_time=None, max_evaluations=None, print_output=True):
        self.errorEstimator = errorOperator
        self.recalculate_frequently = recalculate_frequently
        self.realIntegral = reference_solution
        self.print_output = print_output
        self.init_adaptive_combi(f, minv, maxv, refinement_container, tol)
        self.error_array = []
        self.surplus_error_array = []
        self.num_point_array = []
        self.test_scheme = test_scheme
        self.reevaluate_at_end = reevaluate_at_end
        self.do_plot = do_plot
        self.reference_solution = reference_solution
        return self.continue_adaptive_refinement(tol=tol, max_time=max_time, max_evaluations=max_evaluations)


    def continue_adaptive_refinement(self, tol=10 ** -3, max_time=None, max_evaluations=None):
        start_time = time.time()
        while True:
            error, surplus_error = self.evaluate_integral()
            self.error_array.append(error)
            if self.reference_solution is not None:
                self.surplus_error_array.append(surplus_error/abs(self.reference_solution))
            else:
                self.surplus_error_array.append(surplus_error)
            self.num_point_array.append(self.get_total_num_points(distinct_function_evals=True))
            if self.print_output:
                print("Current error:", error)
            # check if tolerance is already fullfilled with current refinement
            if error > tol:
                if max_evaluations is not None:
                    if self.get_total_num_points() > max_evaluations:
                        break
                if max_time is not None:
                    current_time = time.time()
                    if current_time - start_time > max_time:
                        break
                # refine further
                self.refine()
                if self.do_plot:
                    print("Refinement Graph:")
                    self.draw_refinement()
                    print("Combi Scheme:")
                    self.print_resulting_combi_scheme(markersize=5)
                    print("Resulting Sparse Grid:")
                    self.print_resulting_sparsegrid(markersize=3)
            else:  # refinement finished
                break
        # finished adaptive algorithm
        if self.print_output:
            print("Number of refinements", self.refinements)
            print("Number of distinct points used during the refinement", self.get_total_num_points())
            print("Time used (s):", time.time() - start_time)
            print("Final error:", error)
        if self.test_scheme:
            self.check_combi_scheme()
        if self.reevaluate_at_end:
            # evaluate final integral
            combiintegral, number_of_evaluations = self.evaluate_final_combi()
        else:
            combiintegral = self.refinement.integral
            number_of_evaluations = self.refinement.evaluationstotal
        return self.refinement, self.scheme, self.lmax, combiintegral, number_of_evaluations, self.error_array, self.num_point_array, self.surplus_error_array


    @abc.abstractmethod
    def initialize_refinement(self):
        pass

    @abc.abstractmethod
    def get_points_component_grid(self, levelvec, numSubDiagonal):
        return

    @abc.abstractmethod
    def evaluate_area(self, f, area, component_grid):
        pass

    @abc.abstractmethod
    def do_refinement(self, area, position):
        pass

    # this is a default implementation that should be overritten if necessary
    def prepare_refinement(self):
        pass

    # this is a default implementation that should be overritten if necessary
    def refinement_postprocessing(self):
        self.refinement.apply_remove()
        self.refinement.refinement_postprocessing()

    # this is a default implementation that should be overritten if necessary
    def calc_error(self, objectID, f):
        self.refinement.calc_error(objectID, f, self.norm)

    # this is a default implementation that should be overritten if necessary
    def get_new_areas(self):
        return self.refinement.get_new_objects()

    # this is a default implementation that should be overritten if necessary
    def get_areas(self):
        return self.refinement.get_objects()

    # this method can be overwritten if for the method a graphical refinement visualization exists
    def draw_refinement(self, filename=None):
        pass

    # this method modifies the level if necessary and indicates if the area should be computed (second boolean return value)
    def coarsen_grid(self, levelvector, area, num_sub_diagonal):
        return levelvector, True

    def finalize_evaluation(self):
        pass
