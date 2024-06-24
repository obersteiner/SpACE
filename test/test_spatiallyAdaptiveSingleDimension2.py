import unittest
import sparseSpACE
from sparseSpACE.StandardCombi import *
import math
from sparseSpACE.Grid import *
from sparseSpACE.Integrator import *
from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *


class TestSpatiallyAdaptiveSingleDimension2(unittest.TestCase):

    def test_integrate(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = GlobalTrapezoidalGrid(a * np.ones(d), b * np.ones(d), boundary=True, modified_basis=False)
            f = FunctionLinear([10 * (i + 1) for i in range(d)])
            operation = Integration(f, grid=grid, dim=d, reference_solution=f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
            errorOperator = ErrorCalculatorSingleDimVolumeGuided()
            for l in range(2, 4):
                for num_points in np.linspace(100, 1000, 5):
                    spatiallyAdaptive = SpatiallyAdaptiveSingleDimensions2(a * np.ones(d), b * np.ones(d), version=3,
                                                                           operation=operation)
                    _, _, _, combiintegral, _, error_array, _, surplus_error_array, _, _ = spatiallyAdaptive.performSpatiallyAdaptiv(
                        lmin=1, lmax=l, errorOperator=errorOperator, tol=-1, max_evaluations=num_points,
                        print_output=False)
                    self.assertEqual(combiintegral, f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
                    self.assertTrue(all([error == 0.0 for error in error_array]))

        a = -3
        b = 6
        for d in range(2, 4):
            grid = GlobalHierarchicalLagrangeGrid(a * np.ones(d), b * np.ones(d), boundary=True, modified_basis=False, p=2)
            f = FunctionPolynomial([10 * (i + 1) for i in range(d)], degree=2)
            operation = Integration(f, grid=grid, dim=d, reference_solution=f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
            errorOperator = ErrorCalculatorSingleDimVolumeGuided()
            for l in range(2, 3):
                for num_points in np.linspace(100, 1000, 5):
                    spatiallyAdaptive = SpatiallyAdaptiveSingleDimensions2(a * np.ones(d), b * np.ones(d), version=3,
                                                                           operation=operation)
                    _, _, _, combiintegral, _, error_array, _, surplus_error_array, _, _ = spatiallyAdaptive.performSpatiallyAdaptiv(
                        lmin=1, lmax=l, errorOperator=errorOperator, tol=-1, max_evaluations=num_points,
                        print_output=False)
                    self.assertAlmostEqual(combiintegral[0] / f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)), 1.0, places=12)
                    self.assertTrue(all([ -10**-13 * combiintegral[0] <= error <= 10**-13 * combiintegral[0] for error in error_array]))


        a = 2
        b = 6
        for d in range(2, 4):
            grid = GlobalTrapezoidalGrid(a * np.ones(d), b * np.ones(d), boundary=True, modified_basis=False)
            f = FunctionLinear([10 * (i + 1) for i in range(d)])
            operation = Integration(f, grid=grid, dim=d, reference_solution=f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
            errorOperator = ErrorCalculatorSingleDimVolumeGuided()
            for l in range(2, 3):
                for num_points in np.linspace(100, 1000, 5):
                    spatiallyAdaptive = SpatiallyAdaptiveSingleDimensions2(a * np.ones(d), b * np.ones(d), version=3,
                                                                           operation=operation)
                    _, _, _, combiintegral, _, error_array, _, surplus_error_array, _, _ = spatiallyAdaptive.performSpatiallyAdaptiv(
                        lmin=1, lmax=l, errorOperator=errorOperator, tol=-1, max_evaluations=num_points,
                        print_output=False)
                    self.assertEqual(combiintegral, f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
                    self.assertTrue(all([error == 0.0 for error in error_array]))

        a = -6
        b = -3
        for d in range(2, 4):
            grid = GlobalTrapezoidalGrid(a * np.ones(d), b * np.ones(d), boundary=True, modified_basis=False)
            f = FunctionLinear([10 * (i + 1) for i in range(d)])
            operation = Integration(f, grid=grid, dim=d, reference_solution=f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
            errorOperator = ErrorCalculatorSingleDimVolumeGuided()
            for l in range(2, 3):
                for num_points in np.linspace(100, 1000, 5):
                    spatiallyAdaptive = SpatiallyAdaptiveSingleDimensions2(a * np.ones(d), b * np.ones(d), version=3,
                                                                           operation=operation)
                    _, _, _, combiintegral, _, error_array, _, surplus_error_array, _, _ = spatiallyAdaptive.performSpatiallyAdaptiv(
                        lmin=1, lmax=l, errorOperator=errorOperator, tol=-1, max_evaluations=num_points,
                        print_output=False)
                    self.assertEqual(combiintegral, f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
                    self.assertTrue(all([error == 0.0 for error in error_array]))

    def test_interpolate(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = GlobalTrapezoidalGrid(a * np.ones(d), b * np.ones(d), boundary=True, modified_basis=False)
            f = FunctionLinear([10 * (i + 1) for i in range(d)])
            operation = Integration(f, grid=grid, dim=d, reference_solution=f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
            errorOperator = ErrorCalculatorSingleDimVolumeGuided()
            for l in range(2, 4):
                for num_points in np.linspace(100, 1000, 5):
                    spatiallyAdaptive = SpatiallyAdaptiveSingleDimensions2(a * np.ones(d), b * np.ones(d), version=3,
                                                                           operation=operation)
                    _, _, _, combiintegral, _, error_array, _, surplus_error_array, _, _ = spatiallyAdaptive.performSpatiallyAdaptiv(
                        lmin=1, lmax=l, errorOperator=errorOperator, tol=-1, max_evaluations=num_points,
                        print_output=False)
                    points = get_cross_product_list([np.linspace(a, b, 5, endpoint=False) for _ in range(d)])
                    f_values = spatiallyAdaptive(points)
                    for i, value in enumerate(f_values):
                        factor = abs(f(points[i])[0]) if abs(f(points[i])[0]) != 0 else 1
                        self.assertAlmostEqual((value[0] - f(points[i])[0]) / factor, 0.0, places=12)

        a = -1
        b = 6
        for d in range(2, 5):
            grid = GlobalHierarchicalLagrangeGrid(a * np.ones(d), b * np.ones(d), boundary=True, modified_basis=False, p=2)
            f = FunctionPolynomial([(i + 1) for i in range(d)], degree=2)
            operation = Integration(f, grid=grid, dim=d, reference_solution=f.getAnalyticSolutionIntegral(a * np.ones(d), b * np.ones(d)))
            errorOperator = ErrorCalculatorSingleDimVolumeGuided()
            for l in range(2, 4):
                for num_points in np.linspace(100, 1000, 5):
                    spatiallyAdaptive = SpatiallyAdaptiveSingleDimensions2(a * np.ones(d), b * np.ones(d), version=3,
                                                                           operation=operation)
                    _, _, _, combiintegral, _, error_array, _, surplus_error_array, _, _ = spatiallyAdaptive.performSpatiallyAdaptiv(
                        lmin=1, lmax=l, errorOperator=errorOperator, tol=-1, max_evaluations=num_points,
                        print_output=False)
                    points = get_cross_product_list([np.linspace(a, b, 3, endpoint=False) for _ in range(d)])
                    f_values = spatiallyAdaptive(points)
                    for i, value in enumerate(f_values):
                        factor = abs(f(points[i])[0]) if abs(f(points[i])[0]) != 0 else 1
                        self.assertAlmostEqual((value[0] - f(points[i])[0]) / factor, 0.0, places=10)

if __name__ == '__main__':
    unittest.main()
