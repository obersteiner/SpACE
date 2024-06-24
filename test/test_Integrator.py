import unittest
import sparseSpACE
from sparseSpACE.StandardCombi import *
import math
from sparseSpACE.Grid import *
from sparseSpACE.Integrator import *
from sparseSpACE.Function import *


class TestIntegrator(unittest.TestCase):

    def test_integrate_non_basis_functions(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = GlobalTrapezoidalGrid(a*np.ones(d), b*np.ones(d), boundary= True, modified_basis = False)
            for integrator in [IntegratorArbitraryGrid(grid), IntegratorArbitraryGridScalarProduct(grid)]:
                grid.integrator = integrator
                for l in range(7 - d):
                    f = FunctionLinear([10*(i+1) for i in range(d)])
                    grid_points = [np.linspace(a,b,2**(l+i)+ 1) for i in range(d)]
                    grid_levels = [np.zeros(2**(l+i) + 1, dtype=int) for i in range(d)]
                    for i in range(d):
                        for l2 in range(1,l+i+1):
                            offset = 2**(l+i - l2)
                            for j in range(offset, len(grid_levels[i]), 2*offset):
                                grid_levels[i][j] = l2
                    grid.set_grid(grid_points, grid_levels)
                    integral = grid.integrate(f, [l + i for i in range(d)], a * np.ones(d), b * np.ones(d))
                    #assert(False)
                    #print(integral, f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d)), f.eval(np.ones(d)))
                    self.assertAlmostEqual((integral[0] - f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))) / abs(f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))), 0.0, places=13)

    def test_integrate_basis_functions(self):
        a = -3
        b = 6
        for d in range(1, 5):
            for p in range(1,7):
                grid = GlobalHierarchicalLagrangeGrid(a*np.ones(d), b*np.ones(d), boundary= True, modified_basis = False, p=p)
                for l in range(p - 1, 8 - d):
                    f = FunctionPolynomial([10*(i+1) for i in range(d)], degree=p)
                    grid_points = [np.linspace(a,b,2**l+ 1) for _ in range(d)]
                    grid_levels = [np.zeros(2**l + 1, dtype=int) for _ in range(d)]
                    for i in range(d):
                        for l2 in range(1,l+1):
                            offset = 2**(l - l2)
                            for j in range(offset, len(grid_levels[i]), 2*offset):
                                grid_levels[i][j] = l2
                    grid.set_grid(grid_points, grid_levels)
                    integral = grid.integrate(f, [l for i in range(d)], a * np.ones(d), b * np.ones(d))
                    #print(integral, f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d)), (integral[0] - f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))) / abs(f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))), p)
                    self.assertAlmostEqual((integral[0] - f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))) / abs(f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))), 0.0, places=12)

            for p in range(1,  7, 2):
                grid = GlobalBSplineGrid(a*np.ones(d), b*np.ones(d), boundary= True, modified_basis = False, p=p)
                for l in range(int(log2(p)) + 1, 8 - d):
                    f = FunctionPolynomial([10*(i+1) for i in range(d)], degree=p)
                    grid_points = [list(np.linspace(a,b,2**l + 1)) for _ in range(d)]
                    grid_levels = [np.zeros(2**l + 1, dtype=int) for _ in range(d)]
                    for i in range(d):
                        for l2 in range(1,l+1):
                            offset = 2**(l - l2)
                            for j in range(offset, len(grid_levels[i]), 2*offset):
                                grid_levels[i][j] = l2
                    grid.set_grid(grid_points, grid_levels)
                    integral = grid.integrate(f, [l for i in range(d)], a * np.ones(d), b * np.ones(d))
                    #print(integral, f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d)), p)
                    #print(integral, f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d)), (integral[0] - f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))) / abs(f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))), p)
                    # Here exactness is not guaranteed but it should be close
                    self.assertAlmostEqual((integral[0] - f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))) / abs(f.getAnalyticSolutionIntegral(a*np.ones(d), b*np.ones(d))), 0.0, places=11)

if __name__ == '__main__':
    unittest.main()