import unittest
import sparseSpACE
from sparseSpACE.StandardCombi import *
import math
from sparseSpACE.Hierarchization import *
from sparseSpACE.Grid import *
from sparseSpACE.Function import *


class TestHierarchization(unittest.TestCase):


    def test_interpolation(self):
        a = -3
        b = 6
        for d in range(2, 5):
            grid = GlobalHierarchicalLagrangeGrid(a*np.ones(d), b*np.ones(d), boundary= True, modified_basis = False, p = 1)
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
                grid.integrate(f, [l + i for i in range(d)], a * np.ones(d), b * np.ones(d))
                component_grid = ComponentGridInfo([l+i for i in range(d)], 1)
                grid_points = get_cross_product_list(grid_points)
                f_values = grid.interpolate(grid_points, component_grid)
                for i, p in enumerate(grid_points):
                    factor = abs(f(p)[0] if f(p)[0] != 0 else 1)
                    self.assertAlmostEqual((f(p)[0] - f_values[i][0]) / factor, 0, 11)

            grid = GlobalBSplineGrid(a * np.ones(d), b * np.ones(d), boundary=True, modified_basis=False, p=1)
            for l in range(7 - d):
                f = FunctionLinear([10 * (i + 1) for i in range(d)])
                grid_points = [list(np.linspace(a, b, 2 ** (l + i) + 1)) for i in range(d)]
                grid_levels = [np.zeros(2 ** (l + i) + 1, dtype=int) for i in range(d)]
                for i in range(d):
                    for l2 in range(1, l + i + 1):
                        offset = 2 ** (l + i - l2)
                        for j in range(offset, len(grid_levels[i]), 2 * offset):
                            grid_levels[i][j] = l2
                grid.set_grid(grid_points, grid_levels)
                grid.integrate(f, [l + i for i in range(d)], a * np.ones(d), b * np.ones(d))
                component_grid = ComponentGridInfo([l + i for i in range(d)], 1)
                grid_points = get_cross_product_list(grid_points)
                f_values = grid.interpolate(grid_points, component_grid)
                for i, p in enumerate(grid_points):
                    factor = abs(f(p)[0] if f(p)[0] != 0 else 1)
                    self.assertAlmostEqual((f(p)[0] - f_values[i][0]) / factor, 0, 11)

if __name__ == '__main__':
    unittest.main()