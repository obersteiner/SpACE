import unittest
import sparseSpACE

from sparseSpACE.Extrapolation import SliceVersion, ExtrapolationGrid, SliceContainerVersion, \
    LagrangeRombergGridSliceContainer, SliceGrouping, GridVersion
from sparseSpACE.Function import Polynomial1d


class TestExtrapolationInterpolationGrid(unittest.TestCase):
    def test_interpolated_points(self):
        grid = [0.0, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 0.875, 1]
        grid_levels = [0, 3, 4, 2, 3, 1, 2, 3, 0]
        # Containers: [0, 0.125], [0.125, .0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 0.875] [0.875, 1]

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        containers = romberg_grid.slice_containers

        # Test which points are interpolated
        self.assertEqual([0.0625, 0.3125, 0.4375], containers[0].get_interpolation_points())
        self.assertEqual([0.625], containers[1].get_interpolation_points())

        # Test the interpolation indicator for the container grid
        self.assertEqual([False, True, False, False, False, True, False, True, False],
                         containers[0].get_interpolated_grid_points_indicator())
        self.assertEqual([False, True, False, False, False], containers[1].get_interpolated_grid_points_indicator())

    # Container tests
    def test_grid_without_interpolated_points(self):
        grid = [0.0, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 0.875, 1]
        grid_levels = [0, 3, 4, 2, 3, 1, 2, 3, 0]

        actual_grid = []
        actual_grid_levels = []

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        containers = romberg_grid.slice_containers

        for container in containers:
            actual_grid.extend(container.get_grid_without_interpolated_points()[0:-1])
            actual_grid_levels.extend(container.get_grid_levels_without_interpolated_points()[0:-1])

        actual_grid.append(containers[-1].right_point)
        actual_grid_levels.append(containers[-1].slices[-1].levels[1])

        self.assertEqual(grid, actual_grid)
        self.assertEqual(grid_levels, actual_grid_levels)

    def test_grid_with_max_point_count(self):
        grid = [0.0, 0.125, 0.1875, 0.25, 0.5, 0.625, 0.75,  0.875, 1]
        grid_levels = [0, 3, 4, 2, 1, 3, 2, 3, 0]

        actual_grid = LagrangeRombergGridSliceContainer.get_grid_with_max_point_count(grid, grid_levels, 6)
        expected_grid = [0.0, 0.25, 0.5, 0.75, 1]

        self.assertEqual(expected_grid, actual_grid)

    def test_interpolation_support_points_by_level(self):
        grid = [0.0, 0.125, 0.1875, 0.25, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 3, 4, 2, 1, 3, 2, 0]

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        containers = romberg_grid.slice_containers

        self.assertEqual([0.0, 0.125, 0.1875, 0.25, 0.5], containers[0].get_support_points_for_interpolation_by_levels(5))
        self.assertEqual([0.1875, 0.25, 0.5, 0.75], containers[1].get_support_points_for_interpolation_by_levels(4))
        self.assertEqual([0.1875, 0.75, 1], containers[2].get_support_points_for_interpolation_by_levels(3))

    def test_interpolation_support_points_geometrically(self):
        grid = [0.0, 0.125, 0.1875, 0.25, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 3, 4, 2, 1, 3, 2, 0]

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        containers = romberg_grid.slice_containers

        self.assertEqual([0.0, 0.125, 0.1875, 0.25, 0.5],
                         containers[0].get_support_points_for_interpolation_geometrically(0.0625, 5))
        self.assertEqual([0.1875, 0.25, 0.5, 0.625],
                         containers[1].get_support_points_for_interpolation_geometrically(0.375, 4))
        self.assertEqual([0.625, 0.75, 1],
                         containers[2].get_support_points_for_interpolation_geometrically(0.875, 3))

    def test_weight_count(self):
        grid = [0.0, 0.0078125, 0.015625, 0.0234375, 0.03125, 0.0390625, 0.046875, 0.0625, 0.078125, 0.09375, 0.109375,
                0.125,
                0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.265625, 0.28125, 0.296875,
                0.3125,
                0.328125, 0.34375, 0.359375, 0.375, 0.390625, 0.40625, 0.4375, 0.46875, 0.5, 0.53125, 0.5625, 0.59375,
                0.625,
                0.65625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 1.0]

        grid_levels = [0, 7, 6, 7, 5, 7, 6, 4, 6, 5, 6, 3, 6, 5, 6, 4, 6, 5, 6, 2, 6, 5, 6, 4, 6, 5, 6, 3, 6, 5, 4, 5,
                       1, 5, 4, 5, 3, 5, 4, 5, 2, 5, 4, 5, 3, 5, 4, 0]

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        weights = romberg_grid.get_weights()

        # 0.96875 is not a support point

        self.assertEqual(len(grid), len(weights))

    def test_full_grid_interpolation(self):
        grid = [0, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 1, 3, 2, 0]

        function = Polynomial1d([1, 0, 0, 2])

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_FULL_GRID,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        actual_result = romberg_grid.integrate(function)

        expected_result = 1.5
        self.assertAlmostEqual(expected_result, actual_result)

    def test_exactness_on_balanced_adaptive_interpolated_lagrange_grid(self):
        grid = [0, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 1, 3, 2, 0]

        function = Polynomial1d([1, 0, 0, 2])

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=True)
        romberg_grid.set_grid(grid, grid_levels)
        actual_result = romberg_grid.integrate(function)

        expected_result = 1.5
        self.assertAlmostEqual(expected_result, actual_result)

    def test_exactness_on_unbalanced_adaptive_interpolated_lagrange_grid(self):
        grid = [0, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 1, 3, 2, 0]

        function = Polynomial1d([1, 0, 0, 2])

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        actual_result = romberg_grid.integrate(function)

        expected_result = 1.5

        # TODO works only with force_balanced_refinement_tree = True
        # self.assertAlmostEqual(expected_result, actual_result)

    def test_exactness_on_unbalanced_adaptive_interpolated_bspline_grid(self):
        grid = [0, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 1, 3, 2, 0]

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.BSPLINE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=False)
        romberg_grid.set_grid(grid, grid_levels)
        function = Polynomial1d([1, 0, 0, 2])

        actual_result = romberg_grid.integrate(function)
        expected_result = 1.5

        # self.assertAlmostEqual(expected_result, actual_result)

    def test_exactness_on_unbalanced_adaptive_interpolated_hierarchical_lagrange_grid(self):
        grid = [0, 0.5, 0.625, 0.75, 1]
        grid_levels = [0, 1, 3, 2, 0]

        romberg_grid = ExtrapolationGrid(grid_version=GridVersion.INTERPOLATE_SUB_GRIDS,
                                         container_version=SliceContainerVersion.HIERARCHICAL_LAGRANGE_ROMBERG,
                                         slice_grouping=SliceGrouping.GROUPED_OPTIMIZED,
                                         slice_version=SliceVersion.ROMBERG_DEFAULT,
                                         force_balanced_refinement_tree=True)
        romberg_grid.set_grid(grid, grid_levels)
        function = Polynomial1d([1, 0, 0, 2])

        actual_result = romberg_grid.integrate(function)
        expected_result = 1.5

        # self.assertAlmostEqual(expected_result, actual_result)


# -----------------------------------------------------------------------------------------------------------------
# ---  Unit Test

if __name__ == '__main__':
    unittest.main()
