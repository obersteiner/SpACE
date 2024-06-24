from sparseSpACE.spatiallyAdaptiveBase import *
from sparseSpACE.GridOperation import *
from collections import Counter
from sparseSpACE.Extrapolation import GridBinaryTree

import time
from math import copysign

def sortToRefinePosition(elem):
    # sort by depth
    return elem[1]


class NodeInfo(object):
    def __init__(self, child, left_parent, right_parent, left_parent_of_left_parent, right_parent_of_right_parent, has_left_child, has_right_child, left_refinement_object, right_refinement_object, level_child):
        self.child = child
        self.left_parent = left_parent
        self.right_parent = right_parent
        self.left_parent_of_left_parent = left_parent_of_left_parent
        self.right_parent_of_right_parent = right_parent_of_right_parent
        self.has_left_child = has_left_child
        self.has_right_child = has_right_child
        self.left_refinement_object = left_refinement_object
        self.right_refinement_object = right_refinement_object
        self.level_child = level_child


class SpatiallyAdaptiveSingleDimensions2(SpatiallyAdaptivBase):
    def __init__(self, a: Sequence[float], b: Sequence[float], norm: int = np.inf, dim_adaptive: bool = True,
                 version: int = 6, operation: GridOperation = None, margin: float = None, timings=None,
                 rebalancing: bool = True, rebalancing_safety_factor: float = 0.1, chebyshev_points=False,
                 use_volume_weighting=False, force_balanced_refinement_tree: bool = False, grid_surplusses=None,
                 log_level: int = log_levels.INFO, print_level: int = print_levels.INFO, use_relative_surplus: bool=False):
        SpatiallyAdaptivBase.__init__(self, a, b, operation=operation, norm=norm, log_level=log_level, print_level=print_level)
        assert self.grid is not None

        if grid_surplusses is None:
            self.grid_surplusses = GlobalTrapezoidalGrid(a, b, boundary=self.grid.boundary, modified_basis=self.grid.modified_basis)
        else:
            self.grid_surplusses = self.grid

        self.dim_adaptive = dim_adaptive
        #self.evaluationCounts = None
        self.version = version
        #self.dict_integral = {}
        #self.dict_points = {}
        self.no_previous_integrals = True
        self.use_local_children = True #self.version == 2 or self.version == 3
        if margin is None:
            self.margin = 0.9#1-10**-12 if self.use_local_children else 0.9
        else:
            self.margin = margin
        self.operation = operation
        self.equidistant = False #True
        self.rebalancing = rebalancing
        self.rebalancing_safety_factor = rebalancing_safety_factor
        assert self.rebalancing_safety_factor >= 0
        self.subtraction_value_cache = {}
        self.max_level_dict = {}
        self.chebyshev_points = chebyshev_points
        self.use_volume_weighting = use_volume_weighting
        self.timings = timings
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        self.log_util.set_print_prefix('SpatiallyAdaptiveSingleDimensions2')
        self.log_util.set_log_prefix('SpatiallyAdaptiveSingleDimensions2')
        # If set to true, all grid points have either 0 or two children
        self.force_balanced_refinement_tree = force_balanced_refinement_tree
        self.grid_binary_tree = GridBinaryTree()
        self.use_relative_surplus = use_relative_surplus


    def interpolate_points(self, interpolation_points: Sequence[Tuple[float, ...]], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        # check if dedicated interpolation routine is present in grid
        interpolation_op = getattr(self.grid, "interpolate", None)
        if callable(interpolation_op):
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
            self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            return self.grid.interpolate(interpolation_points, component_grid)
        else:
            # call default d-linear interpolation based on points in grid
            # Attention: This only works if we interpolate in between the grid points -> extrapolation not supported
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
            return self.operation.interpolate_points_component_grid(component_grid, gridPointCoordsAsStripes, interpolation_points)

    def interpolate_grid_component(self, grid_coordinates: Sequence[Sequence[float]], component_grid: ComponentGridInfo) -> Sequence[Sequence[float]]:
        # check if dedicated interpolation routine is present in grid
        interpolation_op = getattr(self.grid, "interpolate_grid", None)
        if callable(interpolation_op):
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)
            self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
            return self.grid.interpolate_grid(grid_coordinates, component_grid)
        else:
            # call default d-linear interpolation based on points in grid
            # Attention: This only works if we interpolate in between the grid points -> extrapolation not supported
            return super().interpolate_grid_component(grid_coordinates, component_grid)

    def coarsen_grid(self, area, levelvec: Sequence[int]):
        pass

    # returns the points coordinates of a single component grid with refinement
    def get_points_all_dim(self, levelvec: Sequence[int]) -> List[Tuple[float, ...]]:
        indicesList, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(levelvec)
        if not self.grid.boundary:
            indicesList = [indices[1:-1] for indices in indicesList]
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        allPoints = list(set(zip(*[g.ravel() for g in np.meshgrid(*indicesList)])))
        return allPoints

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec: Sequence[int]) -> List[Tuple[float, ...]]:
        return self.get_points_all_dim(levelvec)

    def get_points_and_weights_component_grid(self, levelvec: Sequence[int]) -> Tuple[Sequence[Tuple[float, ...]], Sequence[float]]:
        point_coords, point_levels, _ =self.get_point_coord_for_each_dim(levelvec)
        self.grid.set_grid(point_coords, point_levels)
        points, weights = self.grid.get_points_and_weights()
        return points, weights

    def get_num_points_each_dim(self) -> Sequence[int]:
        """This method returns the number of points in each dimension including boundary points.

        :return: numpy array with number of points
        """
        num_points = np.zeros(self.dim, dtype=int)

        for d in range(self.dim):
            num_points[d] = self.refinement.get_refinement_container_for_dim(d).size()
            assert(num_points[d] > 0)

        return num_points

    def get_point_coord_for_each_dim(self, levelvec: Sequence[int]) -> Tuple[Sequence[Sequence[float]], Sequence[Sequence[int]], Sequence[Sequence[NodeInfo]]]:
        """This method returns the 1D list of point coordinates and the point levels for each dimension.
        In addition the 1D list of children for each dimension is returned.

        If self.use_local_children is set the children will be the ones in the local compoenent grid, otherwise we will
        return only grids that are also children in the global refinement structure (which might not exist).

        :param levelvec: Level vector of the component grid
        :return: List of point coordinates and levels and children for each dimension.
        """
        refinement = self.refinement
        # get a list of all coordinates for every this_dim (so (0, 1), (0, 0.5, 1) for example)
        point_coordinates = []
        children_indices = []
        points_level = []
        max_coarsenings = np.zeros(self.dim, dtype=int) #only for dimensions with level > 1

        for d in range(self.dim):
            #if levelvec[d] > self.lmin[d]:
            max_coarsenings[d] = refinement.get_max_coarsening(d)

        for d in range(0, self.dim):
            max_coarsenings_dim = list(max_coarsenings)
            refineContainer = refinement.get_refinement_container_for_dim(d)
            refine_container_objects = refineContainer.get_objects()
            points_dim = []
            points_level_dim = []

            children_indices_dim = []
            points_dim.append(refine_container_objects[0].start)
            points_level_dim.append(refine_container_objects[0].levels[0])
            for i in range(len(refine_container_objects)):
                refineObj = refine_container_objects[i]
                if i + 1 < len(refine_container_objects):
                    next_refineObj = refine_container_objects[i + 1]
                else:
                    next_refineObj = None

                subtraction_value = self.get_subtraction_value(refineObj, refineContainer, i, max_coarsenings, d, levelvec)

                if (refineObj.levels[1] <= max(levelvec[d] - subtraction_value, 1)):
                    points_dim.append(refineObj.end)
                    if not self.use_local_children:
                        if next_refineObj is not None and self.is_child(refineObj.levels[0], refineObj.levels[1], next_refineObj.levels[0]):
                            children_indices_dim.append(self.get_node_info(refineObj.end, refineObj.levels[1], refineObj.start, refineObj.levels[0], next_refineObj.end, next_refineObj.levels[1], d))
                    else:
                        points_level_dim.append(refineObj.levels[1])

            # Force full binary tree (each non-boundary grid point should have either 0 or 2 children)
            if self.force_balanced_refinement_tree:
                points_dim, points_level_dim = self.transform_to_full_binary_tree_grid(points_dim, points_level_dim, refine_container_objects)

            # Compute children indices
            if self.use_local_children:
                for i in range(1,len(points_level_dim)-1):
                    if self.is_child(points_level_dim[i-1], points_level_dim[i], points_level_dim[i+1]):
                        children_indices_dim.append((self.get_node_info(i, points_dim, points_level_dim, d)))

            point_coordinates.append(points_dim)
            children_indices.append(children_indices_dim)
            points_level.append(points_level_dim)

            # Test if children_indices is valid
            for c in children_indices_dim:
                points_dim.index(c.left_parent)
                points_dim.index(c.right_parent)
            # Test if indices are valid
            assert all(points_dim[i] <= points_dim[i + 1] for i in range(len(points_dim) - 1))

        return point_coordinates, points_level, children_indices

    def transform_to_full_binary_tree_grid(self, grid, grid_levels, refinement_objects):
        """ This method expands a grid so that it becomes a full binary tree.
        This means each grid point has either 0 or 2 children.

        :param grid: Contains grid points as array
        :param grid_levels: Contains levels of the grid as array
        :return: Returns grid and grid levels that have a full binary tree structure
        """
        full_grid = [refinement_object.start for refinement_object in refinement_objects] + [refinement_objects[-1].end]
        full_grid_levels = [refinement_object.levels[0] for refinement_object in refinement_objects] + [refinement_objects[-1].levels[1]]
        missing_points = []
        matched_points = set()
        for i, level in enumerate(grid_levels):
            if i not in matched_points:
                for j in range(i+1, len(grid_levels)):
                    if grid_levels[j] == level:
                        matched_points.add(i)
                        matched_points.add(j)
                        break
                    if grid_levels[j] <= level - 2:
                        missing_point = self.find_missing_point(grid[i], grid_levels[i], full_grid, full_grid_levels)
                        missing_points.append((missing_point, level))
                        break
        previous_points = list(zip(grid,grid_levels))
        balanced_points = previous_points + missing_points
        new_grid = [point[0] for point in sorted(balanced_points)]
        new_levels = [point[1] for point in sorted(balanced_points)]
        return new_grid, new_levels

        '''
        is_full_binary_tree = True
        level_counter = Counter(grid_levels)  # Counts occurences of each level in the list
        critical_levels = []
        
        for (level, count) in level_counter.items():
            if (level >= 2) and (count % 2 == 1):
                is_full_binary_tree = False
                critical_levels.append(level)
        
        if is_full_binary_tree:
            return grid, grid_levels
        
        # print("This grid has no full binary tree structure. Critical levels are {}.".format(critical_levels))
        # print("    Grid: {}".format(grid))
        # print("    Levels: {}".format(grid_levels))
        
        # Build up full binary tree structure
        self.grid_binary_tree.init_tree(grid, grid_levels)
        self.grid_binary_tree.force_full_tree_invariant()
        
        return self.grid_binary_tree.get_grid(), self.grid_binary_tree.get_grid_levels()
        '''

    def find_missing_point(self, point, level, full_grid, full_levels):
        #print(point, level, full_grid, full_levels)
        for position, grid_point in enumerate(full_grid):
            if point == grid_point:
                assert level == full_levels[position]
                for j, point_candidate in zip(range(position+1, len(full_grid)), full_grid[position+1:]):
                    if full_levels[j] <= level -2:
                        break
                    if full_levels[j] == level:
                        return point_candidate
                for j, point_candidate in zip(range(position-1,-1,-1),reversed(full_grid[:position])):
                    if full_levels[j] <= level -2:
                        assert False # should never happen
                    if full_levels[j] == level:
                        return point_candidate

    def modify_according_to_levelvec(self, subtraction_value, d, max_level, levelvec):
        """This method updates subtraction value so that the maximum level is only reached if level is max. We also
        check that subtraction value is not too large.

        :param subtraction_value: Current value of subtraction_value
        :param d: Respective dimension for subtraction_value
        :param max_level: Maximum level in this dimension.
        :param levelvec:
        :return:
        """
        if levelvec[d] - subtraction_value >= max_level and levelvec[d] < self.lmax[d]:
            subtraction_value = levelvec[d] - max_level + 1
        subtraction_value = min(subtraction_value, levelvec[d] - self.lmin[d])
        return subtraction_value

    def get_subtraction_value(self, refineObj: RefinementObjectSingleDimension, refineContainer: RefinementContainer, i: int, max_coarsenings: Sequence[int], d: int, levelvec: Sequence[int]) -> int:
        """This method calculates the subtraction value according to the version. See publication toDo

        :param refineObj: RefinementObject for which we want to calculate subraction value
        :param refineContainer: RefinementContainer of dimension d
        :param i: Index of RefineObject in RefinementContainer
        :param max_coarsenings: Maximum coarsening values for all dimensions
        :param d: Dimension of RefinementObject
        :param levelvec: Level vector of component grid
        :return: The subtraction value (int)
        """
        if self.version == 5:
            if tuple((d, i)) in self.max_level_dict:
                if tuple((d, self.max_level_dict[tuple((d, i))])) in self.subtraction_value_cache:
                    subtraction_value = self.subtraction_value_cache[tuple((d, self.max_level_dict[tuple((d, i))]))]
                    return self.modify_according_to_levelvec(subtraction_value, d, self.max_level_dict[tuple((d, i))],
                                                             levelvec)
            #if self.combischeme.has_forward_neighbour(levelvec):
                #subtraction_value = None
                #for k in range(self.dim):
                #    if k != d:
                #        levelvec_temp = list(levelvec)
                #        levelvec_temp[k] += 1
                #        if self.combischeme.in_index_set(levelvec_temp):
                #            subtract_value_temp = self.get_subtraction_value(refineObj, refineContainer, i,
                #                                              max_coarsenings, d, levelvec_temp)
                #            #print(subtraction_value, subtract_value_temp, levelvec, d)
                #            if subtraction_value is not None:
                #                assert subtract_value_temp == subtraction_value
                #            else:
                #                subtraction_value = subtract_value_temp
                #return self.modify_according_to_levelvec(subtraction_value, d, self.max_level_dict[tuple((d,i))], levelvec)


        if self.version == 4 or self.version == 5:
            max_coarsenings_levelvec = max_coarsenings
        if 2 <= self.version <= 8:
            max_level = self.get_max_level(refineContainer, refineObj, i, d)
            subtraction_value = (self.lmax[d] - max_level)
            self.max_level_dict[tuple((d,i))] = max_level
            if self.version == 6:
                m = 0
                partial_sum = 0
                while True:
                    if m > 0:
                        partial_sum += sum([1 for i in range(self.dim) if max_coarsenings[i] >= subtraction_value - (m - 1) ])
                    partial_sum_temp = sum([1 for i in range(d+1) if max_coarsenings[i] >= subtraction_value - m])
                    if partial_sum + partial_sum_temp <= subtraction_value:
                        m += 1
                    if partial_sum + partial_sum_temp >= subtraction_value:
                        break
                return self.modify_according_to_levelvec(m,d,max_level,levelvec)
            if self.version == 7:
                m = 0
                partial_sum = 0
                while True:
                    partial_sum += sum([1 for i in range(self.dim) if max_coarsenings[i] >= subtraction_value - m])
                    if partial_sum <= subtraction_value:
                        m += 1
                    if partial_sum >= subtraction_value:
                        break
                return self.modify_according_to_levelvec(m,d,max_level,levelvec)
            if self.version == 8:
                m = 0
                partial_sum = 0
                while True:
                    if m > 0:
                        partial_sum += min(max_level - 1, sum([1 for i in range(self.dim) if max_coarsenings[i] >= subtraction_value - (m - 1) ]))
                    partial_sum_temp = min(max_level - 1, sum([1 for i in range(d+1) if max_coarsenings[i] >= subtraction_value - m]))
                    if partial_sum + partial_sum_temp <= subtraction_value:
                        m += 1
                    if partial_sum + partial_sum_temp >= subtraction_value:
                        break
                return self.modify_according_to_levelvec(m,d,max_level,levelvec)
            if (self.version == 4 or self.version == 5) and max_level > 2:
                # if levelvec[d] > self.lmax[d] - subtraction_value:
                #    subtraction_value = self.lmax[d]
                # print(max_coarsenings)
                max_coarsening_other_dims = max([max_coarsenings_levelvec[i] if i != d else 0 for i in range(self.dim)])
                if self.version == 5:
                    max_coarsening_other_dims = min(subtraction_value, max_coarsening_other_dims)
                if max_coarsening_other_dims != 0 and tuple((d, max_level)) not in self.subtraction_value_cache:
                    if max_coarsening_other_dims < subtraction_value:
                        subtraction_value_new = subtraction_value - max_coarsening_other_dims
                        max_coarsenings_temp = list(max_coarsenings_levelvec)
                        max_coarsenings_temp[d] = subtraction_value
                        # if self.version == 5:
                        #    max_coarsenings_temp = [coarsening if coarsening < subtraction_value else subtraction_value for coarsening in max_coarsenings_temp]
                        max_coarsenings_temp[d] -= subtraction_value_new
                        assert subtraction_value >= subtraction_value_new
                        remainder = subtraction_value - subtraction_value_new
                    else:
                        max_coarsenings_temp = [subtraction_value if coarsening > subtraction_value else coarsening for
                                                coarsening in max_coarsenings_levelvec]
                        if self.version == 5:
                            remainder = subtraction_value
                        else:
                            remainder = subtraction_value - sum(
                                np.asarray(max_coarsenings_levelvec) - np.asarray(max_coarsenings_temp)) + max_coarsenings_levelvec[
                                            d] - subtraction_value
                        if self.version == 5:
                            # print(max_coarsenings_temp, max_coarsenings, remainder, sum(np.asarray(max_coarsenings) - np.asarray(max_coarsenings_temp)))
                            assert remainder == subtraction_value
                        subtraction_value_new = 0
                    while (remainder > 0):
                        #print(max_coarsenings_temp, remainder)
                        print(subtraction_value_new, remainder, "d", d, "i", i)
                        max_coarsening = max(max_coarsenings_temp)
                        second_largest_coarsening = max(
                            [coarsening if coarsening < max_coarsening else 0 for coarsening in max_coarsenings_temp])
                        assert max_coarsening > 0
                        # print(remainder, max_coarsening, second_largest_coarsening, max_coarsenings_temp)
                        # print(max_coarsening, second_largest_coarsening, max_coarsenings_temp)
                        max_dimensions = [1 if coarsening == max_coarsening else 0 for coarsening in
                                          max_coarsenings_temp]
                        num_max_coarsening = min(sum(max_dimensions), min((max_level - 1), sum([1 if self.lmax[k] > 2 else 0 for k in range(self.dim)])))
                        #print("num_max_coarsening", num_max_coarsening, sum(max_dimensions), (max_level - 1), sum([1 if self.lmax[k] > 2 else 0 for k in range(self.dim)]))
                        #print(num_max_coarsening, self.combischeme.lmax_adaptive - max(sum([1 if self.lmax[k] <= 2 else 0 for k in range(self.dim)]), 1), self.combischeme.lmax_adaptive, self.lmax)
                        my_position = sum(max_dimensions[:d])
                        if remainder >= (max_coarsening - second_largest_coarsening) * num_max_coarsening:
                            subtraction_value_new += max_coarsening - second_largest_coarsening
                            remainder -= (max_coarsening - second_largest_coarsening) * num_max_coarsening
                        else:

                            added_subtraction_value = int(remainder / num_max_coarsening)
                            print(added_subtraction_value)
                            #subtraction_value_new += added_subtraction_value
                            if added_subtraction_value - int(
                                    added_subtraction_value) > my_position / num_max_coarsening:
                                subtraction_value_new += int(math.ceil(added_subtraction_value))
                            else:
                                subtraction_value_new += int(added_subtraction_value)
                            break
                        max_coarsenings_temp = [coarsening if coarsening < max_coarsening else second_largest_coarsening
                                                for coarsening in max_coarsenings_temp]
                        assert subtraction_value >= subtraction_value_new
                        print(subtraction_value_new, remainder, "d", d, "i", i)
                    if True:
                        m = 0
                        partial_sum = 0
                        while True:
                            partial_sum += sum(
                                [1 for i in range(self.dim) if max_coarsenings[i] >= subtraction_value - m])
                            if partial_sum <= subtraction_value:
                                m += 1
                            if partial_sum >= subtraction_value:
                                break
                    if subtraction_value_new != m:
                        print("subtraction_value", subtraction_value_new, "m", m, "d", d, "i", i)
                        print("max_coarsenings", max_coarsenings)
                    assert subtraction_value_new == m
                    assert subtraction_value >= subtraction_value_new
                    subtraction_value = subtraction_value_new
                    # print(subtraction_value, d, max_level, max_coarsenings)
                    self.subtraction_value_cache[tuple((d, max_level))] = subtraction_value
                else:
                    if tuple((d, max_level)) in self.subtraction_value_cache:
                        subtraction_value = self.subtraction_value_cache[tuple((d, max_level))]
                subtraction_value = self.modify_according_to_levelvec(subtraction_value, d, self.max_level_dict[tuple((d, i))], levelvec)
                # print(subtraction_value, d, max_level, max_coarsenings)
            if self.version == 3 and max_level > 2:
                subtraction_value /= self.dim
                if subtraction_value - int(subtraction_value) > d / self.dim:
                    subtraction_value = int(math.ceil(subtraction_value))
                else:
                    subtraction_value = int(subtraction_value)
            # subtraction_value = max(refineObj.coarsening_level, next_refineObj.coarsening_level) if next_refineObj is not None else refineObj.coarsening_level

        else:
            subtraction_value = 0
        return subtraction_value

    def get_max_level(self, refine_container:RefinementContainer, refine_obj: RefinementObjectSingleDimension, i: int, d: int):
        if not tuple((d, i)) in self.max_level_dict:
            max_level = refine_obj.levels[1]
            k = 0
            while (i - k > 0):
                refine_obj_temp = refine_container.get_objects()[i - k]
                max_level = max(max_level, refine_obj_temp.levels[0])
                if refine_obj_temp.levels[0] <= refine_obj.levels[1]:
                    break
                k += 1
            k = 1
            while i + k < len(refine_container.get_objects()):
                refine_obj_temp = refine_container.get_objects()[i + k]
                max_level = max(max_level, refine_obj_temp.levels[1])
                if refine_obj_temp.levels[1] <= refine_obj.levels[1]:
                    break
                k += 1
        else:
            max_level = self.max_level_dict[tuple((d, i))]
        return max_level

    def is_child(self, level_left_point: int, level_point: int, level_right_point: int) -> bool:
        """This method returns if the point of level level_point is a child in the refinement structure.
        A point with level <= 1 is not considered to be a child.

        :param level_left_point: level of point to left
        :param level_point: level of point
        :param level_right_point: level of point to right
        :return: Bool if point is a child
        """
        return (level_left_point < level_point and level_right_point < level_point) and level_point > 1
        #if level_left_point < level_point or level_right_point < level_point:
        #    return True
        #else:
        #    return False

    def get_node_info(self, position: int, coords_dim: Sequence[float], level_dim: Sequence[int], d: int) -> NodeInfo:
        """This method calculates the left and right parent of a child. It might happen that a child has already a child
        in one direction but it may not have one in both as it would not be considered to be a child anymore.

        :param position: Position of child in coords_dim and level_dim
        :param coords_dim: Coordinates of all points in the dimension d
        :param level_dim:  Level of all points in the dimension d
        :param d: Current dimension
        :return: NodeInfo object containing parents and other info for error calculation
        """
        child = coords_dim[position]
        level_child = level_dim[position]
        if self.equidistant:
            width = (self.b[d] - self.a[d]) / 2**level_child
            right_parent_of_right_parent = child + 2* width if level_child > 1 else None
            left_parent_of_left_parent = child - 2 * width if level_child > 1 else None
            return NodeInfo(child, child - width, child + width, left_parent_of_left_parent , right_parent_of_right_parent ,True, True, None,None, level_child)
        else:
            left_parent = None
            level_parent = None
            left_of_left_parent = None
            for i in reversed(range(position)):
                if level_dim[i] < level_child and left_parent is None:
                    left_parent = coords_dim[i]
                    level_parent = level_dim[i]
                if left_parent is not None and level_dim[i] < level_parent:
                    left_of_left_parent = coords_dim[i]
                    break

            assert left_parent is not None
            right_parent = None
            level_parent = None
            right_of_right_parent = None
            for i in (range(position+1, len(coords_dim))):
                if level_dim[i] < level_child and right_parent is None:
                    right_parent = coords_dim[i]
                    level_parent = level_dim[i]
                if right_parent is not None and level_dim[i] < level_parent:
                    right_of_right_parent = coords_dim[i]
                    break
            assert right_parent is not None
            return NodeInfo(child, left_parent, right_parent, left_of_left_parent, right_of_right_parent, True, True, None,None, level_child)

    # this method draws the 1D refinement of each dimension individually
    def draw_refinement(self, filename: str=None, markersize:int =20, fontsize=60, single_dim:int=None, fill_boundary_points:bool=False, showPlot: bool = True):  # update with meta container
        plt.rcParams.update({'font.size': fontsize})
        refinement = self.refinement
        dim = self.dim if single_dim is None else 1
        fig, ax = plt.subplots(ncols=1, nrows=dim, figsize=(20, 5*dim))
        offset = 0 if single_dim is None else single_dim
        for d in range(dim):
            axis = ax[d] if single_dim is None else ax
            objs = refinement.refinementContainers[d+offset].get_objects()
            infinite_bounds = isinf(objs[0].start)
            if infinite_bounds:
                # Ignore refinement objects with infinite borders
                objs = objs[1:-1]
            starts = [refinementObject.start for refinementObject in objs]
            starts_levels = [refinementObject.levels[0] for refinementObject in objs]
            ends = [refinementObject.end for refinementObject in objs]
            ends_levels = [refinementObject.levels[1] for refinementObject in objs]
            for i in range(len(starts)):
                axis.add_patch(
                    patches.Rectangle(
                        (starts[i], -0.1),
                        ends[i] - starts[i],
                        0.2, linestyle='-',
                        fill=False  # remove background
                    )
                )
                axis.text(starts[i]+0.015, 0.01, str(starts_levels[i]),
                          fontsize=fontsize-10, ha='center', color="blue")
            axis.text(ends[-1] - 0.015, 0.01, str(ends_levels[-1]),
                fontsize=fontsize-10, ha='center', color="blue")
            xValues = starts + ends
            yValues = np.zeros(len(xValues))
            if fill_boundary_points:
                axis.plot(xValues, yValues, 'bo', markersize=markersize, color="black")
            else:
                axis.plot(xValues[1:-1], yValues[1:-1], 'bo', markersize=markersize, color="black")
                axis.plot([xValues[0], xValues[-1]], [yValues[0], yValues[-1]], 'bo', markersize=markersize, color="black", fillstyle='none')
            if infinite_bounds:
                start, end = objs[0].start, objs[-1].end
                # offset_bound = (end - start) * 0.1
            else:
                start, end = self.a[d], self.b[d]
                # offset_bound = (end - start) * 0.005
            axis.set_xlim([start-0.005, end+0.005])
            axis.set_ylim([-0.05, 0.05])
            axis.set_yticks([])
            axis.set_title("$x_" + str(d + 1 + offset) + "$")

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        if showPlot:
            plt.show()
        plt.close()
        return fig

    def draw_refinement_trees(self, filename: str=None, markersize:int =20, fontsize=20, single_dim:int=None, fill_boundary_points:bool=False):
        """This method plots the refinement trees of the current refinement structure.

        :param filename: Will save plot to specified filename if set.
        :param markersize: Specifies the used marker size for plotting
        :param fontsize: Specifies the used fontsize for plotting
        :param single_dim: Can be set when only specified dimension should be plotted
        :return: Figure object of plot
        """
        plt.rcParams.update({'font.size': fontsize})
        refinement = self.refinement
        dim = self.dim if single_dim is None else 1
        height = 5*sum(self.lmax) if single_dim is None else 5 * self.lmax[single_dim]
        fig, ax = plt.subplots(ncols=1, nrows=dim if single_dim is None else 1, figsize=(20, height))
        offset = 0 if single_dim is None else single_dim
        for d in range(dim):
            axis = ax[d] if single_dim is None else ax
            starts = [refinementObject.start for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            starts_levels = [refinementObject.levels[0] for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            ends = [refinementObject.end for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            ends_levels = [refinementObject.levels[1] for refinementObject in refinement.refinementContainers[d + offset].get_objects()]
            max_level = max(starts_levels)
            yValues = np.zeros(len(starts) + 1)
            for i in range(len(starts)):
                x_position = starts[i]
                y_position = starts_levels[i]
                yValues[i] = y_position
                if i !=0:
                    for j in reversed(range(i)):
                        if starts_levels[j] == starts_levels[i] + 1:
                            target_x_position = starts[j]
                            target_y_position = starts_levels[j]
                            axis.arrow(x_position, y_position, target_x_position - x_position + 0.001,
                                        target_y_position - y_position - 0.04,
                                        head_width=0.04, head_length=0.04, fc='k', ec='k', length_includes_head=True, overhang=0, capstyle="butt")
                            break
                        if starts_levels[j] <= starts_levels[i]:
                            break

                for j in range(i+1, len(starts)):
                    if starts_levels[j] == starts_levels[i] + 1:
                        target_x_position = starts[j]
                        target_y_position = starts_levels[j]
                        axis.arrow(x_position, y_position, target_x_position - x_position - 0.001,
                                    target_y_position - y_position - 0.04,
                                    head_width=0.04, head_length=0.04, fc='k', ec='k', length_includes_head=True)
                        break
                    if starts_levels[j] <= starts_levels[i]:
                        break


            x_position = ends[-1]
            y_position = ends_levels[-1]
            yValues[-1] = y_position
            for j in reversed(range(len(starts))):
                if ends_levels[-1] + 1 == starts_levels[j]:
                    target_x_position = starts[j]
                    target_y_position = starts_levels[j]
                    axis.arrow(x_position, y_position, target_x_position - x_position + 0.001,
                                target_y_position - y_position - 0.04,
                                head_width=0.04, head_length=0.04, fc='k', ec='k', length_includes_head=True)
                    break
                if ends_levels[-1] >= starts_levels[j]:
                    break



            xValues = starts + ends[-1:]
            if fill_boundary_points:
                axis.plot(xValues, yValues, 'bo', markersize=markersize, color="black")
            else:
                axis.plot(xValues[1:-1], yValues[1:-1], 'bo', markersize=markersize, color="black")
                axis.plot([xValues[0], xValues[-1]], [yValues[0], yValues[-1]], 'bo', markersize=markersize,
                          color="black", fillstyle='none')
            axis.set_xlim([self.a[d + offset]-0.01, self.b[d + offset]+0.01])
            axis.set_ylim([-0.04, max_level+0.04])
            axis.set_ylim(axis.get_ylim()[::-1])
            axis.set_yticks(list(list(range(max_level+1))))
            axis.set_ylabel("level")
            axis.set_title("$x_" + str(d + 1 + offset) + "$")
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        #plt.show()
        return fig

    def init_evaluation_operation(self, areas):
        self.operation.initialize_evaluation_dimension_wise(areas[0])

    def evaluate_operation_area(self, component_grid: ComponentGridInfo, area, additional_info=None):
        if self.grid.is_global():
            # get 1d coordinates of the grid points that define the grid; they are calculated based on the levelvector
            gridPointCoordsAsStripes, grid_point_levels, children_indices = self.get_point_coord_for_each_dim(component_grid.levelvector)

            # calculate the operation on the grid
            self.log_util.time_func("spatAdaptDimWise: calculate_operation_dimension_wise time taken ", self.operation.calculate_operation_dimension_wise, gridPointCoordsAsStripes, grid_point_levels, component_grid)

            # compute the error estimates for further refining the Refinementobjects and therefore the future grid
            if not self.errorEstimator.is_global:
                self.log_util.time_func("spatAdaptDimWise: compute_error_estimates_dimension_wise time taken ", self.compute_error_estimates_dimension_wise, gridPointCoordsAsStripes, grid_point_levels, children_indices, component_grid)

            # save the number of evaluations used per d-1 dimensional slice
            #for d in range(self.dim):
            #    factor = component_grid.coefficient if self.grid.isNested() else 1
            #    self.evaluationCounts[d][component_grid.levelvector[d] - 1] += factor * np.prod([self.grid.numPoints[d2] if d2 != d else 1 for d2 in range(self.dim)])
            return np.prod(self.grid.numPoints)
        else:
            pass

    # This method computes additional values after the compution of the integrals for the current
    # refinement step is finished. This method is executed before the refinement process.
    def finalize_evaluation_operation(self, areas, evaluation_array):
        # TODO: global error calculation
        ### global error calc goes here
        if self.errorEstimator.is_global:
            self.errorEstimator.calc_global_error(self.operation.validation_set, self)
        super().finalize_evaluation_operation(areas, evaluation_array)
        #self.refinement.print_containers_only()
        #if self.version == 1:
        #    for d in range(self.dim):
        #        container_d = self.refinement.get_refinement_container_for_dim(d)
        #        for area in container_d.get_objects():
        #            level = max(area.levels)
        #            area.set_evaluations(np.sum(self.evaluationCounts[d][level-1:]))

    def _initialize_points(self, points, func_mid: Callable[[float, float], float], d: int, i1: int, i2: int) -> None:
        """This method recursively initializes the point coordinates in the refinement structure.

        :param points: numpy array where points should be stored in
        :param func_mid: Function determining the center of two points
        :param d: Current dimension
        :param i1: Index of left point
        :param i2: Index of right point
        :return: None
        """
        if i1+1 >= i2:
            return
        i = (i1 + i2) // 2
        points[i] = func_mid(points[i1], points[i2], d)
        self._initialize_points(points, func_mid, d, i1, i)
        self._initialize_points(points, func_mid, d, i, i2)

    def _initialize_levels(self, levels, i1, i2, level):
        """This method recursively initializes the levels of the points in the refinement structure.

        :param levels: numpy array where levels should be stored in
        :param i1: Index of left point
        :param i2: Index of right point
        :param level: Current level
        :return:
        """
        if i1+1 >= i2:
            return
        i = (i1 + i2) // 2
        level = level+1
        levels[i] = level
        self._initialize_levels(levels, i1, i, level)
        self._initialize_levels(levels, i, i2, level)

    def initialize_refinement(self):
        initial_points = []
        maxv = self.lmax[0]
        assert maxv > 1
        assert all([l == maxv for l in self.lmax])
        num_points = 2 ** maxv + 1
        levels = [0 for _ in range(num_points)]
        self._initialize_levels(levels, 0, num_points-1, 0)
        func_mid = self.grid.get_mid_point
        for d in range(self.dim):
            points = [None for _ in range(num_points)]
            points[0] = self.a[d]
            points[num_points-1] = self.b[d]
            if self.chebyshev_points:
                points = np.linspace(0,1, 2**maxv + 1)
                points = [self.a[d] + (self.b[d]- self.a[d]) * (1 - math.cos(p * math.pi)) / 2 for p in points]
            else:
                self._initialize_points(points, func_mid, d, 0, num_points-1)
            initial_points.append(np.array(points))
        self.refinement = MetaRefinementContainer([RefinementContainer
                                                   ([RefinementObjectSingleDimension(initial_points[d][i],
                                                                                     initial_points[d][i + 1], d, self.dim, list((levels[i], levels[i+1])), grid=self.grid,
                                                                                     coarsening_level=0, a=self.a[d], b=self.b[d], chebyshev=self.chebyshev_points) for i in
                                                     range(2 ** maxv)], d, self.errorEstimator) for d in
                                                   range(self.dim)], calculate_volume_weights=self.use_volume_weighting)
        if self.dim_adaptive:
            self.combischeme.init_adaptive_combi_scheme(self.lmax[0], self.lmin[0])
        #self.evaluationCounts = [np.zeros(self.lmax[d]) for d in range(self.dim)]
        if self.operation is not None:
            self.operation.init_dimension_wise(self.grid, self.grid_surplusses, self.refinement, self.lmin, self.lmax, self.a, self.b, self.version)


    def get_areas(self):
        if (self.grid.is_global() == True):
            return [self.refinement]
        assert False
        # get a list of lists which contains range(refinements[d]) for each dimension d where the refinements[d] are the number of subintervals in this dimension
        #indices = [list(range(len(refineDim))) for refineDim in self.refinement.get_new_objects()]
        # this command creates tuples of size this_dim of all combinations of indices (e.g. this_dim = 2 indices = ([0,1],[0,1,2,3]) -> areas = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)] )
        #return list(zip(*[g.ravel() for g in np.meshgrid(*indices)]))

    def get_new_areas(self):
        return self.get_areas()

    def do_refinement(self, area, position):
        # print("-------------------\nREFINING", position)
        lmaxChange = self.refinement.refine(position)
        # the following is currently solved by initializing all data structures anew before each evalute_integral()
        refinement_dim = position[0]
        #max_level = self.rebalance(refinement_dim)
        #if lmaxChange is not None and max_level <= self.lmax[refinement_dim]:
        #    lmaxChange = None
        #if lmaxChange is not None:
        #    for d in range(self.dim):
        #        if lmaxChange[d] != 0:
        #            self.raise_lmax(d, lmaxChange[d])
        #    #print("New scheme:")
        #    self.scheme = self.combischeme.getCombiScheme(self.lmin[0], self.lmax[0], do_print=False)
        #    return False
        return False

    def raise_lmax(self, d: int, value: int) -> None:
        """This method raises the maximum level of the combination scheme if previous one is exceeded by refinement.

        :param d: Dimension where we want to increase level.
        :param value: Value by which maximum level is increased.
        :return: None
        """
        self.lmax[d] += value
        if self.dim_adaptive:
            self.log_util.log_debug("New lmax: {0}".format(self.lmax))
            while (True):
                refinements = 0
                active_indices = set(self.combischeme.get_active_indices())
                for index in active_indices:
                    if max(self.lmax) + self.lmin[0] * (self.dim - 1) > sum(index) and all(
                            [self.lmax[d] > index[d] for d in range(self.dim)]):
                        self.combischeme.update_adaptive_combi(index)
                        refinements += 1
                if refinements == 0:
                    break

    def rebalance(self, d: int) -> None:
        """This method rebalances the refinement tree of dimension d. See also publication toDo

        :param d: Dimension of refinement tree.
        :return: None
        """
        refinement_container = self.refinement.get_refinement_container_for_dim(d)
        self.rebalance_interval(0, refinement_container.size(), 1, refinement_container)
        #refinement_container.printContainer()

    def rebalance_interval(self, start, end, level, refinement_container):
        """This method recursively rebalances the refinement tree.

        :param start: Start of interval in which we want to rebalance.
        :param end: End of interval in which we want to rebalance
        :param level: Current level of the rebalancing procedure.
        :param refinement_container: RefinementContainer that stores the RefinementObjects.
        :return: None
        """
        if end - start <= 2:
            return
        refineContainer = refinement_container
        position_level = None
        position_level_1_left = None
        position_level_1_right = None
        for i, refinement_object in enumerate(refineContainer.get_objects()[start:end]):
            if refinement_object.levels[1] == level:
                position_level = i
            if refinement_object.levels[1] == level + 1:
                if position_level_1_left is None and position_level is None:
                    position_level_1_left = i
                elif position_level is not None:
                #else:
                    #assert position_level_1_right is None
                    assert position_level_1_right is None
                    position_level_1_right = i
                else:
                    assert False
                #if position_level is not None and position_level_1_left is not None and position_level_1_right is not None:
                #    break

        #refineContainer.printContainer()
        #print(refinement_object.this_dim, position_level, position_level_1_left, position_level_1_right, start, end, level )
        # if position_level is None:
        #     print('stop')
        # if position_level_1_left is None:
        #     print('stop')
        # if position_level_1_right is None:
        #     print('stop')
        assert position_level is not None
        #assert position_level_1_right is not None
        #assert position_level_1_left is not None
        new_leaf_reached = False
        #print(i+2, end - start + 1, (i + 2) / (end - start + 1), i, start, end, level)
        if position_level_1_right is not None and abs((position_level) / (end-start - 2) - 0.5) > abs((position_level_1_right) / (end-start - 2) - 0.5) + self.rebalancing_safety_factor:
            assert position_level < position_level_1_right
            position_new_leaf = None
            self.log_util.log_debug("Rebalancing!")
            for j, refinement_object in enumerate(refineContainer.get_objects()[start:end]):
                if j < end - start - 1:
                    next_refinement_object = refineContainer.get_object(j+1+start)
                    if j <= position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1
                    elif j == position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1

                    else:
                        if refinement_object.levels[1] == level + 1:
                            assert j == position_level_1_right
                            new_leaf_reached = True
                            position_new_leaf = start + j
                        if j > position_level and new_leaf_reached:
                            refinement_object.levels[1] -= 1
                            next_refinement_object.levels[0] -= 1
            assert position_new_leaf is not None
            self.rebalance_interval(start,position_new_leaf + 1, level + 1, refinement_container)
            self.rebalance_interval(position_new_leaf + 1, end, level + 1, refinement_container)
            return
            #refineContainer.printContainer()

        new_leaf_reached = True
        if position_level_1_left is not None and abs((position_level) / (end-start - 2) - 0.5) > abs((position_level_1_left) / (end-start - 2) - 0.5) + self.rebalancing_safety_factor:
            assert position_level_1_left < position_level
            position_new_leaf = None
            self.log_util.log_debug("Rebalancing!")
            for j, refinement_object in enumerate(refineContainer.get_objects()[start:end]):
                if j < end - start - 1:
                    next_refinement_object = refineContainer.get_object(j+1+start)

                    if j >= position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1

                    elif j == position_level:
                        refinement_object.levels[1] += 1
                        next_refinement_object.levels[0] += 1

                    else:
                        if j < position_level and new_leaf_reached:
                            refinement_object.levels[1] -= 1
                            next_refinement_object.levels[0] -= 1
                        if refinement_object.levels[1] == level:
                            assert j == position_level_1_left
                            new_leaf_reached = False
                            position_new_leaf = start + j
            assert position_new_leaf is not None
            self.rebalance_interval(start, position_new_leaf + 1, level + 1, refinement_container)
            self.rebalance_interval(position_new_leaf + 1, end, level + 1, refinement_container)
            return

        self.rebalance_interval(start, start + position_level + 1, level + 1, refinement_container)
        self.rebalance_interval(start + position_level + 1, end, level + 1, refinement_container)

            #refineContainer.printContainer()

    def update_coarsening_values(self, refinement_container_d, d):
        """This method checks if any of the RefinementObject exceeds the maximum level and returns this value.

        :param refinement_container_d: RefinementContainer for dimension d
        :param d: Dimension where we want to update the values.
        :return:
        """
        update_dimension = 0
        for refinement_object in refinement_container_d.get_objects():
            refinement_object.coarsening_level = self.lmax[d] - max(refinement_object.levels)
            if refinement_object.coarsening_level < update_dimension:
                update_dimension = refinement_object.coarsening_level
            #assert refinement_object.coarsening_level >= -1
        return update_dimension * -1

    def refinement_postprocessing(self):
        self.subtraction_value_cache = {}
        self.max_level_dict = {}
        self.refinement.apply_remove(sort=True)
        self.refinement.refinement_postprocessing()
        self.refinement.reinit_new_objects()
        #self.evaluationCounts = [np.zeros(self.lmax[d]) for d in range(self.dim)]
        if self.rebalancing:
            for d in range(self.dim):
                self.rebalance(d)
        for d in range(self.dim):
            refinement_container_d = self.refinement.get_refinement_container_for_dim(d)
            # search if a refinement_object exceeds the maximum level and by how much
            update_d = self.update_coarsening_values(refinement_container_d, d)
            if update_d > 0: # adjust maximum level if necessary
                self.raise_lmax(d, update_d)
                refinement_container_d.update_values(update_d)
        self.scheme = self.combischeme.getCombiScheme(do_print=False)

    def compute_error_estimates_dimension_wise(self, gridPointCoordsAsStripes, grid_point_levels, children_indices, component_grid):
        self.grid_surplusses.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        self.grid.set_grid(gridPointCoordsAsStripes, grid_point_levels)
        if isinstance(self.errorEstimator, ErrorCalculatorSingleDimMisclassificationGlobal):
            pass
            #self.errorEstimator.calc_global_error(self.operation.validation_set, self)
        else:
            self.calculate_surplusses(gridPointCoordsAsStripes, children_indices, component_grid)
    """
    outdated functions
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
        points_children = get_cross_product([self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [child] for d2 in range(self.dim)])
        #points_children = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [child] for d2 in range(self.dim)])]))
        indices = get_cross_product([range(len(self.grid_surplusses.get_coordinates_dim(d2))) if d != d2 else [1] for d2 in range(self.dim)])
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
                volume += factor * abs(value) * (self.operation.get_surplus_width(d, right_parent, left_parent))**exponent
        if self.version == 0 or self.version == 2:
            evaluations = len(points_children) #* (1 + int(left_parent_in_grid) + int(right_parent_in_grid))
        else:
            evaluations = 0
        return abs(volume), evaluations

    # Sum up the 1-d surplusses along the dim-1 dimensional slice through the point child in dimension d.
    #  The surplusses are calculated based on the left and right parents.
    def sum_up_volumes_for_point_vectorized(self, child_info: NodeInfo, grid_points: Sequence[Sequence[float]], d: int, component_grid:ComponentGridInfo):
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

        size_slize = np.prod([self.grid_surplusses.numPoints[d2] if d2 != d else 1 for d2 in range(self.dim)])
        # avoid evaluating on boundary points if grids has none
        if left_parent_in_grid:
            if isinf(right_parent):
                factor_left_parent = 1.0
            else:
                factor_left_parent = (right_parent - child)/(right_parent - left_parent)
            points_left_parent = get_cross_product_list([self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [left_parent] for d2 in range(self.dim)])
            #points_left_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2]if d != d2 else [left_parent] for d2 in range(self.dim)])]))
        if right_parent_in_grid:
            if isinf(left_parent):
                factor_right_parent= 1.0
            else:
                factor_right_parent = (child - left_parent)/(right_parent - left_parent)
            points_right_parent = get_cross_product_list([self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [right_parent] for d2 in range(self.dim)])
            #points_right_parent = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [right_parent] for d2 in range(self.dim)])]))
        if self.grid_surplusses.modified_basis and not right_parent_in_grid:
            left_of_left_parents = get_cross_product_list([self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [left_parent_of_left_parent] for d2 in range(self.dim)])
        if self.grid_surplusses.modified_basis and not left_parent_in_grid:
            right_of_right_parents = get_cross_product_list([self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [right_parent_of_right_parent] for d2 in range(self.dim)])

        points_children = get_cross_product_list([self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [child] for d2 in range(self.dim)])
        #points_children = list(zip(*[g.ravel() for g in np.meshgrid(*[self.grid_surplusses.coords[d2] if d != d2 else [child] for d2 in range(self.dim)])]))
        indices = get_cross_product_list([range(len(self.grid_surplusses.get_coordinates_dim(d2))) if d != d2 else [1] for d2 in range(self.dim)])
        #indices = list(zip(*[g.ravel() for g in np.meshgrid(*[range(len(self.grid_surplusses.coords[d2])) if d != d2 else None for d2 in range(self.dim)])]))
        #index = indices[i]
        factors = np.prod(np.asarray(get_cross_product_list([self.grid_surplusses.weights[d2] if d != d2 else [1] for d2 in range(self.dim)])), axis=1).reshape((size_slize, 1))
        #factors = np.prod(np.asarray([[self.grid_surplusses.weights[d2][index[d2]] if d2 != d else 1 for d2 in range(self.dim)] for index in indices]), axis=1).reshape((size_slize, 1))
        #factor2 = np.prod([self.grid.weights[d2][index[d2]]  if d2 != d else self.grid.weights[d2][index_child] for d2 in range(self.dim)])
        exponent = 1# if not self.do_high_order else 2
        #if factor2 != 0:
        #print(points_children[i], self.f.f_dict.keys())
        # avoid evaluating on boundary points if grids has none
        #assert (tuple(point_child) in self.f.f_dict)
        point_values = self.operation.get_point_values_component_grid(points_children, component_grid) #np.asarray([self.f(p) for p in points_children])
        values = point_values
        if left_parent_in_grid:
            point_values_left_parent = self.operation.get_point_values_component_grid(points_left_parent, component_grid) #np.asarray([self.f(p) for p in points_left_parent])
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
                    point_values_left_of_left_parent = self.operation.get_point_values_component_grid(left_of_left_parents, component_grid) #np.asarray([self.f(p) for p in left_of_left_parents])

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
            point_values_right_parent = self.operation.get_point_values_component_grid(points_right_parent, component_grid) #np.asarray([self.f(p) for p in points_right_parent])
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
                    point_values_right_of_right_parent = self.operation.get_point_values_component_grid(right_of_right_parents, component_grid) #np.asarray([self.f(p) for p in right_of_right_parents])
                    m = (point_values_right_of_right_parent - point_values_right_parent)  / (right_parent_of_right_parent - right_parent)
                    previous_value_at_child = m * (child - right_parent) + point_values_right_parent
                    values = point_values - previous_value_at_child
                    #print("Hey", m, previous_value_at_child, value, (self.f(tuple(right_of_right_parent)) - self.f(points_right_parent[i])), (right_of_right_parent - right_parent))
                    #assert(tuple(right_of_right_parent) in self.f.f_dict)
            else:
                #print(points_right_parent[i], self.f.f_dict.keys())
                #assert point_right_parent in self.f.f_dict or self.grid.weights[d][index_right_parent] == 0
                values -= factor_right_parent * point_values_right_parent
        volume = np.sum(factors * abs(values), axis=0) * (self.operation.get_surplus_width(d, right_parent, left_parent))**exponent

        if self.version == 0 or self.version == 2:
            evaluations = size_slize #* (1 + int(left_parent_in_grid) + int(right_parent_in_grid))
        else:
            evaluations = 0
        return abs(volume), evaluations
    """

    # Sum up the 1-d surplusses along the dim-1 dimensional slice through the point child in dimension d.
    #  The surplusses are calculated based on the left and right parents.
    def sum_up_volumes_for_point_completely_vectorized(self, child_infos: NodeInfo, grid_points: Sequence[Sequence[float]],
                                            d: int, component_grid: ComponentGridInfo):
        children = np.asarray([child_info.child for child_info in child_infos])
        left_parents = np.asarray([child_info.left_parent for child_info in child_infos])
        right_parents = np.asarray([child_info.right_parent for child_info in child_infos])
        if self.grid_surplusses.modified_basis:
            left_parents_of_left_parents = np.asarray([child_info.left_parent_of_left_parent for child_info in child_infos], dtype=np.float)
            right_parents_of_right_parents = np.asarray([child_info.right_parent_of_right_parent for child_info in child_infos], dtype=np.float)

        left_parents_in_grid = np.logical_or(np.logical_not(np.isclose(left_parents, self.a[d])), self.grid_surplusses.boundary)
        right_parents_in_grid = np.logical_or(np.logical_not(np.isclose(right_parents, self.b[d])), self.grid_surplusses.boundary)
        size_slize = np.prod([self.grid_surplusses.numPoints[d2] if d2 != d else 1 for d2 in range(self.dim)])

        factors_left_parents = np.ones(len(left_parents[left_parents_in_grid]))
        filter = np.logical_not(np.isinf(right_parents[left_parents_in_grid]))
        factors_left_parents[filter] = (right_parents[left_parents_in_grid][filter] - children[left_parents_in_grid][filter]) / (right_parents[left_parents_in_grid][filter] - left_parents[left_parents_in_grid][filter])
        points_left_parents = [get_cross_product_list(
            [self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [left_parent] for d2 in
             range(self.dim)]) for left_parent in left_parents[left_parents_in_grid]]

        factors_right_parents = np.ones(len(right_parents[right_parents_in_grid]))
        filter = np.logical_not(np.isinf(left_parents[right_parents_in_grid]))
        factors_right_parents[filter] = (children[right_parents_in_grid][filter] - left_parents[right_parents_in_grid][filter]) / (right_parents[right_parents_in_grid][filter] - left_parents[right_parents_in_grid][filter])

        points_right_parents = [get_cross_product_list(
            [self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [right_parent] for d2 in
             range(self.dim)]) for right_parent in right_parents[right_parents_in_grid]]
        if self.grid_surplusses.modified_basis and np.any(np.logical_not(right_parents_in_grid)):
            filter_left_left = np.logical_and(left_parents_in_grid, np.logical_not(right_parents_in_grid))
            points_left_of_left_parents = [get_cross_product_list(
                [self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [left_parent_of_left_parent] for d2 in
                 range(self.dim)]) for left_parent_of_left_parent in left_parents_of_left_parents[filter_left_left]]
        if self.grid_surplusses.modified_basis and np.any(np.logical_not(left_parents_in_grid)):
            filter_right_right = np.logical_and(right_parents_in_grid, np.logical_not(left_parents_in_grid))
            points_right_of_right_parents = [get_cross_product_list(
                [self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [right_parents_of_right_parent] for d2
                 in range(self.dim)]) for right_parents_of_right_parent in right_parents_of_right_parents[filter_right_right]]
        points_children = [get_cross_product_list(
            [self.grid_surplusses.get_coordinates_dim(d2) if d != d2 else [child] for d2 in range(self.dim)]) for child in children]

        factors = np.prod(np.asarray(get_cross_product_list(
            [self.grid_surplusses.weights[d2] if d != d2 else [1] for d2 in range(self.dim)])), axis=1).reshape(
            (size_slize, 1))
        exponent = 1  # if not self.do_high_order else 2
        # assert (tuple(point_child) in self.f.f_dict)
        point_values = self.operation.get_point_values_component_grid_multiple(points_children, component_grid)
        values = np.array(point_values) # avoids changes to point_values
        if len(points_left_parents) > 0:
            point_values_left_parents = self.operation.get_point_values_component_grid_multiple(points_left_parents,
                                                                                          component_grid)

            if self.grid_surplusses.modified_basis:
                filter_target_indices = np.logical_and(left_parents_in_grid,np.logical_not(
                    right_parents_in_grid))  # indices for this if
                left_of_left_parents_not_in_grid = np.isclose(left_parents_of_left_parents[filter_target_indices],
                                                                self.a[d])
                # we have to cases if right of right parents are in grid or not
                filter_not_in_grid = left_of_left_parents_not_in_grid
                filter_target_indices_not_in_grid = np.logical_and(np.isclose(left_parents_of_left_parents, self.a[d]), filter_target_indices)
                if np.any(filter_not_in_grid):
                    point_values_left_parents_filtered = \
                    point_values_left_parents[np.logical_not(right_parents_in_grid[left_parents_in_grid])][
                        filter_not_in_grid]
                    values[filter_target_indices_not_in_grid] = point_values[filter_target_indices][
                                                                            filter_not_in_grid] - \
                                                                        point_values_left_parents_filtered

                filter_in_grid = np.logical_not(filter_not_in_grid)
                filter_target_indices_in_grid = np.logical_and(np.logical_not(np.isclose(left_parents_of_left_parents, self.a[d])), filter_target_indices)
                if np.any(filter_in_grid):
                    left_parents_of_left_parents_filtered = left_parents_of_left_parents[filter_target_indices][
                        filter_in_grid]
                    left_parents_filtered = left_parents[filter_target_indices][filter_in_grid]
                    children_filtered = children[filter_target_indices][filter_in_grid]
                    assert len(left_parents_of_left_parents_filtered) == len(left_parents_filtered) == len(children_filtered) == 1
                    point_values_left_parents_filtered = point_values_left_parents[np.logical_not(right_parents_in_grid[left_parents_in_grid])][
                        filter_in_grid]
                    points_left_of_left_parents_filtered = points_left_of_left_parents
                    point_values_left_of_left_parents_filtered = self.operation.get_point_values_component_grid_multiple(
                        points_left_of_left_parents_filtered,
                        component_grid)
                    m = (point_values_left_of_left_parents_filtered - point_values_left_parents_filtered) / (
                                left_parents_of_left_parents_filtered - left_parents_filtered)
                    previous_value_at_child = m * (
                                children_filtered - left_parents_filtered) + point_values_left_parents_filtered
                    point_values_filtered = point_values[filter_target_indices][filter_in_grid]
                    values[filter_target_indices_in_grid] = point_values_filtered - previous_value_at_child
                filter_target_indices2 = np.logical_and(left_parents_in_grid, right_parents_in_grid)
                values[filter_target_indices2] -= (factors_left_parents[right_parents_in_grid[left_parents_in_grid]] * point_values_left_parents[right_parents_in_grid[left_parents_in_grid]].T).T

            else:  # subtract right parent times its factor
                values[left_parents_in_grid] -= (factors_left_parents * point_values_left_parents.T).T
        if len(points_right_parents) > 0:
            point_values_right_parents = self.operation.get_point_values_component_grid_multiple(points_right_parents,
                                                                                       component_grid)

            if self.grid_surplusses.modified_basis:
                filter_target_indices = np.logical_and(right_parents_in_grid, np.logical_not(left_parents_in_grid)) # indices for this if
                right_of_right_parents_not_in_grid = np.isclose(right_parents_of_right_parents[filter_target_indices], self.b[d])
                filter_target_indices_not_in_grid = np.logical_and(np.isclose(right_parents_of_right_parents, self.b[d]), filter_target_indices)
                # we have to cases if right of right parents are in grid or not
                filter_not_in_grid = right_of_right_parents_not_in_grid
                if np.any(filter_not_in_grid):
                    point_values_right_parents_filtered = point_values_right_parents[np.logical_not(left_parents_in_grid[right_parents_in_grid])][filter_not_in_grid]
                    values[filter_target_indices_not_in_grid] = point_values[filter_target_indices][filter_not_in_grid] - point_values_right_parents_filtered
                filter_in_grid = np.logical_not(filter_not_in_grid)
                filter_target_indices_in_grid = np.logical_and(np.logical_not(np.isclose(right_parents_of_right_parents, self.b[d])), filter_target_indices)
                if np.any(filter_in_grid):
                    right_parents_of_right_parents_filtered = right_parents_of_right_parents[filter_target_indices][filter_in_grid]
                    right_parents_filtered = right_parents[filter_target_indices][filter_in_grid]
                    children_filtered = children[filter_target_indices][filter_in_grid]
                    assert len(right_parents_of_right_parents_filtered) == len(right_parents_filtered) == len(children_filtered) == 1
                    point_values_right_parents_filtered = point_values_right_parents[np.logical_not(left_parents_in_grid[right_parents_in_grid])][filter_in_grid]
                    points_right_of_right_parents_filtered = points_right_of_right_parents
                    point_values_right_of_right_parents_filtered = self.operation.get_point_values_component_grid_multiple(points_right_of_right_parents_filtered,
                                                                                           component_grid)
                    m = (point_values_right_of_right_parents_filtered - point_values_right_parents_filtered)/(right_parents_of_right_parents_filtered - right_parents_filtered)
                    previous_value_at_child = m * (children_filtered - right_parents_filtered) + point_values_right_parents_filtered
                    point_values_filtered = point_values[filter_target_indices][filter_in_grid]
                    values[filter_target_indices_in_grid] = point_values_filtered - previous_value_at_child

                filter_target_indices2 = np.logical_and(left_parents_in_grid, right_parents_in_grid)
                values[filter_target_indices2] -= (
                            factors_right_parents[left_parents_in_grid[right_parents_in_grid]] *
                            point_values_right_parents[left_parents_in_grid[right_parents_in_grid]].T).T

            else: # subtract right parent times its factor
                values[right_parents_in_grid] -= (factors_right_parents * point_values_right_parents.T).T
        widths = np.asarray([(
            self.operation.get_surplus_width(d, right_parent, left_parent)) for (left_parent, right_parent) in zip(left_parents, right_parents)]).reshape((len(children),1)) ** exponent
        if self.use_relative_surplus and not np.all(point_values == 0): # uses a relative surplus calculation with a limiter at a factor of 10**-1 of max value -> this avoids too extreme blow ups of regions close to 0
            values /= np.abs(point_values) + np.max(np.abs(point_values)) * 10**-1
        volumes = np.sum(factors * abs(values), axis=1) * widths
        if self.version == 0 or self.version == 2:
            evaluations = size_slize * len(children)  # * (1 + int(left_parent_in_grid) + int(right_parent_in_grid))
        else:
            evaluations = 0
        return abs(volumes), evaluations

    # This method calculates the surplus error estimates for a point by calculating dim-1 dimensional slices
    # through the domain along the child coordinates. We always calculate the 1-dimensional surplus for every point
    # on this slice.
    def calculate_surplusses(self, grid_points: Sequence[Sequence[float]], children_indices: Sequence[Sequence[int]], component_grid: ComponentGridInfo):
        tol = 10**-84
        if isinstance(self.grid_surplusses, GlobalBSplineGrid) or isinstance(self.grid_surplusses, GlobalHierarchicalLagrangeGrid):
            # grid_values = np.empty((self.f.output_length(), np.prod(self.grid.numPoints)))
            # points = self.grid.getPoints()
            # for i, point in enumerate(points):
            #     grid_values[:, i] = self.f(point)
            grid_values = self.operation.get_component_grid_values(component_grid, self.grid.get_coordinates())
        for d in range(0, self.dim):
            k=0
            refinement_dim = self.refinement.get_refinement_container_for_dim(d)
            if isinstance(self.grid_surplusses, GlobalBSplineGrid) or isinstance(self.grid_surplusses, GlobalHierarchicalLagrangeGrid):
                hierarchization_operator = HierarchizationLSG(self.grid)
                surplusses_1d = hierarchization_operator.hierarchize_poles_for_dim(np.array(grid_values.T), self.grid.numPoints, d)
                surplus_pole = np.zeros((self.operation.point_output_length(), self.grid.numPoints[d]))
                stride = int(np.prod(self.grid.numPoints[d+1:]))
                for j in range(self.grid.numPoints[d]):
                    i = j * stride
                    while i < np.prod(self.grid.numPoints):
                        surplus_pole[:,j] += np.sum(abs(surplusses_1d[:,i:i+stride])) #* weights[i:i+stride]))
                        i += stride * self.grid.numPoints[d]
            if not (isinstance(self.grid_surplusses, GlobalBSplineGrid) or isinstance(self.grid_surplusses, GlobalHierarchicalLagrangeGrid)) and len(children_indices[d]) > 0:
                #print(children_indices)
                volumes, evaluations = self.sum_up_volumes_for_point_completely_vectorized(child_infos=children_indices[d], grid_points=grid_points, d=d, component_grid=component_grid)
            for i, child_info in enumerate(children_indices[d]):
                left_parent = child_info.left_parent
                right_parent = child_info.right_parent
                child = child_info.child
                if isinstance(self.grid_surplusses, GlobalBSplineGrid) or isinstance(self.grid_surplusses, GlobalHierarchicalLagrangeGrid):
                    index_child = grid_points[d].index(child) - int(not(self.grid.boundary))
                    volume = surplus_pole[:, index_child] / np.prod(self.grid.numPoints) * self.grid.numPoints[d] * self.grid.weights[d][index_child]
                    evaluations = np.prod(self.grid.numPoints) / self.grid.numPoints[d]
                else:
                    volume = volumes[i]

                    #if True:#not self.grid_surplusses.modified_basis:
                    #    #volume2, evaluations = self.sum_up_volumes_for_point_vectorized(child_info=child_info, grid_points=grid_points, d=d, component_grid=component_grid)
                    #    #if volume != volume2:
                    #    #    print(volume, volume2, component_grid, child, left_parent, right_parent)
                    #    #    assert False
                    #else:
                    #    volume, evaluations = self.sum_up_volumes_for_point_vectorized(child_info=child_info, grid_points=grid_points, d=d, component_grid=component_grid)
                    assert volume is not None

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
                assert refine_obj.start >= left_parent * factor_left and refine_obj.end <= right_parent * factor_right
                max_level = 1
                while k < refinement_dim.size():
                    refine_obj = refinement_dim.get_object(k)
                    factor = 1 - tol if right_parent >= 0 else 1 + tol
                    if refine_obj.start >= right_parent * factor:
                        break
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

    def initialize_grid(self):
        # Reinitialize grid
        self.grid.initialize_grid()
