# from RefinementContainer import *
from spatiallyAdaptiveBase import *
import itertools
from scipy.interpolate import interpn


class SpatiallyAdaptiveCellScheme(SpatiallyAdaptivBase):
    def __init__(self, a, b, punish_depth=False, operation=None, norm=np.inf):
        SpatiallyAdaptivBase.__init__(self, a, b, operation=operation, norm=norm)
        # dummy container
        self.refinement = RefinementContainer([], self.dim, None)
        self.max_level = np.ones(self.dim)
        self.full_interaction_size = 1
        self.refinements_for_recalculate = 1000000
        self.cell_dict = {}
        self.margin = 0.9
        for d in range(1, self.dim+1):
            self.full_interaction_size += math.factorial(self.dim)/(math.factorial(d)*math.factorial(self.dim - d)) * 2**d
        #print("full interaction size:", self.full_interaction_size)

    # returns the points of a single component grid with refinement
    def get_points_component_grid(self, levelvec, numSubDiagonal):
        return self.operation.f.get_f_dict_points()

    def get_points_and_weights_component_grid(self, levelvec, numSubDiagonal):
        print("Not yet implemented!")
        assert(False)

    # draw a visual representation of refinement tree
    def draw_refinement(self, filename=None):
        plt.rcParams.update({'font.size': 32})
        dim = self.dim
        if dim > 2:
            print("Refinement can only be printed in 2D")
            return
        fig = plt.figure(figsize=(20,20))
        plt.tight_layout()
        ax2 = fig.add_subplot(111, aspect='equal')
        # print refinement
        for i in self.refinement.get_objects():
            startx = i.start[0]
            starty = i.start[1]
            endx = i.end[0]
            endy = i.end[1]
            ax2.add_patch(
                patches.Rectangle(
                    (startx, starty),
                    endx-startx,
                    endy-starty,
                    fill=False      # remove background
                )
            )
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()
        return fig

    def initialize_refinement(self):
        CombiScheme.dim = self.dim
        CombiScheme.lmin = self.lmin[0]
        #CombiScheme.init_adaptive_combi_scheme(self.dim, self.lmax, self.lmin)
        # define root cell that spans the domain
        self.rootCell = RefinementObjectCell(np.array(self.a), np.array(self.b), np.zeros(self.dim), self.a, self.b, self.lmin, cell_dict=self.cell_dict)
        initial_objects = [self.rootCell]
        # split root cell so that the initial cells coorespond to the grid that is generated by lmin
        for d in range(self.dim):
            for l in range(self.lmin[d]):
                print("split dimension", d)
                initial_objects = [i.split_cell_arbitrary_dim(d) for i in initial_objects]  # is now a list of lists
                # flatten list again
                initial_objects = list(itertools.chain(*initial_objects))
        # initialize the refinement with cells of lmin grid
        self.refinement = RefinementContainer(initial_objects, self.dim, self.errorEstimator)
        if self.errorEstimator is None:
            self.errorEstimator = ErrorCalculatorSurplusCell()

    def evaluate_area(self, f, area, component_grid):  # area is a cell here
        # calculates all parents of the cell for which the level vector l >= l_cell - e
        # where e is the unity vector (1, 1 , 1 , ...)

        relevant_parents_of_cell = [(area.get_key(), area.levelvec, 1)]
        subareas_fused = {}
        subareas_fused[area.get_key()] = 1
        for d in range(self.dim):
            new_parents = []
            for subarea in relevant_parents_of_cell:
                levelvec_parent = list(subarea[1])
                levelvec_parent[d] -= 1
                start_subarea = subarea[0][0]
                end_subarea = subarea[0][1]
                coefficient = subarea[2]
                parent_area = RefinementObjectCell.parent_cell_arbitrary_dim(d, list(subarea[1]), start_subarea, end_subarea, self.a, self.b, self.lmin)
                assert((parent_area is not None) or (subarea[1][d] == self.lmin[d]))
                if parent_area is not None:
                    new_parents.append((parent_area, levelvec_parent, coefficient*-1))
            relevant_parents_of_cell.extend(new_parents)
        assert(len(relevant_parents_of_cell) <= 2**self.dim)

        # calculate the integral surplus of this cell by subtracting the parent contribution for this cell area
        # for this we interpolate the corner points of this cell in the parent cell using only the parent corner values
        # -> from these interpolated points we can calculate the parent approximation for the cell and apply the
        #  combination coefficient (this is similar to the integral surplus in
        # "Dimension–Adaptive Tensor–Product Quadrature" from Gerstner and Griebel)
        integral = 0
        evaluations = 0
        #print(len(relevant_parents_of_cell))
        for (parentcell, levelvec, coefficient) in relevant_parents_of_cell:
            #print(area.get_key(), parentcell, coefficient)
            if coefficient != 0:
                sub_integral = self.integrate_subcell_with_interpolation(parentcell, area.get_key())
                self.cell_dict[area.get_key()].sub_integrals.append((sub_integral, coefficient))
                integral += sub_integral * coefficient
                #print(self.integrate_subcell_with_interpolation(area.get_key(), subcell) * coefficient)
                evaluations += 2**self.dim
        #else:
            #pass
            #print("Nothing to do in this region")
        #print("integral of cell", area.get_key(), "is:", integral)
        '''
        for p in self.grid_points:
            print(self.cell_dict)
            value = 0
            cells_with_point, levelvec = self.get_cells_to_point(p)
            level_to_cell_dict = {}
            for cell in cells_with_point:
                level_to_cell_dict[cell[1]] = cell[0]
            print(cells_with_point)
            coefficients = CombiScheme.get_coefficients_to_index_set(set([cell[1] for cell in cells_with_point]))
            print(coefficients)
            for i, coefficient in enumerate(coefficients):
                if coefficient[1] != 0:
                value += self.interpolate_point(level_to_cell_dict[coefficient[0]], p) * coefficient[1]
            print("Combined value at position:", p, "is", value, "with levelevector:", levelvec, "function value is", self.f(p))
        '''
        return integral, None, evaluations

    def evaluate_operation_area(self, component_grid, area, additional_info=None):
        relevant_parents_of_cell = [(area.get_key(), area.levelvec, 1)]
        subareas_fused = {}
        subareas_fused[area.get_key()] = 1
        for d in range(self.dim):
            new_parents = []
            for subarea in relevant_parents_of_cell:
                levelvec_parent = list(subarea[1])
                levelvec_parent[d] -= 1
                start_subarea = subarea[0][0]
                end_subarea = subarea[0][1]
                coefficient = subarea[2]
                parent_area = RefinementObjectCell.parent_cell_arbitrary_dim(d, list(subarea[1]), start_subarea, end_subarea, self.a, self.b, self.lmin)
                assert((parent_area is not None) or (subarea[1][d] == self.lmin[d]))
                if parent_area is not None:
                    new_parents.append((parent_area, levelvec_parent, coefficient*-1))
            relevant_parents_of_cell.extend(new_parents)
        assert(len(relevant_parents_of_cell) <= 2**self.dim)

        # calculate the integral surplus of this cell by subtracting the parent contribution for this cell area
        # for this we interpolate the corner points of this cell in the parent cell using only the parent corner values
        # -> from these interpolated points we can calculate the parent approximation for the cell and apply the
        #  combination coefficient (this is similar to the integral surplus in
        # "Dimension–Adaptive Tensor–Product Quadrature" from Gerstner and Griebel)
        integral = 0
        evaluations = 0
        #print(len(relevant_parents_of_cell))
        for (parent_cell_key, levelvec, coefficient) in relevant_parents_of_cell:
            if coefficient != 0:
                parent_cell = self.cell_dict[parent_cell_key]
                self.operation.compute_subcell_with_interpolation(parent_cell, area, coefficient, self.refinement)
                evaluations += 2**self.dim
        return evaluations

    '''
    def evaluate_area2(self, f, area, levelvec):  # area is a cell here
        subareas_in_cell = [(area.get_key(), area.levelvec, 1)]
        subareas_fused = {}
        subareas_fused[area.get_key()] = 1
        for d in range(self.dim):
            new_subareas = []
            for subarea in subareas_in_cell:
                levelvec_subarea = list(subarea[1])
                levelvec_subarea[d] += 1
                start_subarea = subarea[0][0]
                end_subarea = subarea[0][1]
                coefficient = subarea[2]
                new_areas = RefinementObjectCell.children_cell_arbitrary_dim(d, start_subarea, end_subarea, self.dim)
                new_subareas_refinement = []
                for area_candidate in new_areas:
                    if area_candidate in RefinementObjectCell.cell_dict:
                        new_subareas_refinement.append((area_candidate, levelvec_subarea, (coefficient*-1)))
                new_subareas.extend(new_subareas_refinement)
                #if len(new_subareas_refinement) == len(new_areas):
                #    if subarea[0] in subareas_fused:
                #        subareas_fused[subarea[0]] += coefficient * -1
                #    else:
                #        subareas_fused[subarea[0]] = coefficient * -1
                #else:
                for s in new_subareas_refinement:
                    subareas_fused[s[0]] = s[2]
            subareas_in_cell.extend(new_subareas)

        integral = 0
        evaluations = 0
        if len(subareas_in_cell) != self.full_interaction_size:
            #print(len(subareas_in_cell))
            for subcell, coefficient in subareas_fused.items():
                #print(area.get_key(), subcell, coefficient)
                if coefficient != 0:
                    sub_integral = self.integrate_subcell_with_interpolation(area.get_key(), subcell)
                    RefinementObjectCell.cell_dict[subcell].sub_integrals.append((sub_integral, coefficient))
                    integral += sub_integral * coefficient
                    #print(self.integrate_subcell_with_interpolation(area.get_key(), subcell) * coefficient)
                    evaluations += 2**self.dim
        else:
            pass
            #print("Nothing to do in this region")
        #print("integral of cell", area.get_key(), "is:", integral)
        
        for p in self.grid_points:
            print(RefinementObjectCell.cell_dict)
            value = 0
            cells_with_point, levelvec = self.get_cells_to_point(p)
            level_to_cell_dict = {}
            for cell in cells_with_point:
                level_to_cell_dict[cell[1]] = cell[0]
            print(cells_with_point)
            coefficients = CombiScheme.get_coefficients_to_index_set(set([cell[1] for cell in cells_with_point]))
            print(coefficients)
            for i, coefficient in enumerate(coefficients):
                if coefficient[1] != 0:
                value += self.interpolate_point(level_to_cell_dict[coefficient[0]], p) * coefficient[1]
            print("Combined value at position:", p, "is", value, "with levelevector:", levelvec, "function value is", self.f(p))
        
        return integral, None, evaluations
        '''
    # interpolates the cell at the subcell edge points and evaluates the integral based on the trapezoidal rule
    def integrate_subcell_with_interpolation(self, cell, subcell):
        #print("Cell and subcell", cell, subcell)
        start_subcell = subcell[0]
        end_subcell = subcell[1]
        subcell_points = list(zip(*[g.ravel() for g in np.meshgrid(*[[start_subcell[d], end_subcell[d]] for d in range(self.dim)])]))
        interpolated_values = self.interpolate_points(cell, subcell_points)
        width = np.prod(np.array(end_subcell) - np.array(start_subcell))
        factor = 0.5**self.dim * width
        integral = 0.0
        for p in interpolated_values:
            integral += p * factor
        #print("integral of subcell", subcell, "of cell", cell, "is", integral, "interpolated values", interpolated_values, "on points", subcell_points, "factor", factor)
        return integral

    # interpolates the cell corner function values at the given points
    def interpolate_points(self, cell, points):
        start = cell[0]
        end = cell[1]
        corner_points = list(zip(*[g.ravel() for g in np.meshgrid(*[[start[d], end[d]] for d in range(self.dim)])]))
        values = np.array([self.operation.f(p) if self.grid.point_not_zero(p) else 0.0 for p in corner_points])
        values = values.reshape(*[2 for d in range(self.dim)])
        values = np.transpose(values)
        corner_points_grid = [[start[d], end[d]] for d in range(self.dim)]
        #print("Corner points", corner_points_grid, np.shape(corner_points))
        #print("Values", values, np.shape(values))
        #evaluation_point = np.meshgrid(*[point[d] for d in range(self.dim)])
        interpolated_values = interpn(corner_points_grid, values, points, method='linear')
        #print("Interpolated values", interpolated_values, np.shape(values))
        return interpolated_values

    def do_refinement(self, area, position):
        if area.active:
            self.refinement.refine(position)
            area.benefit = 0
        return False

    def refinement_postprocessing(self):
        #self.refinement.apply_remove()
        self.refinement.refinement_postprocessing()
        #self.refinement.reinit_new_objects()

    def get_areas(self):
        return self.refinement.get_objects()

    def get_new_areas(self):
        #return self.refinement.get_objects()
        return self.refinement.get_new_objects()

    # returns all cells that contain the defined point (if it is on a edge it is still inside cell)
    def get_cells_to_point(self, point):
        return self.get_children_with_point(self.rootCell, point)

    #returns all children of current cell including itself that contain the defined point
    def get_children_with_point(self, cell, point):
        #print("cell:", cell.get_key(),"children:", cell.children)
        cell_list = set()
        cell_list.add(tuple((cell.get_key(), tuple(cell.levelvec))))
        if cell.is_corner(point):
            levelvec = cell.levelvec
        else:
            levelvec = None
        for child in cell.children:
            if child.contains(point):
                cell_list_new, levelvecNew = self.get_children_with_point(child, point)
                cell_list = cell_list | cell_list_new
                if levelvec is None or (levelvecNew is not None and levelvecNew <= levelvec):
                    levelvec = levelvecNew
        return cell_list, levelvec
