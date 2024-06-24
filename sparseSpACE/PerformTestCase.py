import random

import sparseSpACE
from sparseSpACE.DimAdaptiveCombi import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.GridOperation import *
import matplotlib.cm as mplcm
import matplotlib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

import csv
import matplotlib.pyplot as plt
import pylab
import numpy as np


class TestCase:
    def __init__(self, f: Function, a: Sequence[float], b: Sequence[float]):
        self.f = f
        self.a = a
        self.b = b

        self.__init_test_case("./Results/", False, False)

    def __init_test_case(self, filepath, save_csv: bool, clear_csv: bool):
        # realIntegral = scipy.integrate.dblquad(f, a, b, lambda x:a, lambda x:b, epsabs=1e-15, epsrel=1e-15)[0]
        self.reference_solution = self.f.getAnalyticSolutionIntegral(self.a, self.b)
        print("Exact integral", self.reference_solution)
        self.errorArray = []
        self.surplusErrorArray = []

        self.numEvaluationsArray = []
        self.numNaive = []
        self.numIdeal = []
        self.numFEvalIdeal = []
        self.interpolation_error_arrayL2 = []
        self.interpolation_error_arrayMax = []

        self.errorArrayDimAdaptiveArray = []
        self.numFEvalIdealDimAdaptiveArray = []

        # Create folder if it does not exist
        if not Path(filepath).is_dir():
            p = Path(filepath)
            p.mkdir(parents=True, exist_ok=True)

        self.filepath = filepath
        self.save_csv = save_csv
        self.clear_csv = clear_csv

        self.ax = None

    def test_algorithm_vectors(self, adaptive_algorithm_vectors, max_tol, dim, max_lmax, grids=None, grid_names=None,
                               min_lmin=1, max_lmin=3, min_tol=-1, max_evaluations=10 ** 7,
                               evaluation_points=None,
                               calc_standard_schemes=True, calc_dim_adaptive_schemes=False,
                               plot_interpolation_error=False,
                               legend_title="", filepath="./Results/", filename=None,
                               save_plot: bool = False, save_csv: bool = False, clear_csv: bool = False):
        # Init test case
        self.__init_test_case(filepath, save_csv, clear_csv)

        # Plot color cycle
        # https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib
        NUM_COLORS = len(adaptive_algorithm_vectors) + (max_lmin - min_lmin)

        # cm = plt.get_cmap('tab20')  # alternative: gist_rainbow, hsv, tab20
        # cNorm = colors.Normalize(vmin=0, vmax=NUM_COLORS - 1)
        # scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
        # self.ax.set_prop_cycle('color', [scalarMap.to_rgba((4 * i + 2) % NUM_COLORS) for i in range(NUM_COLORS)])

        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='w')
        self.ax = fig.add_subplot(111)

        from_list = matplotlib.colors.LinearSegmentedColormap.from_list
        cm = from_list('Set1', plt.cm.Set1(range(0, NUM_COLORS)), NUM_COLORS)
        # plt.set_cmap(cm)

        # Clear CSV
        if save_csv and clear_csv:
            self.__clear_csv(filepath, filename)

        print("\n\n\n\n\n\n ----------------------- Start {} ----------------------- \n\n\n\n\n\n".format(legend_title))

        # calculate refinements for different tolerance values
        for algorithm_vector in adaptive_algorithm_vectors:
            self.__perform_algorithm_vector_spatially_adaptive(algorithm_vector, max_tol, max_evaluations,
                                                               evaluation_points, filename)

        if calc_dim_adaptive_schemes:
            self.__perform_dim_adaptive_combi(grids, grid_names, dim, max_tol, max_evaluations, filename)

        # calculate different standard combination scheme results
        if calc_standard_schemes:
            self.__perform_standard_combi(grids, grid_names, min_lmin, max_lmin, max_lmax, dim, evaluation_points,
                                          filename, plot_interpolation_error=plot_interpolation_error)

        if calc_dim_adaptive_schemes:
            for i in range(len(self.numFEvalIdealDimAdaptiveArray)):
                name = "{}: Number of Points DimAdaptive lmin= 1".format(grid_names[i])
                print("numPoints =", self.numFEvalIdealDimAdaptiveArray[i])
                print("error=", self.errorArrayDimAdaptiveArray[i], name)
                self.ax.loglog(self.numFEvalIdealDimAdaptiveArray[i], self.errorArrayDimAdaptiveArray[i], label=name)

        line = '-'

        for i in range(len(adaptive_algorithm_vectors)):
            # print(self.numNaive[i], self.errorArray[i], adaptive_algorithm_vectors[i][4] + ' Naive evaluation')
            # print(self.numIdeal[i], self.errorArray[i], adaptive_algorithm_vectors[i][4] + ' total points')
            print(self.numFEvalIdeal[i], self.errorArray[i], adaptive_algorithm_vectors[i][4] + ' error (distinct f evals)')
            # print(self.numFEvalIdeal[i], self.surplusErrorArray[i], adaptive_algorithm_vectors[i][4] + ' surplus error distinct f evals')

            if plot_interpolation_error:
                print(self.numFEvalIdeal[i],self.interpolation_error_arrayL2[i],
                      adaptive_algorithm_vectors[i][4] + ' L2 interpolation error')
                print(self.numFEvalIdeal[i], self.interpolation_error_arrayMax[i],
                      adaptive_algorithm_vectors[i][4] + ' Linf interpolation error')

                self.ax.loglog(self.numFEvalIdeal[i], self.interpolation_error_arrayL2[i],
                               label=adaptive_algorithm_vectors[i][4] + ' L2')
                self.ax.loglog(self.numFEvalIdeal[i], self.interpolation_error_arrayMax[i],
                               label=adaptive_algorithm_vectors[i][4] + ' Linf')

            # self.ax.loglog(self.numNaive[i],self.errorArray[i],label= adaptive_algorithm_vectors[i][3] +' Naive evaluation')
            # self.ax.loglog(self.numIdeal[i],self.errorArray[i],label=adaptive_algorithm_vectors[i][3] +' total points')

            name = adaptive_algorithm_vectors[i][4]
            self.ax.loglog(self.numFEvalIdeal[i], self.errorArray[i], line, label=name + ' error (distinct f evals)')

            # self.ax.loglog(self.numFEvalIdeal[i], self.surplusErrorArray[i], '--', label=adaptive_algorithm_vectors[i][4] + ' surplus error')

        self.ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=1, mode="expand", borderaxespad=0., title=legend_title)
        self.ax.set_xlabel('Number of points')
        self.ax.set_ylabel('Approximation error')

        if save_plot:
            self.ax.figure.savefig("{}{}.pdf".format(filepath, filename), bbox_inches='tight', dpi=300)

        # self.ax.figure.show()

    def __perform_algorithm_vector_spatially_adaptive(self, algorithm_vector, max_tol, max_evaluations,
                                                      evaluation_points, filename=None):
        '''
        for i in range(min_tol, max_tol+1):
            start = time.time()
            if i == min_tol:
                coarsening, combischeme, lmax, integral, numberOfEvaluations, error_array_new, num_point_array_new = algorithm[0].performSpatiallyAdaptiv(
                    algorithm[1], algorithm[2], f, algorithm[3], 10 ** -i, reference_solution=reference_solution)
                # errorArrayAlgorithm.append(abs(integral - realIntegral) / abs(realIntegral))
                errorArrayAlgorithm.extend(error_array_new)
                numEvaluationsArrayAlgorithm.append(numberOfEvaluations)
                # numIdealAlgorithm.extend(algorithm[0].get_total_num_points_arbitrary_dim(False))
                # numNaiveAlgorithm.append(algorithm[0].get_total_num_points_arbitrary_dim(True))
                numFEvalIdealAlgorithm.extend(num_point_array_new)
            else:
                if abs(integral - reference_solution) / abs(reference_solution) > 10 ** -i:
                    coarsening, combischeme, lmax, integral, numberOfEvaluations, error_array_new, num_point_array_new = algorithm[0].performSpatiallyAdaptiv(
                        algorithm[1], algorithm[2], f, algorithm[3], 10 ** -i, coarsening, reference_solution=reference_solution)
                    #errorArrayAlgorithm.append(abs(integral - realIntegral) / abs(realIntegral))
                    errorArrayAlgorithm.extend(error_array_new)
                    numEvaluationsArrayAlgorithm.append(numberOfEvaluations)
                    #numIdealAlgorithm.extend(algorithm[0].get_total_num_points_arbitrary_dim(False))
                    #numNaiveAlgorithm.append(algorithm[0].get_total_num_points_arbitrary_dim(True))
                    numFEvalIdealAlgorithm.extend(num_point_array_new)
            end = time.time()
            print("time spent in case", i, end - start)
            '''
        errorArrayAlgorithm = []
        surplusErrorArrayAlgorithm = []
        numEvaluationsArrayAlgorithm = []
        numNaiveAlgorithm = []
        numIdealAlgorithm = []
        numFEvalIdealAlgorithm = []

        coarsening, combischeme, lmax, integral, numberOfEvaluations, error_array_new, num_point_array_new, \
        surplus_error_array_new, interpolation_errorL2, interpolation_errorMax = \
            algorithm_vector[0].performSpatiallyAdaptiv(
                algorithm_vector[1], algorithm_vector[2], algorithm_vector[3], 10 ** -max_tol,
                max_evaluations=max_evaluations, evaluation_points=evaluation_points)
        # errorArrayAlgorithm.append(abs(integral - reference_solution) / abs(reference_solution))
        errorArrayAlgorithm.extend(error_array_new)
        surplusErrorArrayAlgorithm.extend(surplus_error_array_new)

        numEvaluationsArrayAlgorithm.append(numberOfEvaluations)
        # numIdealAlgorithm.extend(algorithm[0].get_total_num_points_arbitrary_dim(False))
        # numNaiveAlgorithm.append(algorithm[0].get_total_num_points_arbitrary_dim(True))
        numFEvalIdealAlgorithm.extend(num_point_array_new)

        self.errorArray.append(errorArrayAlgorithm)
        self.surplusErrorArray.append(surplusErrorArrayAlgorithm)

        self.numEvaluationsArray.append(numEvaluationsArrayAlgorithm)
        self.numNaive.append(numNaiveAlgorithm)
        self.numIdeal.append(numIdealAlgorithm)
        self.numFEvalIdeal.append(numFEvalIdealAlgorithm)
        self.interpolation_error_arrayL2.append(interpolation_errorL2)
        self.interpolation_error_arrayMax.append(interpolation_errorMax)

        # Export Data to csv
        if self.save_csv:
            self.__export_to_csv(self.filepath, filename,
                                 [(algorithm_vector[4], numFEvalIdealAlgorithm, errorArrayAlgorithm)])

        # Spacing in console
        print("\n\n\n\n\n\n-----------------------------------------------------------------------------------------------\n\n\n\n\n\n")

    def __perform_dim_adaptive_combi(self, grids, grid_names, dim, max_tol, max_evaluations, filename=None):
        for i, grid in enumerate(grids):
            # dimAdaptiveCombi = DimAdaptiveCombi(self.a, self.b, grid)
            operation = Integration(self.f, grid, dim, self.reference_solution)
            dimAdaptiveCombi = DimAdaptiveCombi(self.a, self.b, operation=operation)
            scheme, error, result, errorArrayDimAdaptive, numFEvalIdealDimAdaptive \
                = dimAdaptiveCombi.perform_combi(1, 1, 10 ** -max_tol, max_number_of_points=max_evaluations)
            self.errorArrayDimAdaptiveArray.append(errorArrayDimAdaptive)
            self.numFEvalIdealDimAdaptiveArray.append(numFEvalIdealDimAdaptive)

            # Export Data to csv
            if self.save_csv:
                name = "{}: Number of Points DimAdaptive lmin= 1".format(grid_names[i])
                self.__export_to_csv(self.filepath, filename, [(name, numFEvalIdealDimAdaptive, errorArrayDimAdaptive)])

    def __perform_standard_combi(self, grids, grid_names, min_lmin, max_lmin, max_lmax, dim, evaluation_points,
                                 filename=None, plot_interpolation_error=False):
        for k, grid in enumerate(grids):
            xArrayStandard = []
            xFEvalArrayStandard = []
            errorArrayStandard = []
            interpolation_error_standardL2 = []
            interpolation_error_standardMax = []

            for i in range(min_lmin, max_lmin):
                xArrayStandardTest, \
                xFEvalArrayStandardTest, \
                errorArrayStandardTest, \
                interpolation_errorL2, \
                interpolation_errorMax = self.execute_standard_combi(grid, i,
                                                                     max_lmax - (i - 1) * (
                                                                             dim - 1) + i - 1,
                                                                     dim,
                                                                     self.reference_solution,
                                                                     evaluation_points=evaluation_points)

                xArrayStandard.append(xArrayStandardTest)
                xFEvalArrayStandard.append(xFEvalArrayStandardTest)
                errorArrayStandard.append(errorArrayStandardTest)
                interpolation_error_standardL2.append(interpolation_errorL2)
                interpolation_error_standardMax.append(interpolation_errorMax)

            # plot
            print(max_lmin)
            print(min_lmin)

            for i in range(max_lmin - min_lmin):
                print(xArrayStandard[i], errorArrayStandard[i], "Number of Points Standard lmin= " + str(i + min_lmin))
                print(xFEvalArrayStandard[i], errorArrayStandard[i],
                      "Distinct f evals Standard lmin= " + str(i + min_lmin))

                # self.ax.loglog(xArrayStandard[i], errorArrayStandard[i], label='standardCombination lmin=' + str(i + min_lmin))
                name = "{} (Standard Combi) lmin={}".format(grid_names[k], str(i + min_lmin))
                self.ax.loglog(xFEvalArrayStandard[i], errorArrayStandard[i], "--", label=name + " distinct f evals")

                if plot_interpolation_error:
                    print(xFEvalArrayStandard[i], interpolation_error_standardL2[i],
                          "L2 interpolation error lmin= " + str(i + min_lmin))
                    print(xFEvalArrayStandard[i], interpolation_error_standardMax[i],
                          "Linf interpolation error lmin= " + str(i + min_lmin))
                    self.ax.loglog(xFEvalArrayStandard[i], interpolation_error_standardL2[i],
                               label='standardCombination L2 lmin=' + str(i + min_lmin))
                    self.ax.loglog(xFEvalArrayStandard[i], interpolation_error_standardMax[i],
                               label='standardCombination Linf lmin=' + str(i + min_lmin))

                # Export Data to csv
                if self.save_csv:
                    self.__export_to_csv(self.filepath, filename,
                                         [(name, xFEvalArrayStandard[i], errorArrayStandard[i])])

    def __clear_csv(self, filepath, filename):
        file = "{}{}.csv".format(filepath, filename)
        print("Clearing {}".format(file))

        f = open(file, "w+")
        f.close()

    def __export_to_csv(self, filepath, filename, export_data):
        with open("{}{}.csv".format(filepath, filename), 'a') as out:
            csv_out = csv.writer(out, delimiter="|", quoting=csv.QUOTE_NONNUMERIC)

            for name, num_points, error in export_data:
                csv_out.writerow(["name", name])
                csv_out.writerow(["num_points"] + num_points)
                csv_out.writerow(["error"] + error)
                csv_out.writerow([])

    def execute_standard_combi(self, grid: Grid, lmin: int, max_lmax: int, dim: int, reference_solution: Sequence[float],
                               evaluation_points: Sequence[Tuple[float, ...]]):
        # calculate standard combination scheme results
        errorArrayStandard = []
        pointArray = []
        distinctFEvalArray = []
        operation = Integration(self.f, grid, dim, reference_solution)
        standardCombi = StandardCombi(self.a, self.b, operation=operation)
        interpolation_errorL2 = []
        interpolation_errorMax = []

        for i in range(lmin + 1, lmin + max_lmax):
            scheme, error, result = standardCombi.perform_operation(lmin, i)
            errorArrayStandard.append(error / abs(reference_solution))
            pointArray.append(standardCombi.get_total_num_points())
            distinctFEvalArray.append(standardCombi.get_total_num_points(distinct_function_evals=True))

            if evaluation_points is not None:
                interpolated_values = np.asarray(standardCombi(evaluation_points))
                real_values = np.asarray([self.f.eval(point) for point in evaluation_points])
                diff = [real_values[i] - interpolated_values[i] for i in range(len(evaluation_points))]
                # print(interpolated_values, diff)
                interpolation_errorL2.append(scipy.linalg.norm(diff, 2))
                interpolation_errorMax.append(scipy.linalg.norm(diff, np.inf))

        return pointArray, distinctFEvalArray, errorArrayStandard, interpolation_errorL2, interpolation_errorMax


class PlotTestCase:
    def __init__(self, colormap: dict = None, line_style_map: dict = None, import_filepath="../", export_filepath="./"):
        """
        Initializes the plots

        Parameters
        ----------
        colormap: dict that maps algorithm name strings to color strings, e.g.:
                    colormap = {
                        "Gauss-Legendre Grid (Standard Combi) lmin=1": "tab:gray",
                        "Trapezoidal Grid (Standard Combi) lmin=1": "violet",
                        "Trapezoidal Grid": "red",
                        "Trapezoidal Grid (Rebalancing)": "royalblue",
                    }
        line_style_map: dict that maps algorithm name strings to color strings, e.g.:
                        line_style_map = {
                            "Gauss-Legendre Grid (Standard Combi) lmin=1": "--",
                            "Trapezoidal Grid (Standard Combi) lmin=1": "--",
                            "Trapezoidal Grid": "-.",
                            "Trapezoidal Grid (Rebalancing)": "-.",
                        }
        import_filepath: From where the CSV files are imported
        export_filepath: Where the plots are written
        """
        self.colormap = colormap if colormap is not None else {}
        self.line_style_map = line_style_map if line_style_map is not None else {}

        self.import_filepath = import_filepath
        self.export_filepath = export_filepath

    def set_import_filepath(self, filepath="../"):
        self.import_filepath = filepath

    def set_export_filepath(self, filepath="./"):
        self.export_filepath = filepath

    def load_csv_data(self, import_filename):
        # Read
        import_data = []

        # if not os.path.isfile(import_filepath):
        #     raise RuntimeError("Import-File {}{}.csv does not exist.".format(import_filepath, import_filename))

        with open("{}{}.csv".format(self.import_filepath, import_filename), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONNUMERIC)

            buffer = []

            for row in csv_reader:
                if len(row) > 0:
                    row_type = row[0]
                    row_data = row[1:]

                    if row_type == "name":
                        buffer.extend(row_data)
                    else:
                        buffer.append(row_data)
                else:
                    import_data.append(buffer)
                    buffer = []

            # Append last algorithm, if buffer is not empty.
            # This is for example the case if there is no blank line at the bottom of the csv file
            if len(buffer) > 0:
                import_data.append(buffer)

        return import_data

    def plot_error_approximation(self, import_filename, algorithm_subset=None, plot_filename=None, legend_title=None):
        """
        This methods imports a csv file, generated by the PerformTestCase class,
        and plots a subset of the provided algorithms. The line colors and styles are according to the configuration.

        Parameters
        ----------
        import_filename
        algorithm_subset: a list of algorithm names that should vbe plotted
        plot_filename
        legend_title: if this argument is None, no legend will be plotted

        Returns
        -------

        """
        print("Processing {}.csv ...".format(import_filename))

        import_data = self.load_csv_data(import_filename)

        # Plot
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='w')
        ax = fig.add_subplot(111)

        for name, num_points, error in import_data:
            if algorithm_subset is not None and name not in algorithm_subset:
                continue

            # Add entries to style maps for plots
            if name not in self.colormap:
                backup_colors = [
                    'black', 'gray', 'silver', 'indianred', 'red', 'tomato', 'chocolate', 'saddlebrown', 'darkorange', 'goldenrod', 'darkkhaki', 'darkolivegreen', 'lightgreen',
                    'forestgreen', 'lime', 'turquoise', 'teal', 'cyan', 'steelblue', 'darkslategray', 'slategray', 'royalblue', 'navy', 'indigo', 'darkviolet', 'violet', 'deeppink', 'crimson', 'lightpink'
                ]
                self.colormap[name] = random.choice(backup_colors)

            if name not in self.line_style_map:
                backup_line_styles = [
                    '-', '.', '--', '-.'
                ]
                self.line_style_map[name] = random.choice(backup_line_styles)

            # Plot
            ax.loglog(num_points, error, self.line_style_map[name], color=self.colormap[name], label=name)

        if legend_title is not None:
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=1, mode="expand", borderaxespad=0., title=legend_title)

        ax.set_xlabel('Number of points')
        ax.set_ylabel('Approximation error')

        if plot_filename is not None:
            ax.figure.savefig("{}{}.pdf".format(self.export_filepath, plot_filename), bbox_inches='tight', dpi=300)

    def export_legend(self, algorithms, filename="legend"):
        """
        Export legend separately from plots.
        https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib

        Parameters
        ----------
        algorithms: a list of algorithms names, that should appear in the legend
        filename
        export_filepath

        Returns
        -------

        """
        figData = pylab.figure()
        ax = pylab.gca()
        x = np.arange(10)

        for i, algo in enumerate(algorithms):
            pylab.plot(x, x * (i + 1), self.line_style_map[algo], color=self.colormap[algo], label=algo)

        figlegend = pylab.figure()
        figlegend.legend(*ax.get_legend_handles_labels(), loc="center", mode="expand", prop={'size': 9})
        figlegend.show()
        figlegend.savefig("{}/{}.pdf".format(self.export_filepath, filename))