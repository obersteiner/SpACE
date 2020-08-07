import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
import time
import random as rnd
from StandardCombi import StandardCombi
from GridOperation import DensityEstimation
from sklearn import datasets, preprocessing, neighbors
from sklearn.utils import shuffle
from typing import List, Tuple, Union, Iterable

from ErrorCalculator import ErrorCalculatorSingleDimVolumeGuided
from Grid import GlobalTrapezoidalGrid
from spatiallyAdaptiveSingleDimension2 import SpatiallyAdaptiveSingleDimensions2


class DataSet:
    """Type of datasets on which to perform DensityEstimation, Classification and Clustering.

    All DataSets have data in form of a tuple of length 2 with one ndarray each:
    The samples in arbitrary dimension, then the corresponding labels in dimension 1.
    Unknown labels are labelled with -1.
    """

    def __init__(self, raw_data: Union[Tuple[np.ndarray, ...], np.ndarray, str], name: str = 'unknown', label: str = 'class'):
        """Constructor of DataSet.

        Takes raw data and optionally a name as parameter and initializes raw data to the form of a tuple with length 2.
        Scaling attributes are unassigned until scaling occurs.

        :param raw_data: Samples (and corresponding labels) of this DataSet. Can be a tuple of samples and labels, only labelless samples, CSV file.
        :param name: Optional. Name of this DataSet
        """
        self.__name = name
        self.__label = label
        self.__data = None
        self.__dim = None
        self.__shape = None
        self.__shuffled = False
        self.__scaled = False
        self.__scaling_range = None
        self.__scaling_factor = None
        self.__original_min = None
        self.__original_max = None
        self.__initialize(raw_data)
        assert ((self.__data is not None) and (self.__dim is not None) and (self.__shape is not None))
        assert (isinstance(self.__data, tuple) and len(self.__data) == 2 and
                isinstance(self.__data[0], np.ndarray) and isinstance(self.__data[1], np.ndarray))

    def __getitem__(self, item: int) -> np.ndarray:
        return self.__data[item]

    def __str__(self) -> str:
        return str(self.__data)

    def copy(self) -> 'DataSet':
        copied = DataSet(self.__data)
        copied.__dict__.update(self.__dict__)
        copied.set_name("%s_copy" % self.get_name())
        return copied

    def set_name(self, name: str) -> None:
        self.__name = name

    def set_label(self, label: str) -> None:
        self.__label = label

    def get_name(self) -> str:
        return self.__name

    def get_label(self) -> str:
        return self.__label

    def get_data(self) -> Tuple[np.ndarray, ...]:
        return self.__data

    def get_min_data(self) -> Union[np.ndarray, None]:
        if not self.is_empty():
            return np.amin(self.__data[0], axis=0)
        else:
            return None

    def get_max_data(self) -> Union[np.ndarray, None]:
        if not self.is_empty():
            return np.amax(self.__data[0], axis=0)
        else:
            return None

    def get_original_min(self) -> Union[np.ndarray, None]:
        return self.__original_min

    def get_original_max(self) -> Union[np.ndarray, None]:
        return self.__original_max

    def get_length(self) -> int:
        length = round(self.__data[0].size / self.__dim) if (self.__dim != 0) else 0
        assert ((length * self.__dim) == self.__data[0].size)
        return length

    def get_dim(self) -> int:
        return self.__dim

    def get_number_labels(self) -> int:
        return len([x for x in set(self.__data[1]) if x >= 0])

    def get_labels(self) -> List[int]:
        return list(set(self.__data[1]))

    def has_labelless_samples(self) -> bool:
        return -1 in self.__data[1]

    def is_empty(self) -> bool:
        return self.__data[0].size == 0

    def is_shuffled(self) -> bool:
        return self.__shuffled

    def is_scaled(self) -> bool:
        return self.__scaled

    def get_scaling_range(self) -> Tuple[float, float]:
        return self.__scaling_range

    def get_scaling_factor(self) -> float:
        return self.__scaling_factor

    def __initialize(self, raw_data: Union[Tuple[np.ndarray, np.ndarray], np.ndarray, str]) -> None:
        """Private initialization method for DataSet.

        Provides several checks of the input parameter raw_data of the constructor and raises an error if raw_data can't be converted to
        appropriate form.

        :param raw_data: Samples (and corresponding labels) of this DataSet. Can be a tuple of samples and labels, only labelless samples, CSV file.
        :return: None
        """
        if isinstance(raw_data, str):
            # raw_data = read_csv_file()
            pass  # TODO implement DataSet csv reader
        if isinstance(raw_data, np.ndarray):
            if raw_data.size == 0:
                self.__dim = 0
                self.__shape = 0
                raw_data = np.reshape(raw_data, 0)
            else:
                self.__dim = round(raw_data.size / len(raw_data))
                self.__shape = (len(raw_data), self.__dim)
                assert ((len(raw_data) * self.__dim) == raw_data.size)
                raw_data = np.reshape(raw_data, self.__shape)
            self.__data = raw_data, np.array(([-1] * len(raw_data)), dtype=np.int64)
        elif isinstance(raw_data, tuple) and (len(raw_data) == 2):
            if raw_data[0].size == 0:
                self.__dim = 0
                self.__shape = 0
                self.__data = tuple([np.reshape(raw_data[0], 0), raw_data[1]])
            elif raw_data[1].ndim == 1 and not any([x < -1 for x in list(set(raw_data[1]))]) and (len(raw_data[0]) == len(raw_data[1])):
                self.__dim = round(raw_data[0].size / len(raw_data[0]))
                self.__shape = (len(raw_data[0]), self.__dim)
                assert ((len(raw_data[0]) * self.__dim) == raw_data[0].size)
                self.__data = tuple([np.reshape(raw_data[0], self.__shape), raw_data[1]])
            else:
                raise ValueError("Invalid raw_data parameter in DataSet Constructor.")
        else:
            raise ValueError("Invalid raw_data parameter in DataSet Constructor.")

    def __update_internal(self, to_update: 'DataSet') -> 'DataSet':
        """Update all internal attributes which can normally only be changed through DataSet methods or should be changed automatically.

        Mainly used to keep scaling of DataSets after methods that change the internal data.

        :param to_update: DataSet whose internal attributes need to updated
        :return: Input DataSet with updated internal attributes
        """
        to_update.__label = self.__label
        to_update.__shuffled = self.__shuffled
        to_update.__scaled = self.__scaled
        to_update.__scaling_range = self.__scaling_range
        to_update.__scaling_factor = self.__scaling_factor
        if self.__scaled:
            to_update.__original_min = self.__original_min.copy()
            to_update.__original_max = self.__original_max.copy()
        else:
            to_update.__original_min = self.__original_min
            to_update.__original_max = self.__original_max
        return to_update

    def same_scaling(self, to_check: 'DataSet') -> bool:
        """Check whether self and to_check have the same scaling.

        Compares the scaling range and factor of self and to_check and returns False if anything doesn't match.

        :param to_check: DataSet whose internal scaling should be compared to self's internal scaling
        :return: Boolean value which indicates whether the internal scaling of input DataSet and self are completely equal
        """
        if not self.__scaled == to_check.__scaled:
            return False
        if not self.__scaled and not to_check.__scaled:
            return True
        assert (self.__scaled == to_check.__scaled)
        if not (isinstance(self.__scaling_range[0], Iterable) != isinstance(to_check.__scaling_range[0], Iterable)):
            if isinstance(self.__scaling_range[0], Iterable):
                scaling_range = all([(x[0] == y[0]) and (x[1] == y[1]) for x, y in zip(self.__scaling_range, to_check.__scaling_range)])
            else:
                scaling_range = all([x == y for x, y in zip(self.__scaling_range, to_check.__scaling_range)])
        else:
            return False
        if not (isinstance(self.__scaling_factor, Iterable) != isinstance(to_check.__scaling_factor, Iterable)):
            if isinstance(self.__scaling_factor, Iterable):
                scaling_factor = all([x == y for x, y in zip(self.__scaling_factor, to_check.__scaling_factor)])
            else:
                scaling_factor = self.__scaling_factor == to_check.__scaling_factor
        else:
            return False
        return scaling_range and scaling_factor

    def remove_samples(self, indices: List[int]) -> 'DataSet':
        """Remove samples of DataSet at specified indices.

        If the list of indices is empty, no samples are removed.

        :param indices: List of indices at which to remove samples
        :return: New DataSet in which samples are removed
        """
        if any([(i < 0) or (i > self.get_length()) for i in indices]):
            raise ValueError("Can't remove samples out of bounds of DataSet.")
        removed_samples = [self.__update_internal(DataSet((np.array([self.__data[0][i]]), np.array([self.__data[1][i]])))) for i in indices]
        self.__data = tuple([np.delete(self.__data[0], indices, axis=0), np.delete(self.__data[1], indices, axis=0)])
        return DataSet.list_concatenate(removed_samples)

    def scale_range(self, scaling_range: Tuple[float, float], override_scaling: bool = False) -> None:
        """Scale DataSet to a specified range.

        If override_scaling is set, current scaling (if available) is turned into the original scaling of this DataSet and the new scaling
        specified by the input parameter is applied.

        :param scaling_range: Range to which all samples should be scaled
        :param override_scaling: Optional. Conditional parameter which indicates whether old scaling (if available) should be overridden
        :return: None
        """
        if not self.__scaled or override_scaling:
            scaler = preprocessing.MinMaxScaler(feature_range=scaling_range)
            scaler.fit(self.__data[0])
            self.__data = tuple([scaler.transform(self.__data[0]), np.array([c for c in self.__data[1]])])
            self.__scaled = True
            self.__scaling_range = scaling_range
            self.__scaling_factor = scaler.scale_
            self.__original_min = scaler.data_min_
            self.__original_max = scaler.data_max_
        else:
            scaler = preprocessing.MinMaxScaler(feature_range=scaling_range)
            scaler.fit(self.__data[0])
            self.__data = tuple([scaler.transform(self.__data[0]), np.array([c for c in self.__data[1]])])
            self.__scaling_range = scaling_range
            self.__scaling_factor *= scaler.scale_

    def scale_factor(self, scaling_factor: Union[float, np.ndarray], override_scaling: bool = False) -> None:
        """Scale DataSet by a specified factor.

        If override_scaling is set, current scaling (if available) is turned into the original scaling of this DataSet and the new scaling
        specified by the input parameter is applied.

        :param scaling_factor: Factor by which all samples should be scaled. Can either be a float value for general scaling or np.ndarray with
        dimension self.__dim to scale each dimension individually
        :param override_scaling: Optional. Conditional parameter which indicates whether old scaling (if available) should be overridden
        :return: None
        """
        if not self.__scaled or override_scaling:
            self.__original_min = self.get_min_data()
            self.__original_max = self.get_max_data()
            self.__data = tuple([np.array(list(map(lambda x: x * scaling_factor, self.__data[0]))), self.__data[1]])
            self.__scaling_range = (np.amin(self.__data[0], axis=0), np.amax(self.__data[0], axis=0))
            self.__scaling_factor = scaling_factor
            self.__scaled = True
        else:
            if isinstance(scaling_factor, np.ndarray) and len(scaling_factor) != self.__dim:
                raise ValueError("Multidimensional scaling factor needs to have the same dimension as DataSet it is applied to.")
            self.__data = tuple([np.array(list(map(lambda x: x * scaling_factor, self.__data[0]))), self.__data[1]])
            self.__scaling_range = (np.amin(self.__data[0], axis=0), np.amax(self.__data[0], axis=0))
            self.__scaling_factor *= scaling_factor

    def shift_value(self, shift_val: Union[float, np.ndarray], override_scaling: bool = False) -> None:
        """Shift DataSet by a specified value.

        If override_scaling is set, current scaling (if available) is turned into the original scaling of this DataSet and the new scaling
        specified by the input parameter is applied.

        :param shift_val: Value by which all samples should be shifted. Can either be a float value for general shifting or np.ndarray with
        dimension self.__dim to shift each dimension individually
        :param override_scaling: Optional. Conditional parameter which indicates whether old scaling (if available) should be overridden
        :return: None
        """
        if not self.__scaled or override_scaling:
            self.__original_min = self.get_min_data()
            self.__original_max = self.get_max_data()
            self.__data = tuple([np.array(list(map(lambda x: (x + shift_val), self.__data[0]))), self.__data[1]])
            self.__scaling_range = (np.amin(self.__data[0], axis=0), np.amax(self.__data[0], axis=0))
            self.__scaling_factor = 1.0
            self.__scaled = True
        else:
            if isinstance(shift_val, np.ndarray) and len(shift_val) != self.__dim:
                raise ValueError("Multidimensional shifting value needs to have the same dimension as DataSet it is applied to.")
            self.__data = tuple([np.array(list(map(lambda x: (x + shift_val), self.__data[0]))), self.__data[1]])
            self.__scaling_range = (np.amin(self.__data[0], axis=0), np.amax(self.__data[0], axis=0))

    def shuffle(self) -> None:
        """Shuffle DataSet randomly.

        :return: None
        """
        shuffled = shuffle(tuple(zip(self.__data[0], self.__data[1])))
        self.__data = tuple([np.array([[v for v in x[0]] for x in shuffled]), np.array([y[1] for y in shuffled])])
        self.__shuffled = True

    def concatenate(self, other_dataset: 'DataSet') -> 'DataSet':
        """Concatenate this DataSet's data with the data of specified DataSet.

        If either this or the specified DataSet are empty, the other one is returned.
        Only data of DataSets with equal dimension can be concatenated.

        :param other_dataset: DataSet whose data should be concatenated with data of this DataSet
        :return: New DataSet with concatenated data
        """
        if not (self.__dim == other_dataset.get_dim()):
            if other_dataset.is_empty():
                return self
            elif self.is_empty():
                return other_dataset
            else:
                raise ValueError("DataSets must have the same dimensions for concatenation.")
        values = np.concatenate((self.__data[0], other_dataset[0]), axis=0)
        labels = np.concatenate((self.__data[1], other_dataset[1]))
        concatenated_set = DataSet((values, labels))
        self.__update_internal(concatenated_set)
        equal_scaling = self.same_scaling(concatenated_set)
        if not equal_scaling:
            raise ValueError("Can't concatenate DataSets with different scaling")
        return concatenated_set

    @staticmethod
    def list_concatenate(list_datasets: List['DataSet']) -> 'DataSet':
        """Concatenate a list of DataSets to a single DataSet.

        If an empty list is received as a parameter, an empty DataSet is returned.

        :param list_datasets: List of DataSet's which to concatenate.
        :return: New DataSet which contains the concatenated data of all DataSets within the list
        """
        if len(list_datasets) == 0:
            return DataSet(tuple([np.array([]), np.array([])]))
        dataset = list_datasets[0]
        for i in range(1, len(list_datasets)):
            dataset = dataset.concatenate(list_datasets[i])
        return dataset

    def split_labels(self) -> List['DataSet']:
        """Split samples with the same labels into their own DataSets.

        Creates a DataSet for each label and puts all samples with corresponding label into this DataSet.
        Stores all of those single-label-DataSets into a list.

        :return: A List which contains all single-label-DataSets
        """
        set_labels = []
        for j in self.get_labels():
            current_values = np.array([x for i, x in enumerate(self.__data[0]) if self.__data[1][i] == j])
            current_label = np.array(([j] * len(current_values)), dtype=np.int64)
            current_set = DataSet(tuple([current_values, current_label]))
            self.__update_internal(current_set)
            set_labels.append(current_set)
        return set_labels

    def split_without_labels(self) -> Tuple['DataSet', 'DataSet']:
        """Separates samples without labels from samples with labels.

        Creates a DataSet for samples with and without labels each.
        Samples are stored in the respective DataSet.
        If there are no labelless samples and/ or samples with labels, the respective DataSet stays empty.

        :return: A Tuple of two new DataSets which contain all labelless samples and all samples with labels
        """
        labelless_values = np.array([x for i, x in enumerate(self.__data[0]) if self.__data[1][i] == -1])
        labelfull_values = np.array([x for i, x in enumerate(self.__data[0]) if self.__data[1][i] >= 0])
        set_labelless = DataSet(labelless_values)
        set_labelfull = DataSet(tuple([labelfull_values, np.array([c for c in self.__data[1] if c >= 0], dtype=np.int64)]))
        self.__update_internal(set_labelless)
        self.__update_internal(set_labelfull)
        return set_labelless, set_labelfull

    def split_pieces(self, percentage: float) -> Tuple['DataSet', 'DataSet']:
        """Splits this DataSet's data into two uneven pieces.

        The first split piece contains all samples until index (percentage * this data's length) rounded down to the next integer.
        The second split piece contains all other samples.
        Before the splitting is performed, percentage is checked if in range (0, 1) and if not is set to 1.

        :param percentage: Percentage of this DataSet's data at whose last index the split occurs
        :return: A Tuple of two new DataSets which contain all samples before and after the index at which the data was splitted
        """
        percentage = percentage if 0 <= percentage < 1 else 1.0
        set0 = DataSet(tuple([np.array(self.__data[0][:(round(self.get_length() * percentage))]),
                              self.__data[1][:(round(self.get_length() * percentage))]]))
        set1 = DataSet(tuple([np.array(self.__data[0][(round(self.get_length() * percentage)):]),
                              self.__data[1][(round(self.get_length() * percentage)):]]))
        self.__update_internal(set0)
        self.__update_internal(set1)
        return set0, set1

    def remove_labels(self, percentage: float) -> None:
        """Removes the labels of percentage samples randomly.

        Before removal of labels, percentage is checked if in range (0, 1) and if not is set to 1.
        Mainly used for testing purposes.

        :param percentage: Percentage of random indices at which to remove labels
        :return: None
        """
        labelless, labelfull = self.split_without_labels()
        indices = rnd.sample(range(0, labelfull.get_length()), round((percentage if (1 > percentage >= 0) else 1.0) * labelfull.get_length()))
        labels = labelfull.__data[1]
        labels[indices] = -1
        if labelless.is_empty():
            self.__data = tuple([labelfull.__data[0], labels])
        elif labelfull.is_empty():
            self.__data = tuple([labelless.__data[0], labelless.__data[1]])
        else:
            self.__data = tuple([np.concatenate((labelfull.__data[0], labelless.__data[0])), np.concatenate((labels, labelless.__data[1]))])

    def move_boundaries_to_front(self) -> None:
        """Move samples with lowest and highest value in each dimension to the front of this DataSet's data.

        Mainly used for initialization in Classification to guarantee the boundary samples being in the learning DataSet.

        :return: None
        """
        search_indices_min = np.where(self.__data[0] == self.get_min_data())
        search_indices_max = np.where(self.__data[0] == self.get_max_data())
        indices = list(set(search_indices_min[0]) | set(search_indices_max[0]))
        for i, x in enumerate(indices):
            self.__data[0][[i, x]] = self.__data[0][[x, i]]
            self.__data[1][[i, x]] = self.__data[1][[x, i]]

    def revert_scaling(self) -> None:
        """Revert this DataSet's data to its original scaling.

        Scaling is applied in reverse to this DataSet's data and scaling attributes are returned to their initial setting.

        :return:
        """
        self.shift_value(-self.__scaling_range[0], override_scaling=False)
        self.scale_factor(1.0 / self.__scaling_factor, override_scaling=False)
        self.shift_value(self.__original_min, override_scaling=False)
        self.__scaled = False
        self.__scaling_range = None
        self.__scaling_factor = None
        self.__original_min = None
        self.__original_max = None

    def density_estimation(self,
                           masslumping: bool = True,
                           lambd: float = 0.0,
                           minimum_level: int = 1,
                           maximum_level: int = 5,
                           plot_de_dataset: bool = True,
                           plot_density_estimation: bool = True,
                           plot_combi_scheme: bool = True,
                           plot_sparsegrid: bool = True) -> Tuple[StandardCombi, DensityEstimation]:
        """Perform the GridOperation DensityEstimation on this DataSet.

        Also is able to plot the DensityEstimation results directly.
        For more information on DensityEstimation refer to the class DensityEstimation in the GridOperation module.

        :param masslumping: Optional. Conditional Parameter which indicates whether masslumping should be enabled for DensityEstimation
        :param lambd: Optional. Parameter which adjusts the 'smoothness' of DensityEstimation results
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation
        :param plot_de_dataset: Optional. Conditional Parameter which indicates whether this DataSet should be plotted for DensityEstimation
        :param plot_density_estimation: Optional. Conditional Parameter which indicates whether results of DensityEstimation should be plotted
        :param plot_combi_scheme: Optional. Conditional Parameter which indicates whether resulting combi scheme of DensityEstimation should be
        plotted
        :param plot_sparsegrid: Optional. Conditional Parameter which indicates whether resulting sparsegrid of DensityEstimation should be plotted
        :return:
        """
        a = np.zeros(self.__dim)
        b = np.ones(self.__dim)
        de_object = DensityEstimation(self.__data, self.__dim, masslumping=masslumping, lambd=lambd)
        combi_object = StandardCombi(a, b, operation=de_object, print_output=False)
        combi_object.perform_operation(minimum_level, maximum_level, print_time=False)
        if plot_de_dataset:
            if de_object.scaled:
                self.scale_range((0, 1), override_scaling=True)
            self.plot(plot_labels=False)
        if plot_density_estimation:
            combi_object.plot(contour=True)
        if plot_combi_scheme:
            combi_object.print_resulting_combi_scheme(operation=de_object)
        if plot_sparsegrid:
            combi_object.print_resulting_sparsegrid(markersize=20)
        return combi_object, de_object

    def density_estimation_dimension_wise(self,
                                          masslumping: bool = True,
                                          lambd: float = 0.0,
                                          minimum_level: int = 1,
                                          maximum_level: int = 5,
                                          reuse_old_values: bool = False,
                                          numeric_calculation: bool = True,
                                          margin: float = 0.5,
                                          tolerance: float = 0.01,
                                          max_evaluations: int = 256,
                                          modified_basis: bool = False,
                                          boundary: bool = False,
                                          plot_de_dataset: bool = True,
                                          plot_density_estimation: bool = True,
                                          plot_combi_scheme: bool = True,
                                          plot_sparsegrid: bool = True) -> Tuple[StandardCombi, DensityEstimation]:
        """Perform the GridOperation DensityEstimation on this DataSet.

        Also is able to plot the DensityEstimation results directly.
        For more information on DensityEstimation refer to the class DensityEstimation in the GridOperation module.

        :param masslumping: Optional. Conditional Parameter which indicates whether masslumping should be enabled for DensityEstimation
        :param lambd: Optional. Parameter which adjusts the 'smoothness' of DensityEstimation results
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation
        :param reuse_old_values: Optional.
        :param numeric_calculation: Optional.
        :param margin: Optional.
        :param tolerance: Optional.
        :param max_evaluations: Optional.
        :param modified_basis: Optional.
        :param boundary: Optional.
        :param plot_de_dataset: Optional. Conditional Parameter which indicates whether this DataSet should be plotted for DensityEstimation
        :param plot_density_estimation: Optional. Conditional Parameter which indicates whether results of DensityEstimation should be plotted
        :param plot_combi_scheme: Optional. Conditional Parameter which indicates whether resulting combi scheme of DensityEstimation should be
        plotted
        :param plot_sparsegrid: Optional. Conditional Parameter which indicates whether resulting sparsegrid of DensityEstimation should be plotted
        :return:

        Parameters
        ----------
        """
        a = np.zeros(self.__dim)
        b = np.ones(self.__dim)
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary)
        de_object = DensityEstimation(self.__data, self.__dim, grid=grid, masslumping=masslumping, lambd=lambd, reuse_old_values=reuse_old_values,
                                      numeric_calculation=numeric_calculation, print_output=False)
        error_calculator = ErrorCalculatorSingleDimVolumeGuided()
        combi_object = SpatiallyAdaptiveSingleDimensions2(a, b, operation=de_object, margin=margin, rebalancing=False)
        combi_object.performSpatiallyAdaptiv(minimum_level, maximum_level, error_calculator, tolerance, max_evaluations=max_evaluations,
                                             do_plot=plot_combi_scheme)
        if plot_de_dataset:
            if de_object.scaled:
                self.scale_range((0, 1), override_scaling=True)
            self.plot(plot_labels=False)
        if plot_density_estimation:
            combi_object.plot(contour=True)
        if plot_combi_scheme:
            combi_object.print_resulting_combi_scheme(operation=de_object)
        if plot_sparsegrid:
            combi_object.print_resulting_sparsegrid(markersize=20)
        return combi_object, de_object

    def plot(self, plot_labels: bool = True, plot_directly: bool = True) -> plt.Figure:
        """Plot DataSet.

        Plotting is only available for dimensions 2 and 3.

        :param plot_directly: Optional. Conditional parameter which indicates whether this function should actually plot the data.
        :param plot_labels: Optional. Conditional parameter which indicates whether labels should be coloured for plotting.
        :return: Figure which is plotted
        """
        plt.rc('font', size=30)
        plt.rc('axes', titlesize=40)
        plt.rc('figure', figsize=(12.0, 12.0))
        fig = plt.figure()

        if self.__dim == 2:
            ax = fig.add_subplot(111)
            if plot_labels:
                if self.has_labelless_samples():
                    data_labelless, data_labelfull = self.split_without_labels()
                    list_labels = data_labelfull.split_labels()
                    x, y = zip(*data_labelless[0])
                    ax.scatter(x, y, s=125, label='%s_?' % self.__label, c='gray')
                else:
                    list_labels = self.split_labels()
                for i, v in enumerate(list_labels):
                    x, y = zip(*v[0])
                    ax.scatter(x, y, s=125, label='%s_%d' % (self.__label, i))
                ax.legend(fontsize=22, loc='upper left', borderaxespad=0.0, bbox_to_anchor=(1.05, 1))
            else:
                x, y = zip(*self.__data[0])
                ax.scatter(x, y, s=125)
            ax.set_title(self.__name)
            ax.title.set_position([0.5, 1.025])
            ax.grid(True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif self.__dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            if plot_labels:
                if self.has_labelless_samples():
                    data_labelless, data_labelfull = self.split_without_labels()
                    list_labels = data_labelfull.split_labels()
                    x, y, z = zip(*data_labelless[0])
                    ax.scatter(x, y, z, s=125, label='%s_?' % self.__label, c='gray')
                else:
                    list_labels = self.split_labels()
                for i, v in enumerate(list_labels):
                    x, y, z = zip(*v[0])
                    ax.scatter(x, y, z, s=125, label='%s_%d' % (self.__label, i))
                ax.legend(fontsize=22, loc='upper left', borderaxespad=0.0, bbox_to_anchor=(1.05, 1))
            else:
                fig.set_figwidth(10.0)
                x, y, z = zip(*self.__data[0])
                ax.scatter(x, y, z, s=125)
            ax.set_title(self.__name)
            ax.title.set_position([0.5, 1.025])
            ax.grid(True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            warnings.formatwarning = lambda msg, ctg, fname, lineno, file=None, line=None: "%s:%s: %s: %s\n" % (fname, lineno, ctg.__name__, msg)
            warnings.warn("Invalid dimension for plotting. Couldn't plot DataSet.", stacklevel=3)

        if plot_directly:
            plt.show()
        return fig

    def write_csv(self) -> None:
        """Write this DataSet to a CSV file.

        :return: None
        """
        pass  # TODO implement DataSet csv writer


class Classification:
    """Type of objects that classify data based on some learning dataset.
    """

    def __init__(self,
                 raw_data: 'DataSet',
                 data_range: Tuple[np.ndarray, np.ndarray] = None,
                 split_percentage: float = 1.0,
                 split_evenly: bool = True):
        """Constructor of Classification.

        Takes raw_data as necessary parameter and some more optional parameters which are specified below.
        Initializes data for this object.
        Stores the following values as private attributes:
        + self.__data: original data, all new testing data is stored here as well
        + self.__omitted_data: during initialization and testing omitted classless data
        + self.__learning_data: in __initialize() assigned learning data
        + self.__testing_data: in __initialize() assigned testing data, all new testing data is stored here as well
        + self.__data_range: the original range of all data samples in each dimension
        + self.__scale_factor: the scaling factor for each dimension with which all samples in this object were scaled
        + self.__calculated_classes_testset: All calculated classes for samples in self.__testing_data in the same order (useful for printing)
        + self.__classificators: List of all in __perform_classification() computed classificators. One for each class
        + self.__de_objects: List of all in in __perform_classification() computed DensityEstimation Objects (useful for plotting)
        + self.__performed_classification: Check if classfication was already performed for this object

        :param raw_data: DataSet on which to perform learning (and if 0 < percentage < 1 also testing)
        :param data_range: Optional. If the user knows the original range of the dataset, they can specify it here.
        :param split_percentage: Optional. If a percentage of the raw data should be used as testing data: 0 < percentage < 1. Default 1.0
        :param split_evenly: Optional. Only relevant when 0 < percentage < 1. Conditional parameter which indicates whether the learning datasets
        for each class should be of near equal size. Default True
        """
        self.__original_data = raw_data
        self.__scaled_data = None
        self.__omitted_data = None
        self.__learning_data = None
        self.__testing_data = None
        self.__data_range = data_range
        self.__scale_factor = None
        self.__calculated_classes_testset = np.array([])
        self.__densities_testset = []
        self.__classificators = []
        self.__de_objects = []
        self.__performed_classification = False
        self.__initialize((split_percentage if isinstance(split_percentage, float) and (1 > split_percentage > 0) else 1.0), split_evenly)

    def __call__(self, data_to_evaluate: 'DataSet', print_removed: bool = True) -> 'DataSet':
        """Evaluate classes for samples in input data and create a new DataSet from those same samples and classes.

        When this method is called, classification needs to already be performed. If it is not, an AttributeError is raised.
        Also the input data mustn't be empty. If it is, a ValueError is raised.
        Before classification, input data is scaled to match the scaling of learning data of this object; samples out of bounds after scaling are
        removed and by default printed to stdout.

        :param data_to_evaluate: Data whose samples are to be classified
        :param print_removed: Optional. Conditional parameter which specifies whether during scaling removed samples should be printed
        :return: New DataSet which consists of samples from input DataSet and for those samples computed classes
        """
        if not self.__performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if data_to_evaluate.is_empty():
            raise ValueError("Can't classificate empty dataset.")
        print("---------------------------------------------------------------------------------------------------------------------------------")
        print("Evaluating classes of %s DataSet..." % data_to_evaluate.get_name())
        evaluate = self.__internal_scaling(data_to_evaluate, print_removed=print_removed)
        if evaluate.is_empty():
            raise ValueError("All given samples for classification were out of bounds. Please only evaluate classes for samples in unscaled range: "
                             "\n[%s]\n[%s]\nwith this classification object" %
                             (', '.join([str(x) for x in self.__data_range[0]]), ', '.join([str(x) for x in self.__data_range[1]])))
        evaluated_data = DataSet(tuple([evaluate[0], np.array(self.__classificate(evaluate))]), name="%s_evaluated_classes" %
                                                                                                     data_to_evaluate.get_name(), label="class")
        del self.__densities_testset[(len(self.__densities_testset) - data_to_evaluate.get_length()):]
        return evaluated_data

    def test_data(self, new_testing_data: DataSet,
                  print_output: bool = True,
                  print_removed: bool = True,
                  print_incorrect_points: bool = False) -> DataSet:
        """Test new data with the classificators of a Classification object.

        As most of other public methods of Classification, classification already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        In case the input testing data is empty, a ValueError is raised.
        Test data is scaled with the same factors as the original data (self.__data) and samples out of bounds after scaling are removed.
        Only test data samples with known classes can be used for testing; the omitted rest ist stored into self.__omitted_data.
        Test data with known classes and samples only inside of bounds is stored into self.__testing_data, results are calculated and printed
        (default) if the user specified it.

        :param new_testing_data: Test DataSet for which classificators should be tested
        :param print_output: Optional. Conditional parameter which specifies whether results of testing should be printed. Default True
        :param print_removed: Optional. Conditional parameter which specifies whether during scaling removed samples should be printed. Default True
        :param print_incorrect_points: Conditional parameter whick specifies whether the incorrectly mapped points should be printed. Default False
        :return: DataSet which contains all classless samples that were omitted
        """
        if not self.__performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if new_testing_data.is_empty():
            raise ValueError("Can't test empty dataset.")
        print("---------------------------------------------------------------------------------------------------------------------------------")
        print("Testing classes of %s DataSet..." % new_testing_data.get_name())
        new_testing_data.set_label("class")
        evaluate = self.__internal_scaling(new_testing_data, print_removed=print_removed)
        if evaluate.is_empty():
            raise ValueError("All given samples for testing were out of bounds. Please only test samples in unscaled range: "
                             "\n[%s]\n[%s]\nwith this classification object" %
                             (', '.join([str(x) for x in self.__data_range[0]]), ', '.join([str(x) for x in self.__data_range[1]])))
        omitted_data, used_data = evaluate.split_without_labels()
        if not omitted_data.is_empty():
            print("Omitted some classless samples during testing and added them to omitted sample collection of this object.")
        self.__omitted_data.concatenate(omitted_data)
        self.__scaled_data.concatenate(used_data)
        self.__testing_data.concatenate(used_data)
        calculated_new_testclasses = self.__classificate(used_data)
        self.__calculated_classes_testset = np.concatenate((self.__calculated_classes_testset, calculated_new_testclasses))
        if print_output:
            self.__print_evaluation(used_data, calculated_new_testclasses,
                                    self.__densities_testset[(len(self.__densities_testset) - new_testing_data.get_length()):],
                                    print_incorrect_points)
        return omitted_data

    def get_original_data(self) -> 'DataSet':
        ret_val = self.__original_data.copy()
        ret_val.set_name(self.__original_data.get_name())
        return ret_val

    def get_omitted_data(self) -> 'DataSet':
        ret_val = self.__omitted_data.copy()
        ret_val.set_name(self.__omitted_data.get_name())
        return ret_val

    def get_learning_data(self) -> 'DataSet':
        ret_val = self.__learning_data.copy()
        ret_val.set_name(self.__learning_data.get_name())
        return ret_val

    def get_testing_data(self) -> 'DataSet':
        ret_val = self.__testing_data.copy()
        ret_val.set_name(self.__testing_data.get_name())
        return ret_val

    def get_dataset_range(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.__data_range

    def get_scale_factor(self) -> float:
        return self.__scale_factor

    def get_calculated_classes_testset(self) -> List[int]:
        return self.__calculated_classes_testset.copy()

    def __initialize(self, percentage: float, split_evenly: bool) -> None:
        """Initialize data for performing classification.

        Calculates which parts of the original data (self.__data) should be used as learning and testing data.
        If percentage is 1, all of the original data is used as learning data.
        Any classless samples in the original dataset are removed and stored in self.__omitted_data first.
        Scaling to range (0.005, 0.995) is performed either simply based on boundary samples (default) or by the original data range (if
        specified by the user in the constructor). If the latter, samples out of bounds are removed, printed to stdout and the user is notified.
        Before splitting the data, it is shuffled to ensure random splitting and the boundary samples are moved to the front to guarantee that
        they are in the learning dataset.

        :param percentage: Percentage of original data (self.__data) which to use as learning dataset.
        :param split_evenly: Only relevant when 0 < percentage < 1. Conditional parameter which indicates whether the learning datasets
        for each class should be of near equal size
        :return: None
        """
        self.__scaled_data = self.__original_data.copy()
        self.__scaled_data.set_name(self.__original_data.get_name())
        self.__scaled_data.set_label("class")
        self.__omitted_data, used_data = self.__scaled_data.split_without_labels()
        self.__omitted_data.set_name("%s_omitted" % self.__scaled_data.get_name())
        used_data.set_name(self.__scaled_data.get_name())
        self.__scaled_data = used_data
        if self.__scaled_data.is_empty():
            raise ValueError("Can't perform classification learning on empty or classless DataSet.")
        if self.__data_range is not None:
            if any(x <= y for x, y in zip(self.__data_range[1], self.__data_range[0])):
                raise ValueError("Invalid dataset range.")
            self.__scale_factor = 0.99 / (self.__data_range[1] - self.__data_range[0])
            self.__scaled_data = self.__internal_scaling(self.__scaled_data, print_removed=True)
        else:
            self.__scaled_data.scale_range((0.005, 0.995), override_scaling=True)
            self.__data_range = (self.__scaled_data.get_original_min(), self.__scaled_data.get_original_max())
            self.__scale_factor = self.__scaled_data.get_scaling_factor()
        if not self.__omitted_data.is_empty():
            print("Omitted some classless samples during initialization and added them to omitted sample collection of this object.")
            self.__omitted_data.shift_value(-self.__data_range[0], override_scaling=True)
            self.__omitted_data.scale_factor(self.__scale_factor, override_scaling=True)
            self.__omitted_data.shift_value(0.005, override_scaling=True)
        self.__scaled_data.shuffle()
        self.__scaled_data.move_boundaries_to_front()
        if split_evenly:
            data_classes = self.__scaled_data.split_labels()
            data_learn_list = [x.split_pieces(percentage)[0] for x in data_classes]
            data_test_list = [x.split_pieces(percentage)[1] for x in data_classes]
            data_learn = DataSet.list_concatenate(data_learn_list)
            data_test = DataSet.list_concatenate(data_test_list)
        else:
            data_learn, data_test = self.__scaled_data.split_pieces(percentage)
        self.__learning_data = data_learn
        self.__learning_data.set_name("%s_learning_data" % self.__scaled_data.get_name())
        self.__testing_data = data_test
        self.__testing_data.set_name("%s_testing_data" % self.__scaled_data.get_name())

    def __classificate(self, data_to_classificate: DataSet) -> np.ndarray:
        """Calculate classes for samples of input data.

        Computes the densities of each class for every sample. The class which corresponds to the highest density for a sample is assigned to it.
        Classes are stored into a list in the same order as the corresponding samples occur in the input data.

        :param data_to_classificate: DataSet whose samples are to be classified
        :return: List of computed classes in the same order as their corresponding samples
        """
        density_data = list(zip(*[x(data_to_classificate[0]) for x in self.__classificators]))
        self.__densities_testset += density_data
        return np.argmax(density_data, axis=1).flatten()
        # return [j for i, a in enumerate(density_data) for j, b in enumerate(a) if b == max_density_per_point[i]]

    def __internal_scaling(self, data_to_check: DataSet, print_removed: bool = False) -> 'DataSet':
        """Scale data with the same factors as the original data (self.__data) was scaled.

        If the input data is already scaled, it is assumed that its scaling matches that of the original data.
        If that's not the case, the user should first revert the scaling of all input data before applying it to a Classification object.
        If not already scaled, the input data will be scaled with the same factors the original data was.
        Any samples out of bounds after scaling are removed, printed to stdout if print_removed is True and a the user is notified.

        :param data_to_check: DataSet which needs to be checked for scaling and scaled if necessary
        :param print_removed: Optional. Conditional parameter which indicates whether any during scaling removed samples should be printed
        :return: Scaled input dataset without samples out of bounds
        """
        if data_to_check.is_scaled():
            if not self.__scaled_data.same_scaling(data_to_check):
                raise ValueError("Provided DataSet's scaling doesn't match the internal scaling of Classification object.")
        else:
            data_to_check.shift_value(-self.__data_range[0], override_scaling=False)
            data_to_check.scale_factor(self.__scale_factor, override_scaling=False)
            data_to_check.shift_value(0.005, override_scaling=False)
        remove_indices = [i for i, x in enumerate(data_to_check[0]) if any([(y < 0.0049) for y in x]) or any([(y > 0.9951) for y in x])]
        removed_samples = data_to_check.remove_samples(remove_indices)
        if not removed_samples.is_empty():
            print("During internal scale checking of %s DataSet some samples were removed due to them being out of bounds of classificators."
                  % data_to_check.get_name())
            if print_removed:
                print("Points removed during scale checking:")
                for i, x in enumerate(removed_samples[0]):
                    print(i, end=": ")
                    print(x, end=" | class: ")
                    print(removed_samples[1][i])
        return data_to_check

    @staticmethod
    def __print_evaluation(testing_data: DataSet,
                           calculated_classes: np.ndarray,
                           density_testdata: List[np.ndarray],
                           print_incorrect_points: bool = True) -> None:
        """Print the results of some specified testing data to stdout.

        Only prints evaluation if input is valid.
        Prints the number and percentage of incorrectly computed samples.
        Prints all samples of input test data that were classified incorrectly.

        :param testing_data: Input testing data for which to print the results
        :param calculated_classes: Input calculated classes for specified testing data
        :return: None
        """
        if testing_data.is_empty():
            warnings.formatwarning = lambda msg, ctg, fname, lineno, file=None, line=None: "%s:%s: %s: %s\n" % (fname, lineno, ctg.__name__, msg)
            warnings.warn("Nothing to print; input test dataset is empty.", stacklevel=3)
            return
        if testing_data.get_length() != len(calculated_classes):
            raise ValueError("Samples of testing DataSet and its calculated classes have to be the same amount.")
        number_wrong = sum([0 if (x == y) else 1 for x, y in zip(testing_data[1], calculated_classes)])
        indices_wrong = [i for i, c in enumerate(testing_data[1]) if c != calculated_classes[i]]
        print("Number of wrong mappings:", end=" ")
        print(number_wrong)
        print("Number of total mappings:", end=" ")
        print(len(calculated_classes))
        print("Percentage of correct mappings:", end=" ")
        print("%2.2f%%" % ((1.0 - (number_wrong / len(calculated_classes))) * 100))
        if number_wrong != 0 and print_incorrect_points:
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            print("Points mapped incorrectly:")
            for i, wr in enumerate(indices_wrong):
                print(i, end=": ")
                print(testing_data[0][wr], end=" | correct class: ")
                print(testing_data[1][wr], end=", calculated class: ")
                print(calculated_classes[wr], end=" | ")
                for j, x in enumerate(density_testdata[wr]):
                    print("density_class%d" % j, end=": ")
                    print(x, end=", ")
                print()

    def __process_performed_classification(self, operation_list: List[Tuple[StandardCombi, DensityEstimation]], start_time: float) -> None:
        """Extract StandardCombi and DensityEstimation objects from operation list for every class and store them in private attributes.

        Revert the scaling of the original dataset, so there is no confusion for the user when looking at the samples' values.
        Print the time used for performing classification.
        Already evaluate testing data if the original data was split into learning and testing data in __initialize().

        :param operation_list: List of Tuples of StandardCombi and DensityEstimation objects which each can be assigned to a class.
        :param start_time: Time when the performing of classification of this object started.
        :return:
        """
        self.__classificators = [x[0] for x in operation_list]
        self.__de_objects = [x[1] for x in operation_list]
        self.__performed_classification = True
        print()
        print("=================================================================================================================================")
        print("Performed Classification of '%s' DataSet." % self.__scaled_data.get_name())
        print("Time used: %.10f seconds" % (time.time() - start_time))
        if not self.__testing_data.is_empty():
            self.__calculated_classes_testset = self.__classificate(self.__testing_data)

    def perform_classification(self,
                               masslumping: bool = True,
                               lambd: float = 0.0,
                               minimum_level: int = 1,
                               maximum_level: int = 5) -> None:
        """Create GridOperation and DensityEstimation objects for each class of samples and store them into lists.

        This method is only called once.
        First the learning dataset is split into its classes in separate DataSets and then the DataSet.density_estimation() function is called for
        each of the single-class-DataSets.
        The DensityEstimation objects are mainly used for plotting the combi scheme.

        :param masslumping: Optional. Conditional Parameter which indicates whether masslumping should be enabled for DensityEstimation
        :param lambd: Optional. Parameter which adjusts the 'smoothness' of DensityEstimation results
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation
        :return: None
        """
        if self.__performed_classification:
            raise ValueError("Can't perform classification for the same object twice.")
        start_time = time.time()
        learning_data_classes = self.__learning_data.split_labels()
        operation_list = [x.density_estimation(masslumping=masslumping, lambd=lambd, minimum_level=minimum_level, maximum_level=maximum_level,
                                               plot_de_dataset=False, plot_density_estimation=False, plot_combi_scheme=False, plot_sparsegrid=False)
                          for x in learning_data_classes]
        self.__process_performed_classification(operation_list, start_time)

    def perform_classification_dimension_wise(self,
                                              masslumping: bool = True,
                                              lambd: float = 0.0,
                                              minimum_level: int = 1,
                                              maximum_level: int = 5,
                                              reuse_old_values: bool = False,
                                              numeric_calculation: bool = True,
                                              margin: float = 0.5,
                                              tolerance: float = 0.01,
                                              max_evaluations: int = 256,
                                              modified_basis: bool = False,
                                              boundary: bool = False) -> None:
        """Create GridOperation and DensityEstimation objects for each class of samples and store them into lists.

        This method is only called once.
        First the learning dataset is split into its classes in separate DataSets and then the DataSet.density_estimation() function is called for
        each of the single-class-DataSets.
        The DensityEstimation objects are mainly used for plotting the combi scheme.

        :param masslumping: Optional. Conditional Parameter which indicates whether masslumping should be enabled for DensityEstimation
        :param lambd: Optional. Parameter which adjusts the 'smoothness' of DensityEstimation results
        :param minimum_level: Optional. Minimum Level of Sparse Grids on which to perform DensityEstimation
        :param maximum_level: Optional. Maximum Level of Sparse Grids on which to perform DensityEstimation
        :param reuse_old_values: ...
        :param numeric_calculation: ...
        :param margin: ...
        :param tolerance: ...
        :param max_evaluations: ...
        :param modified_basis: ...
        :param boundary: ...
        :return: None
        """
        if self.__performed_classification:
            raise ValueError("Can't perform classification for the same object twice.")
        start_time = time.time()
        learning_data_classes = self.__learning_data.split_labels()
        operation_list = [x.density_estimation_dimension_wise(masslumping=masslumping,
                                                              lambd=lambd,
                                                              minimum_level=minimum_level,
                                                              maximum_level=maximum_level,
                                                              reuse_old_values=reuse_old_values,
                                                              numeric_calculation=numeric_calculation,
                                                              margin=margin,
                                                              tolerance=tolerance,
                                                              max_evaluations=max_evaluations,
                                                              modified_basis=modified_basis,
                                                              boundary=boundary,
                                                              plot_de_dataset=False,
                                                              plot_density_estimation=False,
                                                              plot_combi_scheme=False,
                                                              plot_sparsegrid=False) for x in learning_data_classes]
        self.__process_performed_classification(operation_list, start_time)

    def print_evaluation(self, print_incorrect_points: bool = True) -> None:
        """Print results of all testing data that was evaluated with this object.

        As most of other public methods of Classification, classification already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        In case self.__testing_data is empty, a warning is issued and the method returns without printing anything.

        :return: None
        """
        if not self.__performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if self.__testing_data.is_empty():
            warnings.formatwarning = lambda msg, ctg, fname, lineno, file=None, line=None: "%s:%s: %s: %s\n" % (fname, lineno, ctg.__name__, msg)
            warnings.warn("Nothing to print; test dataset of this object is empty.", stacklevel=3)
            return
        print("---------------------------------------------------------------------------------------------------------------------------------")
        print("Printing evaluation of all current testing data...")
        self.__print_evaluation(self.__testing_data, np.array(self.__calculated_classes_testset), self.__densities_testset, print_incorrect_points)

    def plot(self,
             plot_class_dataset: bool = True,
             plot_class_density_estimation: bool = True,
             plot_class_combi_scheme: bool = True,
             plot_class_sparsegrid: bool = False) -> None:
        """Plot a Classification object.

        As most of other public methods of Classification, classification already has to be performed before this method is called. Otherwise an
        AttributeError is raised.
        The user can specify exactly what to plot with conditional parameters to this method.

        :param plot_class_dataset: Optional. Conditional parameter which specifies whether the learning DataSet should be plotted. Default True
        :param plot_class_density_estimation: Optional. Conditional parameter which specifies whether the density estimation of each class should be
        plotted. Default True
        :param plot_class_combi_scheme: Optional. Conditional parameter which specifies whether the resulting combi schemes of each class should be
        plotted. Default True
        :param plot_class_sparsegrid: Optional. Conditional parameter which specifies whether the resulting sparsegrids of each class should be
        plotted. Default True
        :return: None
        """
        if not self.__performed_classification:
            raise AttributeError("Classification needs to be performed on this object first.")
        if plot_class_dataset:
            self.__learning_data.plot()
        if plot_class_density_estimation:
            for x in self.__classificators:
                x.plot(contour=True)
        if plot_class_combi_scheme:
            for x, y in zip(self.__classificators, self.__de_objects):
                x.print_resulting_combi_scheme(operation=y)
        if plot_class_sparsegrid:
            for x in self.__classificators:
                x.print_resulting_sparsegrid(markersize=20)


class Clustering:
    pass
    # TODO insert with next commit


if __name__ == "__main__":
    import sys

    sys.path.append('../src/')

    # generate a dataset of size with the sklearn library
    size = 500
    # sklearn_dataset = datasets.make_circles(n_samples=size, noise=0.05)
    # sklearn_dataset = datasets.make_moons(n_samples=size, noise=0.1)
    # sklearn_dataset = datasets.make_classification(size, n_features=2, n_redundant=0, n_clusters_per_class=1, n_informative=1, n_classes=2)
    # sklearn_dataset = datasets.make_classification(size, n_features=3, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=4)
    sklearn_dataset = datasets.make_blobs(n_samples=size, n_features=2, centers=10)
    # sklearn_dataset = datasets.make_gaussian_quantiles(n_samples=size, n_features=2, n_classes=6)
    # sklearn_dataset = datasets.load_digits(return_X_y=True)
    # sklearn_dataset = datasets.make_biclusters((2, 2), n_clusters=3)
    # sklearn_dataset = datasets.load_iris(return_X_y=True)
    # sklearn_dataset = datasets.load_breast_cancer(return_X_y=True)
    # sklearn_dataset = datasets.load_wine(return_X_y=True)
    # sklearn_dataset = datasets.make_checkerboard()

    # now we can transform this dataset into a DataSet object and give it an appropriate name
    data = DataSet(sklearn_dataset, name='Testset')

    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # now let's look at some functions of the DataSet class

    # DataSet objects can e.g. be ...
    data_copy = data.copy()  # deepcopied
    data_copy.scale_range((0.005, 0.995))  # scaled
    part0, part1 = data_copy.split_pieces(0.5)  # split
    data_copy = part0.concatenate(part1)  # concatenated
    data_copy.set_name('2nd_Set')  # renamed
    data_copy.remove_labels(0.2)  # freed of some class assignments to samples
    without_classes, with_classes = data_copy.split_without_labels()  # seperated into samples with and without classes
    #data_copy.plot()  # plotted

    # and of course we can perform a regular density estimation on a DataSet object:
    #de_retval = data_copy.density_estimation(plot_de_dataset=True, plot_sparsegrid=False, plot_density_estimation=True, plot_combi_scheme=True)

    # ==============================================================================================================================================

    # initialize Classification object with our original unedited data, 80% of this data is going to be used as learning data which has equally
    # distributed classes
    classification = Classification(data, split_percentage=0.8, split_evenly=True)

    # after that we should immediately perform the classification for the learning data tied to the Classification object, since we can't really
    # call any other method before that without raising an error
    classification.perform_classification(masslumping=True, lambd=0.0, minimum_level=1, maximum_level=5)

    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # now we can perform some other operations on this classification object

    # we could e.g. plot its classificators and corresponding density estimations
    #classification.plot(plot_class_sparsegrid=False, plot_class_combi_scheme=False, plot_class_dataset=True, plot_class_density_estimation=True)

    # if we already added some testing data to the Classification object (which we did in the initialization process, 20% of samples are testing
    # samples), we can print the current evaluation
    classification.print_evaluation(print_incorrect_points=True)

    # we can also add more testing data and print the results immediately
    with_classes.set_name("Test_new_data")
    classification.test_data(with_classes, print_output=True, print_incorrect_points=False)

    # and we can call the Classification object to perform blind classification on a dataset with unknown class assignments to its samples
    data_copy.remove_labels(1.0)
    calcult_classes = classification(data_copy)

    # because we used 2D datasets before we can plot the results to easily see which samples were classified correctly and which not
    correct_classes = data.copy()
    correct_classes.scale_range((0.005, 0.995))
    correct_classes.set_name('Correct_classes')
    calcult_classes.set_name('Calculated_classes')
    correct_classes.plot()
    calcult_classes.plot()
