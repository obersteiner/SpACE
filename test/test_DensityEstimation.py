import unittest
from sys import path

path.append('../src/')
from StandardCombi import *
from Grid import *
from GridOperation import DensityEstimation

dim = 2

# define integration domain boundaries
a = np.zeros(dim)
b = np.ones(dim)

# define data
data = datasets.make_moons()

# define operation to be performed
operation = DensityEstimation(data, dim)
operation.initialize()

combiObject = StandardCombi(a, b, operation=operation)
minimum_level = 1
maximum_level = 4


class TestDensityEstimation(unittest.TestCase):

    def test_DE(self):
        levelvecs = [(1, 4), (2, 3), (3, 2), (4, 1), (1, 3), (2, 2), (3, 1)]
        alphas = {(4, 1): [2.41206017, 0.1531592, 0.38504719, -0.38292677, 1.55369606,
                           3.34255515, 0.2680686, 0.78571641, 0.2680686, 3.34255515,
                           1.55369606, -0.38292677, 0.38504719, 0.1531592, 2.41206017],
                  (3, 2): [0.29486562, -0.32856807, 3.46802364, -0.80132467, 0.00657869,
                           -0.11637183, 2.49762678, 3.29185638, -0.04437461, 1.84573932,
                           -2.82205134, 1.84573932, -0.04437461, 3.29185638, 2.49762678,
                           -0.11637183, 0.00657869, -0.80132467, 3.46802364, -0.32856807,
                           0.29486562], (2, 3): [-0.4893803, 0.55551402, 0.6274426, 0.7448376, 1.1356473,
                                                 -0.1588984, 1.96568153, 2.82296495, 0.58130668, 0.91681425,
                                                 0.60534591, 0.91681425, 0.58130668, 2.82296495, 1.96568153,
                                                 -0.1588984, 1.1356473, 0.7448376, 0.6274426, 0.55551402,
                                                 -0.4893803],
                  (1, 4): [3.3174027, 1.16074918, 1.13812157, 0.79453512, 0.84930383,
                           1.37679405, 0.76569613, 1.29273513, 0.76569613, 1.37679405,
                           0.84930383, 0.79453512, 1.13812157, 1.16074918, 3.3174027],
                  (3, 1): [1.20595699, -0.34008514, 2.97776214, -0.37941974, 2.97776214,
                           -0.34008514, 1.20595699],
                  (2, 2): [0.22760649, 0.87148175, 0.89247333, 1.73628713, 0.27021045,
                           1.73628713, 0.89247333, 0.87148175, 0.22760649],
                  (1, 3): [2.31825217, 0.51211262, 1.23434597, 0.93710882, 1.23434597,
                           0.51211262, 2.31825217]}
        for i in range(len(levelvecs)):
            R = operation.construct_R(levelvecs[i])
            b = operation.calculate_B(operation.data, levelvecs[i])
            numbOfPoints = np.prod(operation.grid.levelToNumPoints(levelvecs[i]))
            # Check matrix size
            self.assertEqual(numbOfPoints, len(R))
            self.assertEqual(numbOfPoints, len(R[0]))
            self.assertEqual(numbOfPoints * numbOfPoints, np.prod(R.shape))
            self.assertEqual(numbOfPoints, len(b))
            alpha1 = solve(R, b)
            alpha2 = alphas.get(levelvecs[i])
            self.assertEqual(len(alpha1), len(alpha2))
            for j in range(len(alpha1)):
                self.assertAlmostEqual(alpha1[j], alpha2[j])


if __name__ == '__main__':
    unittest.main()