import unittest
import sparseSpACE
import sparseSpACE.HP_Optimization as hpo
import numpy as np
import math

class TestHP_Optimization(unittest.TestCase):
	def test_HPO(self):
		OC = hpo.Optimize_Classification(data_name = "moons")
		HPO = hpo.HP_Optimization(OC.pea_classification, OC.classification_space)
		#test check_if_in_array (only works with np.array)
		array = np.array([[0, 1, 1, 1], [7, 3, 6, 5], [4.2, 3.3, 5.2, 9], [7, 7, 7, 7]])
		b = HPO.check_if_in_array([7, 7, 7, 7], array)
		print("b = " + str(b))
		self.assertTrue(b)
		self.assertTrue(HPO.check_if_in_array([4.2, 3.3, 5.2, 9], array))
		self.assertFalse(HPO.check_if_in_array([1, 0, 0, 0], array))
		#test get_beta
		beta_1 = HPO.get_beta(1)
		self.assertAlmostEqual(beta_1, 34.74985580606742)
		#test cov
		v1 = np.array([1, 2])
		v2 = np.array([3, 4])
		res = HPO.cov(v1, v2)
		self.assertAlmostEqual(res, math.exp(-16))
		#test cov_matrix
		C_x=np.array([[1, 2], [3, 4]])
		K = HPO.get_cov_matrix(C_x, 2)
		control_K = np.array([[1, math.exp(-16)], [math.exp(-16), 1]])
		np.testing.assert_almost_equal(K, control_K)
		#test cov_vector
		C_x=np.array([[1, 2], [5, 6]])
		x = np.array([3, 4])
		v = HPO.get_cov_vector(C_x, x, 2)
		control_v = np.array([math.exp(-16), math.exp(-64)])
		np.testing.assert_almost_equal(v, control_v)
		#test check_hp_space
		simple_function = lambda x: x[0]+x[1]
		simple_space = []
		cmp_space = [["list", 1]]
		HPO_simple = hpo.HP_Optimization(simple_function, simple_space)
		self.assertEqual(HPO_simple.hp_space, cmp_space)
		simple_space = [["interval", 5, 3.3], [7], ["interval_int", 4.2, 1.1], ["interval", 7]]
		cmp_space = [["interval", 3.3, 5], ["list", 1], ["interval_int", 1, 4], ["interval", 7, 7]]
		HPO_simple = hpo.HP_Optimization(simple_function, simple_space)
		self.assertEqual(HPO_simple.hp_space, cmp_space)
		simple_space = [["list", 0.42, 3.3], ["interval", 2, 4], ["interval_int", 2, 4]]
		HPO_simple = hpo.HP_Optimization(simple_function, simple_space)
		self.assertEqual(HPO_simple.hp_space, [["list", 0.42, 3.3], ["interval", 2, 4], ["interval_int", 2, 4]])
		#test cart_prod_hp_space
		cart_prod = HPO_simple.cart_prod_hp_space(1)
		cmp_cart_prod = [(0.42, 2.0, 2), (0.42, 2.0, 4), (0.42, 4.0, 2), (0.42, 4.0, 4), (3.3, 2.0, 2), (3.3, 2.0, 4), (3.3, 4.0, 2), (3.3, 4.0, 4)]
		self.assertEqual(cart_prod, cmp_cart_prod)
		#test perform_eval_at
		res = HPO_simple.perform_evaluation_at([0.42, 4.0, 2])
		self.assertEqual(res, 4.42)
		res2 = HPO_simple.perform_evaluation_at([42, 1337])
		self.assertEqual(res2, 1379)
		#test GO
		simple_space = [["list", 0.42, 1, 2.5, 3.3], ["list", 2, 4]]
		HPO_simple = hpo.HP_Optimization(simple_function, simple_space)
		res_GO = HPO_simple.perform_GO()
		np.testing.assert_almost_equal(res_GO[0], [3.3, 4])
		self.assertAlmostEqual(res_GO[1], 7.3)
		#test round x
		simple_space = [["list", 0.42, 1, 2.5, 3.3], ["interval", 2, 4], ["interval_int", 1, 5]]
		HPO_simple = hpo.HP_Optimization(simple_function, simple_space)
		x = [0.4, 3.33, 2.7]
		x_rd = HPO_simple.round_x(x)
		y = [2.5, 5, 5]
		y_rd = HPO_simple.round_x(y)
		self.assertEqual(x_rd, [0.42, 3.33, 3])
		self.assertEqual(y_rd, [2.5, 4, 5])
		#test get random x
		x = HPO_simple.create_random_x()
		x_rd = HPO_simple.round_x(x)
		self.assertEqual(x, x_rd)
		#test create_evidence_set
		C = HPO_simple.create_evidence_set(5, 3)
		self.assertEqual(len(C[0]), 9)
		self.assertEqual(len(C[0][0]), 3)
		self.assertEqual(HPO_simple.perform_evaluation_at(C[0][0]), C[1][0])
		cmp = [0, 0, 0]
		np.testing.assert_equal(C[0][7], cmp)
		#test get_bounds_and_x0
		#bounds are supposed to be smallest and biggest possible values, x0 consists of smallest values
		bounds_and_x0 = HPO_simple.get_bounds_and_x0(0)
		self.assertEqual(bounds_and_x0[0][0][0], 0.42)
		self.assertEqual(bounds_and_x0[0][0][1], 3.3)
		self.assertEqual(bounds_and_x0[0][1][0], 2)
		self.assertEqual(bounds_and_x0[0][1][1], 4)
		self.assertEqual(bounds_and_x0[0][2][0], 1)
		self.assertEqual(bounds_and_x0[0][2][1], 5)

		self.assertEqual(bounds_and_x0[1][0], 0.42)
		self.assertEqual(bounds_and_x0[1][1], 2)
		self.assertEqual(bounds_and_x0[1][2], 1)

	def simple_function(self, x):
		return x[0]+x[1]

if __name__ == '__main__':
	unittest.main()
