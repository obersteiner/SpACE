import sparseSpACE.DEMachineLearning as deml
import sparseSpACE
import numpy as np
import scipy as sp
import math
import random
import itertools
from scipy.optimize import fmin
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#class HP_Optimization gets (data not needed bc only pea needs that which is the function!),
#the function on which HP should be Optimized and search space for the HPs
#space is in form [["interval", <start>, <end>], ["list", <l0>, <l1>, <l2>,...], ....]
class HP_Optimization:
	def __init__(self, function, hp_space, f_max = None):
		self.function = function
		self.hp_space = hp_space
		self.check_hp_space()
		self.f_max = f_max
		self.r = 3.0
		self.delta = 0.1
		self.a = 1.0
		self.b = 1.0
		self.l = 0.5

	def check_hp_space(self):
		if(len(self.hp_space)<1):
			print("Please enter values for hp_space! Using default value ['list', 1], but will likely throw errors")
			self.hp_space.append(["list", 1])
			return 0
		for i in range(0, len(self.hp_space)):
			if(len(self.hp_space[i])<2):
				print("Please enter at least a descriptor and one value for hp_space index " + str(i))
				print("Default value ['list', 1] will be used")
				self.hp_space[i] = ["list", 1]
			elif(self.hp_space[i][0]=="interval"):
				if(len(self.hp_space[i])<3):
					self.hp_space[i].append(self.hp_space[i][1])
				elif(self.hp_space[i][1]>self.hp_space[i][2]):
					self.hp_space[i] = ["interval", self.hp_space[i][2], self.hp_space[i][1]]
				else:
					self.hp_space[i] = ["interval", self.hp_space[i][1], self.hp_space[i][2]]
			elif(self.hp_space[i][0]=="interval_int"):
				if(len(self.hp_space[i])<3):
					self.hp_space[i] = ["interval_int", round(self.hp_space[i][1]), round(self.hp_space[i][1])]
				elif(self.hp_space[i][1]>self.hp_space[i][2]):
					self.hp_space[i] = ["interval_int", round(self.hp_space[i][2]), round(self.hp_space[i][1])]
				else:
					self.hp_space[i] = ["interval_int", round(self.hp_space[i][1]), round(self.hp_space[i][2])]
			elif(self.hp_space[i][0]=="list"):
				self.hp_space[i] = self.hp_space[i]
			else:
				print("Unknown descriptor for hp_space at index " + str(i) +". Using ['list', <first value given>]")
				self.hp_space[i] = ["list", self.hp_space[i][1]]

	#performs Grid Optimization
	def perform_GO(self, interval_levels = 4, search_space = None):
		#sklearn_dataset = deml.datasets.make_moons(n_samples=samples, noise=0.15)
		#data = deml.DataSet(sklearn_dataset, name='Input_Set')
		#for storing the current best evaluation and time
		best_evaluation = None
		#best_time = None
		#storing the parameters of the best evaluation
		best_x = None
		x_set = []
		y_set = []
		#original classification: classification = deml.Classification(data, split_percentage=0.8, split_evenly=True, shuffle_data=True)
		#original settings: classification.perform_classification(masslumping=True, lambd=0.0, minimum_level=1, maximum_level=5, print_metrics=True)
		#perform grid search for lambd_levels many lambd between 0 and 1 and all values for the rest
		#Note: lambd vllt besser logarithmisch, sehr kleine Werte oft besser (1/10, 1/100, 1/1000)
		#für for-schleife karthesisches produkt aus HO um alle tupel zu bilden und über diese tupel zu iterieren für beliebig dimensional
		#Note: lambd vor Allem sinnvoll wenn masslumping aus ist, sollte kaum unterschied machen wenn an
		if(search_space == None):
			search_space = self.cart_prod_hp_space(interval_levels)
		x_set = search_space
		for x in search_space:
			#cur_time = classification._time_used
			#print ("current time needed = " + str(cur_time))
			cur_evaluation = self.perform_evaluation_at(x)
			y_set.append(cur_evaluation)
			print("Current Evaluation: ",cur_evaluation)
			#if best_evaluation == None or cur_evaluation > best_evaluation or (cur_evaluation == best_evaluation and (best_time == None or best_time>cur_time)):
			if(best_evaluation == None or cur_evaluation>best_evaluation):
				best_evaluation = cur_evaluation
				best_x = x
				print("Best evaluation is now " + str(best_evaluation) + " at " + str(best_x))
			else:	print("Best evaluation is still " + str(best_evaluation) + " at " + str(best_x))
			print ("END OF CURRENT EVALUATION \n")
			if(best_evaluation==self.f_max):
				print("We've reached the specified maximum of f. Stopping GO.")
				break
		print("In the end, best evaluation is " + str(best_evaluation)  + " at " + str(best_x))
		return best_x, best_evaluation, x_set, y_set
		#+ " and time = " + str(best_time))

	def cart_prod_hp_space(self, interval_levels):
		res = []
		if (interval_levels == 0 or interval_levels == None):
			interval_levels = 1
		for i in range (0, len(self.hp_space)):
			new = []
			if(len(self.hp_space[i])<2):
				print("please at least 1 value for hp_space list")
				new.append(1)
			elif(self.hp_space[i][0] == "list"):
				new = self.hp_space[i].copy()
				new.remove("list")
			elif(len(self.hp_space[i])<3):
				print("please enter at least 2 values for hp_space interval. Using the value given")
				new.append(self.hp_space[i][1])
			elif(self.hp_space[i][0] == "interval"):
				h = self.hp_space[i][2]-self.hp_space[i][1]
				dh = h /interval_levels
				#currently adds 1 more than interval_levels so that both borders are included - is there a better way to do this?
				if(h == 0):
					new.append(self.hp_space[i][1])
				else:
					for j in range (0, interval_levels+1):
						new.append(self.hp_space[i][1]+j*dh)
			elif(self.hp_space[i][0] == "interval_int"):
				h = int(self.hp_space[i][2]-self.hp_space[i][1])
				dh = h /interval_levels
				#currently adds 1 more than interval_levels so that both borders are included - is there a better way to do this?
				if(h == 0):
					new.append(int(self.hp_space[i][1]))
				else:
					if(dh<1):
						dh = 1
					#may yield different values than expected due to rounding
					for j in range (self.hp_space[i][1], self.hp_space[i][2]+1, int(dh)):
						new.append(int(j))
			else:
				print("please enter valid types for hp_space! Using the first value given")
				new.append(self.hp_space[i][1])
			res.append(new)
		res_cart = list(itertools.product(*res))
		return res_cart

	#performs random optimization
	def perform_RO(self, amt_it: int):
		best_evaluation = 0
		best_x = None
		x_set = []
		y_set = []
		for i in range (0, amt_it, 1):
			print("Random step " + str(i))
			x = self.create_random_x()
			new_eval = self.perform_evaluation_at(x)
			x_set.append(x)
			y_set.append(new_eval)
			if new_eval>best_evaluation:
				best_x = x
				best_evaluation = new_eval
			if(best_evaluation == self.f_max):
				print("We've reached the specified maximum of f. Stopping RO at iteration " + str(i))
				break
		print("Best evaluation in " + str(amt_it) + " random steps: " + str(best_evaluation) + " at " + str(best_x))
		return best_x, best_evaluation, x_set, y_set

	#returns evaluation for certain Parameters on a certain data set
	def perform_evaluation_at(self, params):
		#classification = deml.Classification(cur_data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
		#classification.perform_classification(masslumping=cur_massl, lambd=cur_lambd, minimum_level=cur_min_lv, maximum_level=5, one_vs_others=cur_one_vs_others, print_metrics=False)
		#evaluation = classification.evaluate()
		#print("Percentage of correct mappings",evaluation["Percentage correct"])
		##wenn Zeit mit reingerechnet werden soll o.ä. in dieser Funktion
		#return evaluation["Percentage correct"]
		print("performing evaluation at " + str(params))
		res = self.function(params)
		print("returning " + str(res))
		return res

	#classification_space = [["interval", 0, 1], ["list", 0, 1], ["list", 1, 2, 3], ["list", 0, 1]]

	#performs Bayesian Optimization
	#Note: HP Input of classification is (lambd:float, massl:bool, min_lv:int <4, one_vs_others:bool)
	def perform_BO(self, amt_it: int, r: float = 3.0, delta: float = 0.1, a: float = 1.0, b: float = 1.0):
		self.r = r
		self.delta = delta
		self.a = a
		self.b = b
		#notes how many x values currently are in the evidence set - starts with amt_HP+1
		amt_HP = len(self.hp_space)
		cur_amt_x: int = amt_HP+1
		x_ret = None
		y_ret = None
		print("I should do Bayesian Optimization for " + str(amt_it) + " iterations!")
		C = self.create_evidence_set(amt_it, amt_HP)
		C_x=C[0]
		print("C_x: " + str(C_x))
		C_y=C[1]
		print("C_y: " + str(C_y))
		#print("Evidence Set: \n" + str(C))
		#For easier checking in what iterations random x has been used
		used_random_x = []
		for i in range (0, amt_it, 1):
			print("iteration: " + str(i))
			beta = self.get_beta(i+1)
			l = self.get_l_k(i)
			print("beta: " + str(beta) + ", cur_amt_x: " + str(cur_amt_x))
			if(np.linalg.det(self.get_cov_matrix(C_x, cur_amt_x))==0):
				print("With the nex x value added the Covariance Matrix is singular.\nBayesian Optimization can not be continued and will be ended here, at the start of iteration " + str(i))
				break
			if((self.get_cov_matrix(C_x, cur_amt_x) == np.identity(len(C_x))*(cur_amt_x+10)).all()):
				print("cov_matr has been modified to be Id*scalar! This means it was singular -> ending at iteration " + str(i))
				break
			else:
				print("Apparently the cov_matr is not singular at the beginning of iteration " + str(i))

			#value that will be evaluated and added to the evidence set
			print("Getting new x:")
			new_x=self.acq_x(beta, l, C_x, C_y, cur_amt_x)
			print("acq_x gives new x = " + str(new_x))
			new_x_rd = self.round_x(new_x)
			old_x_rd = None
			old_beta_and_l = [beta, l]
			print("new x rd: " + str(new_x_rd))
			#erstelle neues beta und l und berechne neuen wert damit
			amt_tries = 0
			while(self.check_if_in_array(new_x_rd, C_x)):
				print("!!!!!!!!!!!Need new beta and l")
				old_x_rd = new_x_rd.copy()
				print("old x rd was: " + str(old_x_rd))
				new_beta_and_l = self.get_new_beta_and_l(beta, cur_amt_x, new_x, C_x, C_y, amt_tries = amt_tries)
				amt_tries += 1
				new_x = self.acq_x(new_beta_and_l[0], new_beta_and_l[1], C_x, C_y, cur_amt_x)
				new_x_rd = self.round_x(new_x)
				print("new x rd is: " + str(new_x_rd))
				if(old_x_rd==new_x_rd and amt_tries >= 3):
					print("It seems a beta and l that give a new x cannot be found. Trying beta = 0 and l = 0.5")
					new_x = self.acq_x(0, 0.5, C_x, C_y, cur_amt_x)
					new_x_rd = self.round_x(new_x)
					if(self.check_if_in_array(new_x_rd, C_x)):
						print("Beta = 0 didn't help. Using random new x and then getting out.")
						print("If a random x that's not in C_x can not be found in 10 tries it is assumed that all possible x have been used already and BO is stopped")
						used_random_x.append(i)
						while(self.check_if_in_array(new_x_rd, C_x) and amt_tries < 13):
							new_x_rd = self.create_random_x()
							amt_tries += 1
						break
					else:
						print("Beta = 0 DID help!")
			print("out of getting beta and l loop")
			if(old_x_rd==new_x_rd or self.check_if_in_array(new_x_rd, C_x)):
				print("There was an infinite loop when getting new x and / or no new x has been found. Ending BO at iteration " + str(i))
				break
			print("adding " + str(new_x_rd))
			C_x[cur_amt_x] = new_x_rd
			C_y[cur_amt_x] = self.perform_evaluation_at(new_x_rd)
			cur_amt_x += 1
			if(C_y[cur_amt_x-1]==self.f_max):
				print("We've reached the specified maximum of f. Stopping BO at iteration " + str(i))
				break
			#ends everything if cov_matr is singular. Should this be dependant on l? Cov_matr being singular should not depend on l I think
			print("Checking if cov_matr is singular")
			if(np.linalg.det(self.get_cov_matrix(C_x, cur_amt_x)) == 0):
				print("With the nex x value added the Covariance Matrix is singular.\nBayesian Optimization can not be continued and will be ended here, after " + str(i) + " iterations")
				break
			if((self.get_cov_matrix(C_x, cur_amt_x) == np.identity(len(C_x))*(cur_amt_x+10)).all()):
				print("cov_matr has been modified to be Id*scalar! This means it was singular -> ending at iteration " + str(i))
				break
			else:
				print("Apparently the cov_matr is not singular after iteration " + str(i))

		for i in range(0, cur_amt_x, 1):
			if(y_ret == None or y_ret<C_y[i]):
				y_ret = C_y[i]
				x_ret = C_x[i]
		print("The Best value found in " + str(amt_it) + " iterations is " + str(y_ret) + " at " + str(x_ret))
		print("Random x has been used in iterations " + str(used_random_x))
		print("C_y: " + str(C_y))
		print("C_x: " + str(C_x))
		return x_ret, y_ret, C_x, C_y, used_random_x

	#use squared exp. kernel as covariance function k, with parameters sigma_k and l_k
	sigma_k = 1
	l_k = 0.5
	#since creation of covariance matrices is done in another function k(x) needs to be outside
	def cov(self, x, y):
		k = lambda x, y: (self.sigma_k**2)*math.exp((-0.5/self.l_k**2)*np.linalg.norm(x-y)**2)
		return k(x, y)

	#cur_length is needed so that rest of matrix stays Id
	def get_cov_matrix(self, C_x, cur_length: int):
		K = np.identity(len(C_x))
		for i in range (0, cur_length, 1):
			for j in range (0, cur_length, 1):
				K[i][j]=self.cov(C_x[i], C_x[j])
		if(np.linalg.det(K) == 0):
			print("Oh no! The covariance matrix is singular!! It is " + str(K))
			print("Current C_x is: " + str(C_x))
			#Maybe change matrix if it is singular? E.g. to scalar*Id, so it's def. not the smallest?
			#Would have to deal with that in other methods though
			#Important!! K is inverted. But also it's used some way in g?? Unsure what to do
			#at least it doesn't crash rn
			#add small values to diagonal? -> verfälscht sachen. Müsste relativ zu werten in Matrix angepasst werden
			#z.b. 10^-15 oder so
			K = np.identity(len(C_x))*(cur_length+10)
		return K

	#returns the vector k for a certain input new_x
	def get_cov_vector(self, C_x, new_x, cur_length: int):
	        #somehow the rest of the function would still work even if x is wrong size, but no idea why / what it does. Better to check
	        if(len(C_x[0]) != len(new_x)):
	                print("Input x for cov_vector has the wrong size")
	                return None
	        k_vec = np.zeros(len(C_x))
	        for i in range (0, cur_length, 1):
	                k_vec[i] = self.cov(C_x[i], new_x)
	        return k_vec

	#creates evidence set using random values
	def create_evidence_set(self, amt_it: int, dim_HP: int):
		x = np.zeros(shape=(amt_it+dim_HP+1, dim_HP))
		y = np.zeros((amt_it+dim_HP+1))
		for i in range(0, dim_HP+1, 1):
			new_x = self.create_random_x()
			while(self.check_if_in_array(new_x, x)):
				new_x = self.create_random_x()
			x[i] = new_x
			#evaluating takes quite some time, especially for min_lv>1
			#y[i] = perform_evaluation_at(data, new_x[0], new_x[1], new_x[2], new_x[3])
			#needs to be casted to int bc otherwise it's automatically float which doesn't work
			y[i] = self.perform_evaluation_at(x[i])

		return x, y

	#returns a random x for the given purpose - needs search space
	def create_random_x(self):
		print("creating random x")
		res = []
		for i in range (0, len(self.hp_space)):
			new_x = 1
			if (len(self.hp_space[i])<2):
				print("Please enter at least 1 value for hp_space list!")
			elif (self.hp_space[i][0] == "list"):
				sel = self.hp_space[i].copy()
				sel.remove("list")
				new_x = random.choice(sel)
			elif (len(self.hp_space[i])<3):
				print("Please enter at least 2 values for hp_space interval! Using the 1 value given")
				new_x = self.hp_space[i][1]
			elif (self.hp_space[i][0] == "interval"):
				new_x = random.uniform(self.hp_space[i][1], self.hp_space[i][2])
			elif (self.hp_space[i][0] == "interval_int"):
				new_x = int(random.randrange(int(self.hp_space[i][1]), int(self.hp_space[i][2]+1)))
				#max+1 bc second border value is not included
			else:
				print("Unknown type of space! Using first value given for index " + str(i))
				new_x = self.hp_space[i][1]
			res.append(new_x)
		return res

	#checks if array has the element x, returns True if yes. Caution: If x has same length but different dimension to the elements of array this may give unintended results
	#maybe later a threshhold can be added if only part of the matrix should be checked
	def check_if_in_array(self, x, array):
	    if(len(x) != len(array[0])):
	        #print("not same size")
	        return False
	    for i in range(0, len(array), 1):
	        if((x == array[i]).all()):
	            return True
	            break
	    return False

	#calculates and returns current beta. t is current iteration
	def get_beta(self, t: int):
		r: float = self.r
		d: int = len(self.hp_space)
		delta: float = self.delta
		a: float = self.a
		b: float = self.b
		beta = 2*math.log(t**2*2*math.pi**2/(3*delta))+2*d*math.log(t**2*d*b*r*math.sqrt(math.log(4*d*a/delta)))
		print("get_beta returns: " + str(beta))
		return beta

	#TODO: get optimized l_k in each iteration
	def get_l_k(self, t: int):
		return self.l
	def set_l_k(self, l: float):
		self.l = l

	#gets new beta and l if old beta and l give values that are already in C
	def get_new_beta_and_l(self, cur_beta: float, cur_amt_x, cur_x, C_x, C_y, amt_tries: int=0):
		print("Getting new beta and l. Current beta: " + str(cur_beta) +", Current l: " + str(self.l_k))
		print("Try: " + str(amt_tries))
		if(np.linalg.det(self.get_cov_matrix(C_x, cur_amt_x))==0):
			print("Getting new beta and l, but suddenly the cov matr is singular??")
		else:
			print("Apparently the cov_matr is not singular. Getting new beta and l")
		#making upper bounds dependable on current values so bounds are never too low
		beta_h = cur_beta+(100*2**amt_tries)
		l_h = self.l_k+(50*2**amt_tries)
		#0 if rd(x) is in ev set C_x, constant otherwise (5)
		#penalty p(x) erhöhen wenn gleicher Wert rauskommt. z.B. immer +50 oder *2 oder anders exponentiell
		p = lambda x: self.check_if_in_array(self.round_x(x), C_x)*(200*2**amt_tries)
		#gets the x value for certain l (z[1]) and beta (z[0])
		new_x = lambda z: self.acq_x(z[0], z[1], C_x, C_y, cur_amt_x)
		#for g: x[0] is \beta+d\beta, x[1] is l. Also d\beta = \beta+d\beta-\beta
		g = lambda x: (x[0]-cur_beta)+np.linalg.norm(cur_x-new_x(x))+p(new_x(x))
		bounds_g = ((cur_beta, beta_h), (self.l_k, l_h))
		print("About to minimize g(...)")
		#due to numerical inaccuracies a matrix might become singular with new beta and l
		#(even though it wouldn't be) mathematically - how to deal with that??
		result = minimize(g, [cur_beta, self.l_k], method='L-BFGS-B', bounds=bounds_g).x
		print("New beta: " + str(result[0]) + ", New l: " + str(result[1]))
		#result is in the form [new beta, new l]
		return result

	def get_bounds_and_x0(self, offset = 0):
		bounds = []
		x0 = np.zeros(len(self.hp_space))
		for i in range(0, len(self.hp_space)):
			new = [0, 1]
			#should one be able to enter just one value for "list"??
			if(len(self.hp_space[i])<2):
				print("please enter at least 1 value for hp_space list. Using default values [0, 1]")
			elif(self.hp_space[i][0] == "list"):
				max = self.hp_space[i][1]
				min = max
				for j in range (2, len(self.hp_space[i])):
					if(self.hp_space[i][j]<min):
						min = self.hp_space[i][j]
					if(self.hp_space[i][j]>max):
						max = self.hp_space[i][j]
				new = [min-offset, max+offset]
			elif(len(self.hp_space[i])<3):
				print("please enter at least 2 values for hp_space interval. Using the one value given")
				new = [self.hp_space[i][1]-offset, self.hp_space[i][1]+offset]
			elif(self.hp_space[i][0] == "interval"):
				new = self.hp_space[i].copy()
				new.remove("interval")
				new = [new[0]-offset, new[1]+offset]
			elif(self.hp_space[i][0] == "interval_int"):
				new = self.hp_space[i].copy()
				new.remove("interval_int")
				new = [new[0]-offset, new[1]+offset]
			else:
				print("please enter a valid hp_space. Using first value given")
				new = [self.hp_space[i][1]-offset, self.hp_space[i][1]+offset]
			bounds.append(new)
			x0[i] = new[0]
		return bounds, x0

	#acquire new x for l, beta, C_x, cur_amt_x, using GP-UCB
	def acq_x(self, beta: float, l: float, C_x, C_y, cur_amt_x):
		#old_l=self.l_k
		self.l_k=l
		K_matr = self.get_cov_matrix(C_x, cur_amt_x)
		if(np.linalg.det(K_matr) == 0):
			print("Covariance Matrix is indeed singular!")
			#return a value that's bullshit? like [0, 0, 0, 0] (min lv is >=1)
		#print(K_matr)
		mu = lambda x: self.get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(C_y))
		sigma_sqrd = lambda x: self.cov(x, x)-self.get_cov_vector(C_x, x, cur_amt_x).dot(np.linalg.inv(K_matr).dot(self.get_cov_vector(C_x, x, cur_amt_x)))
		#takes sqrt of abs(sigma_sqrd) bc otherwise fmin gives an error - might be an overall warning sign tho
		sigma = lambda x: math.sqrt(abs(sigma_sqrd(x)))
		alpha = lambda x: mu(x)+(math.sqrt(abs(beta)))*sigma(x)
		#trying out stuff:
		#alpha = lambda x: mu(x)
		#negates alpha bc maximum has to be found
		alpha_neg = lambda x: -alpha(x)
		#x2 = np.linspace(0, 20, 21)
		#y2 = [alpha_neg(np.array([x_i, 0, 1, 0])) for x_i in x2]
		#plt.plot(x2, y2)
		bounds_and_x0 = self.get_bounds_and_x0() #bounds search space to vicinity of useful values
		bounds_an = bounds_and_x0[0]
		#problem: Nelder-Mead (same as fmin) cannot handle bounds -> use L-BFGS-B for now
		x0 = bounds_and_x0[1]
		new_x=minimize(alpha_neg, x0, method='L-BFGS-B', bounds=bounds_an).x
		#plt.title(str(new_x))
		#plt.show()
		#print("new x: " + str(new_x) + " with function value: " + str(alpha(new_x)))
		print("acq_x returns: " + str(new_x) + " with a_neg(new_x) = " + str(alpha_neg(new_x)))
		print("mu(new_x) = " + str(mu(new_x)) + " , root(beta) = " + str(math.sqrt(beta)) + " , sigma(new_x) = " + str(sigma(new_x)))
		print("sigma_sqrd(new_x) = " + str(sigma_sqrd(new_x)))
		print("k_vec(x) = " + str(self.get_cov_vector(C_x, new_x, cur_amt_x)))
		return new_x

	#needs search space
	def round_x(self, x):
		print("rounding x")
		if len(x) < len(self.hp_space):
			print("Input too short! Returning default values rd(0) for missing values")
			for k in range (len(x), len(self.hp_space)):
				x.append(0)
		if len(x) > len(self.hp_space):
			print("Input too long! Cropping")
		#rounds the values of new_x to values usable as HPs - the question is what kind of rounding makes sense
		#e.g if lambda should be bounded or more values should be rounded to 0
		new_x_rd = []
		for i in range (0, len(self.hp_space)):
			new_x = 1
			if(len(self.hp_space[i])<2):
				print("Please enter at least 1 value for hp_space list. Index: " + str(i) + ": " + str(self.hp_space[i]))
			elif(self.hp_space[i][0] == "list"):
				new_x = self.hp_space[i][1]
				dis = abs(x[i]-self.hp_space[i][1])
				#takes the value in the list that is least away from x
				for j in range (2, len(self.hp_space[i])):
					cur_dis = abs(x[i]-self.hp_space[i][j])
					if(cur_dis < dis):
						dis = cur_dis
						new_x = self.hp_space[i][j]
			elif(len(self.hp_space[i])<3):
				print("Please enter at least 2 values for hp_space interval. Using the one value given")
				new_x = self.hp_space[i][1]
			elif(self.hp_space[i][0] == "interval"):
				if(x[i]<self.hp_space[i][1]):
					new_x = self.hp_space[i][1]
				elif(x[i]>self.hp_space[i][2]):
					new_x = self.hp_space[i][2]
				else:
					new_x = x[i]
			elif(self.hp_space[i][0] == "interval_int"):
				if(x[i]<self.hp_space[i][1]):
					new_x = int(self.hp_space[i][1])
				elif(x[i]>self.hp_space[i][2]):
					new_x = int(self.hp_space[i][2])
				else:
					new_x = round(x[i])
			else:
				print("Unknown type of space! Using first value given for index " + str(i))
				new_x = self.hp_space[i][1]
			new_x_rd.append(new_x)
		return new_x_rd

class Optimize_Classification:
	def __init__(self, data_name: str = "moons", data = None, samples: int = 500, dimension: int = 2, labels: int = 6, max_lv: int = 3, max_evals: int = 256):
		self.samples = samples
		self.dimension = dimension
		self.labels = labels
		if(data == None):
			self.data = self.create_data(data_name)
		else:
			self.data = data
		self.max_lv = max_lv
		self.max_evals = max_evals
		self.classification_space = [["interval", 0, 20], ["list", 0, 1], ["list", 1], ["list", 0, 1]]
		self.class_dim_wise_space = [["interval", 0, 20], ["list", 0, 1], ["list", 1], ["list", 0, 1, 2], ["interval", 0, 1], ["list", 0, 1], ["list", 0, 1], ["interval_int", 2, self.max_evals]]

	def create_data(self, name: str = "moons"):
		dataset = deml.datasets.make_moons(n_samples=self.samples, noise=0.15, random_state=1)
		if(name == "blobs"):
			dataset = deml.datasets.make_blobs(n_samples=self.samples, n_features=self.dimension, centers=self.labels, random_state=1)
		elif(name == "circles"):
			dataset = deml.datasets.make_circles(n_samples=self.samples, noise=0.05, random_state=1)
		elif(name == "classification"):
			dataset = deml.datasets.make_classification(n_samples=self.samples, n_features=self.dimension, n_redundant=0, n_clusters_per_class=1, n_informative=2, n_classes=(self.labels if self.labels < 4 else 4), random_state=1)
		elif(name == "gaussian_quantiles"):
			dataset = deml.datasets.make_gaussian_quantiles(n_samples=self.samples, n_features=self.dimension, n_classes=self.labels, random_state=1)
		elif(name == "digits"):
			dataset = deml.datasets.load_digits(return_X_y=True) # hint: try only with max_level <= 3
		elif(name == "iris"):
			dataset = deml.datasets.load_iris(return_X_y=True)
		elif(name == "breast_cancer"):
			dataset = deml.datasets.load_breast_cancer(return_X_y=True) # hint: try only with max_level <= 4
		elif(name == "wine"):
			dataset = deml.datasets.load_wine(return_X_y=True)
		elif(name != "moons"):
			print("This is not a valid name for a dataset, using moons")
		data = deml.DataSet(dataset, name="data_"+name)
		return data

	def pea_classification(self, params):
		if(len(params)<4):
			print("too little params for pea_classification. Returning 0.0")
			return 0.0
		params = [params[0], int(params[1]), int(params[2]), int(params[3])]
		cur_data = self.data.copy()
		#should implement a smoother way to put in the data set
		classification = deml.Classification(cur_data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
		classification.perform_classification(masslumping=params[1], lambd=10**(-params[0]), minimum_level=params[2], maximum_level=self.max_lv, one_vs_others=params[3], print_metrics=False)
		evaluation = classification.evaluate()
		print("Percentage of correct mappings",evaluation["Percentage correct"])
		#wenn Zeit mit reingerechnet werden soll o.ä. in dieser Funktion
		return evaluation["Percentage correct"]

	def pea_classification_dimension_wise(self, params):
		if(len(params)<4):
			print("too little params for pea_classification. Returning 0.0")
			return 0.0
		#lambd, massl, min_lv, one_vs_others / error calc, margin(float 0-1), rebalancing (bool), use_relative_surplus (bool), max_evaluations (int)
		params = [float(params[0]), int(params[1]), int(params[2]), int(params[3]), float(params[4]), int(params[5]), int(params[6]), int(params[7])]
		cur_data = self.data.copy()
		error_calculator=sparseSpACE.ErrorCalculator.ErrorCalculatorSingleDimMisclassificationGlobal()
		ec = None
		ovo = False
		if(params[3] == 2):
			ec = error_calculator
			ovo = True
		elif(params[3] == 1):
			ovo = True
		classification = deml.Classification(cur_data, split_percentage=0.8, split_evenly=True, shuffle_data=False)
		classification.perform_classification_dimension_wise(masslumping=params[1], lambd=10**(-params[0]), minimum_level=params[2], maximum_level=self.max_lv, one_vs_others=ovo, error_calculator = ec, margin = params[4], rebalancing = params[5], use_relative_surplus = params[6], max_evaluations = params[7], print_metrics=False)
		evaluation = classification.evaluate()
		print("Percentage of correct mappings",evaluation["Percentage correct"])
		#wenn Zeit mit reingerechnet werden soll o.ä. in dieser Funktion
		return evaluation["Percentage correct"]

#anderes dataset ausprobieren?, vllt höhere dimensionen, bis zu dim=10. Note: moons immer 2d, aber z.B.
def simple_test(params):
	return params[0]
simple_space = [["interval", 20.4, 0.]]

dataset_iris = deml.datasets.load_iris(return_X_y=True)
iris_data = deml.DataSet(dataset_iris, name="data_iris")
OC = Optimize_Classification(data_name = "moons")
#HPO = HP_Optimization(OC.pea_classification, OC.classification_space, r = 1.5)
HPO = HP_Optimization(OC.pea_classification, OC.classification_space)
#HPO = HP_Optimization(simple_test, simple_space, f_max = 100, r = 21)
#HPO.perform_BO(5)
#y_r = HPO.perform_RO(10)[3]
#sol_b = HPO.perform_BO(6)
#y_b = sol_b[3]
#x_b = sol_b[2]
#print("y random: " + str(y_r))
#print("y bayesian: " + str(y_b))
#print("x bayesian: " + str(x_b))
#print(OC.pea_classification_dimension_wise([0.0, 0, 1, 0, 0.0, 0, 0, 2]))

#HPO = HP_Optimization(simple_test, simple_space)
#print(HPO.hp_space)
#HPO.perform_GO(search_space=[[1, 1, 2, 1], [5, 4, 2, 0], [3, 3, 2, 1]])

