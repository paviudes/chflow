#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: ZeroDivisionError=False
import sys
import time
try:
	import numpy as np
	cimport numpy as np
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.linear_model import RidgeCV, ElasticNet, LassoCV
	from sklearn import neighbors
	from sklearn.feature_selection import RFECV
	from sklearn.neural_network import MLPRegressor
except:
	pass
from define import fnames as fn


cpdef PrepareMLData(dbs, physmets, lmet, step, mask):
	# Prepare a training set and a testing set for machine learning.
	# The value given to "step" must be one of "train" or "test".
	# The training set consists of features (for every channel) allowed by the mask and thier repective target (fit obtained physical error rates).
	# The features we would use for every channel are its process matrix entries (except for the first column since that is trivial) and all available standard metrics.
	# The testing set simply consists of the features and the physical error rate will be predicted.
	# mask = np.zeros((4, 4), dtype = np.int8)
	if (np.sum(mask) == 0):
		mask[:, 1:] = 1
	
	cdef:
		int i, m, nmetrics = len(physmets), nelms = np.count_nonzero(mask)
		np.ndarray[np.double_t, ndim = 2] mldata = np.zeros((dbs.channels, nelms + nmetrics + 1), dtype = np.double)
		np.ndarray[np.double_t, ndim = 2] phyerr = np.zeros((dbs.channels, nmetrics), dtype = np.double)
		np.ndarray[np.double_t, ndim = 1] fiterr = np.zeros(dbs.channels, dtype = np.double)
		np.ndarray[np.longdouble_t, ndim = 2] phychan = np.zeros((4, 4), dtype = np.longdouble)

	if (step == 0):
		fiterr = np.load(fn.FitPhysRates(dbs, lmet))
	for m in range(nmetrics):
		phyerr[:, m] = np.load(fn.PhysicalErrorRates(dbs, physmets[m]))
	for i in range(dbs.channels):
		phychan = np.load(fn.LogicalChannel(dbs, dbs.available[i, :(dbs.available.shape[1] - 1)], dbs.available[i, dbs.available.shape[1] - 1]))[0, :, :]
		mldata[i, :nelms] = phychan[np.nonzero(mask)].flatten('C')
		mldata[i, nelms:(nelms + nmetrics)] = phyerr[i, :]
		if (step == 0):
			mldata[i, (nelms + nmetrics)] = fiterr[i]
	# Write the machine learning data set on to a file.
	if (step == 0):
		np.save(fn.TrainingSet(dbs), mldata)
	else:
		np.save(fn.TestingSet(dbs), mldata[:, :(nelms + nmetrics)])
	return None


cpdef Predict(dbstest, dbstrain, learning):
	# Using one of several machine learning methods, learn a model to estimate the fit obtained physical noise rate for a set of channels.
	# Apply the learnt machine to compute a physical noise rate for every channel in a new (testiing) database.
	# The machine learning methods can be one of
	# 1. regr -- Polynomial Ridge regression with cross validation
	# 2. lasso -- Lasso regression with cross validation
	# 3. elas -- Polynomial Elastic net regression
	# 4. mplr -- Multi-layer perceptron regression algorithm using back-tracking
	# 			 http://scikit-learn.org/dev/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
	# 5. nn -- Nearest neighbours or Instance based learning
	# 		   http://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#example-neighbors-plot-regression-py
	# Load the training and testing sets.
	cdef:
		int i, m, unknown = 0, deg = 2
		double start = time.time()
		np.ndarray[np.double_t, ndim = 2] trainset = np.load(fn.TrainingSet(dbstrain))
		np.ndarray[np.double_t, ndim = 2] testset = np.load(fn.TestingSet(dbstest))
		np.ndarray[np.double_t, ndim = 1] predictions = np.zeros(dbstest.channels, dtype = np.double)
	# Obtain the polynomial form of the training features. This is not needed if we use Neural networks.
	if (not (learning in ["mplr", "nn"])):
		polytrainF = PolynomialFeatures(degree = deg).fit_transform(trainset[:, -1])
		polytestF = PolynomialFeatures(degree = deg).fit_transform(testset)
		######################################
		if (learning == "regr"):
			estimator = RidgeCV()
			selector = RFECV(estimator, n_features_to_select = None, step = 1, cv = 5, n_jobs = 4, verbose = False)
			selector.fit(polytrainF[:, :-1], trainset[:, -1])
			predictions = selector.predict(polytestF)
			np.save(fn.RFECVRankings(dbstest, deg), selector.ranking_)
		elif (learning == "lasso"):
			estimator = LassoCV()
			predictor = estimator.fit(polytrainF, polytestF)
			predictions = predictor.predict(polytestF)
		elif (learning == "elas"):
			estimator = ElasticNet(alpha = 0.5, l1_ratio = 0.1)
			predictor = estimator.fit(polytrainF, polytestF)
			predictions = predictor.predict(polytestF)
		else:
			print("\033[2mUnknown learning method.\033[0m")
			return None
	else:
		if (learning == "nn"):
			estimator = neighbors.KNeighborsRegressor(15, weights = 'distance')			
		elif (learning == "mplr"):
			estimator = MLPRegressor(activation = 'relu', solver = 'adam', early_stopping = True, verbose = True, learning_rate = 'invscaling', hidden_layer_sizes = (100, 100, 100, 100), warm_start = True)
		else:
			print("\033[2mUnknown learning method.\033[0m")
			return None
		predictor = estimator.fit(trainset[:, :(trainset.shape[1] - 1)], trainset[:, trainset.shape[1] - 1])
		predictions = predictor.predict(testset)
	if (unknown == 1):
		print("\033[2mUnknown learning method: %s.\033[0m" % (learning))
		return None
	# Save the predictions on to a file.
	np.save(fn.PredictedPhyRates(dbstest), predictions)
	print("\033[2mMachine learning completed in %d seconds.\033[0m" % (time.time() - start))
	return None