cpdef PrepareMLData(dbs, logicalmetric, step):
	# Prepare a training set and a testing set for machine learning.
	# The value given to "step" must be one of "train" or "test".
	# The training set consists of features (for every channel) and thier repective target (fit obtained physical error rates).
	# The features we would use for every channel are its process matrix entries (except for the first column since that is trivial) and all available standard metrics.
	# The testing set simply consists of the features and the physical error rate will be predicted.
	cdef:
		int i, m, nmetrics = len(dbs.physmetrics)
		np.ndarray[np.longdouble_t, ndim = 2] mldata = np.zeros((dbs.nchannels, 12 + nmetrics + 1), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 2] phyErr = np.zeros((dbs.nchannels, nmetrics), dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 1] fitErr = np.zeros(dbs.nchannels, dtype = np.longdouble)
		np.ndarray[np.longdouble_t, ndim = 2] physical = np.zeros((4, 4), dtype = np.longdouble)
	if (step == "train"):
		fitErr = np.load(fn.BestPhysicalNoiseRates(dbs, logicalmetric))
	for m in range(nmetrics):
		phyErr[:, m] = np.load(fn.GatheredPhysErrData(dbs, dbs.physmetrics[m]))
	for i in range(dbs.nchannels):
		physical = fn.PhysicalChannel(dbs, dbs.available[i, 0], dbs.available[i, 1])
		mldata[i, :12] = physical[:, 1:].flatten
		mldata[i, 12:(12 + nmetrics)] = phyErr[i, :]
		if (step == "train"):
			mldata[i, (12 + nmetrics)] = fitErr[i]
	# Write the machine learning data set on to a file.
	if (step == "train"):
		np.save(fn.TrainingSet(dbs, logicalmetric), mldata)
	else:
		np.save(fn.TestingSet(dbs, logicalmetric), mldata[:, :(12 + nmetrics)])
	return None


cpdef Predict(dbstest, dbstrain, logicalmetric, learning = "mplr"):
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
		np.ndarray[np.longdouble_t, ndim = 2] trainset = np.load(fn.TrainingSet(dbstrain, logicalmetric))
		np.ndarray[np.longdouble_t, ndim = 2] testset = np.load(fn.TestingSet(dbstest, logicalmetric))
		np.ndarray[np.longdouble_t, ndim = 1] predictions = np.zeros(dbstest.nchannels, dtype = np.longdouble)
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
			unknown = 1
	else:
		if (learning == "nn"):
			estimator = neighbors.KNeighborsRegressor(15, weights = 'distance')
			predictor = estimator.fit(trainset[:, :-1], trainset[:, -1])
			predictions = predictor.predict(testset)
		elif (learning == "mplr"):
			estimator = MLPRegressor(activation = 'relu', solver = 'adam', early_stopping = True, verbose = True, learning_rate = 'invscaling', hidden_layer_sizes = (100, 100, 100, 100), warm_start = True)
		else:
			unknown = 1
	if (unknown == 1):
		print("Unknown learning method: %s." % (learning))
		return None
	# Save the predictions on to a file.
	np.save((-1) * predictions, fn.LearntPhysicalNoiseRates(dbstest, logicalmetric))
	print("Machine learning completed in %d seconds." % (time.time() - start))
	return None


def ValidatePrediction(dbstest, logicalmetric, stdmetric):
	# Validate a prediction by comparing the fluctuations in the logical error rate with reprect to
	# (i) The predicted physical noise rate
	# (ii) Any standard metric.
	logErr = np.load(fn.GatheredLogErrData(dbstest, logicalmetric))
	phyErr = np.load(fn.GatheredPhysErrData(dbstest, stdmetric))
	mlErr = np.load(fn.LearntPhysicalNoiseRates(dbstest, logicalmetric))
	plotfname = fn.PredictComparePlot(dbstest, logicalmetric, stdmetric)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(gv.canvas_size)
		for l in range(submit.levels):
			plt.plot(phyErr, logErr[:, l + 1], label = metlib.Metrics[stdmetric][1], color = metlib.Metrics[stdmetric][1], marker = metlib.Metrics[stdmetric][2], markersize = gv.marker_size, linestyle = 'None')
			plt.plot(mlErr, logErr[:, l + 1], label = "$\\epsilon_{\\rm predicted}$", color = metlib.Metrics[stdmetric][1], marker = metlib.Metrics[stdmetric][2], markersize = gv.marker_size, linestyle = 'None')
	# Axes labels
	ax = plt.gca()
	ax.set_xlabel("$\\mathcal{N_{0}}$", fontsize = gv.axes_labels_fontsize)
	ax.set_xscale('log')
	ax.set_ylabel(("$\\mathcal{N_{%d}}$  $\\left(%s\\right)$" % (l + 1, metlib.Metrics[logicalmetric][1]).replace("$", "")), fontsize = gv.axes_labels_fontsize)
	ax.set_yscale('log')
	# separate the axes labels from the plot-frame
	ax.tick_params(axis = 'both', which = 'major', pad = 20)
	# Legend
	lgnd = plt.legend(numpoints = 1, loc = 4, shadow = True, fontsize = legend_fontsize)
	lgnd.legendHandles[0]._legmarker.set_markersize(gv.marker_size + 20)
	# Save the plot
	pdf.savefig(fig)
	plt.close()
	#Set PDF attributes
	pdfInfo = pdf.infodict()
	pdfInfo['Title'] = ("Comparing predicted p to physical %s at levels %s, by studying fluctuations of output %s for %d channels." % (metlib.Metrics[stdmetric][0], ", ".map(str, join(range(1, 1 + dbs.levels))), metlib.Metrics[logicalmetric][0], dbs.nchannels))
	pdfInfo['Author'] = "Pavithran Iyer"
	pdfInfo['ModDate'] = dt.datetime.today()
	return None
