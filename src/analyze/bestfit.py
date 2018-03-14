cpdef double Obj1(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 1] logErr, np.ndarray[np.float_t, ndim = 1] dists):
	# Objective fucntion for the different between the measured logical error rate and ansatz predicted logical error rate, on the linear scale.
	# The objective function is given by
	# g(e, l, c) = sum_(e, l, c) [ f_(e, l, c) - A(c,l) [e]^(B(c) d_(l,c)) ]
	# where e is used to label the channels
	# 		l for the levels.
	# 		c for the codes.
	cdef:
		int k, i, l
		double obj = 0.0
	for k in range(logErr.shape[0]):
		for i in range(logErr.shape[1]):
			for l in range(logErr.shape[2]):
				ansatz = optvars[nchannels + k * nlevels + l] * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l])
				obj = obj + np.power(logErr[k, i, l] - ansatz, 2.0)
	return obj

cpdef np.ndarray[np.float_t, ndim = 1] Jacob1(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 1] logErr, np.ndarray[np.float_t, ndim = 1] dists):
	# Jacobian for the objective function Obj1.
	cdef:
		int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
		double fval = 0.0
		np.ndarray[np.float_t, ndim = 1] jacob = np.zeros(optvars.shape[0], dtype = np.float)
	for k in range(logErr.shape[0]):
		for i in range(logErr.shape[1]):
			for l in range(logErr.shape[2]):
				fval = logErr[k, i, l] - optvars[nchannels + k * nlevels + l] * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l])
				# Derivatives with respect to each physical noise rate
				jacob[i] = jacob[i] + 2 * fval * (-optvars[nchannels + k * nlevels + l]) * (optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l] - 1)
				# Derivatives with respect to each level coefficient
				jacob[nchannels + k * nlevels + l] = jacob[nchannels + k * nlevels + l] + 2 * fval * (-1) * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l])
				# Derivatives with respect to each distance exponent
				jacob[nchannels + ndb * nlevels + k] = jacob[nchannels + ndb * nlevels + k] + 2 * fval * (-optvars[nchannels + k * nlevels + l]) * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * np.log(optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * dists[k, l]
	return jacob

cpdef double Obj2(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 1] logErr, np.ndarray[np.float_t, ndim = 1] dists):
	# Objective fucntion for the relative different between the measured logical error rate and ansatz predicted logical error rate, on the linear scale.
	# The objective function is given by
	# g(e, l, c) = sum_(e, l, c) [ 1 - (A(c,l) [e]^(B(c) d_(l,c)))/f_(e, l, c) ]
	# where e is used to label the channels
	# 		l for the levels.
	# 		c for the codes.
	cdef:
		int k, i, l
		double ansatz = 0.0, obj = 0.0
	for k in range(logErr.shape[0]):
		for i in range(logErr.shape[1]):
			for l in range(logErr.shape[2]):
				ansatz = optvars[nchannels + k * nlevels + l] * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l])
				obj = obj + np.power(1 - ansatz/logErr[k, i, l], 2.0)
	return obj

cpdef np.ndarray[np.float_t, ndim = 1] Jacob2(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 1] logErr, np.ndarray[np.float_t, ndim = 1] dists):
	# Jacobian for the objective function Obj2.
	cdef:
		int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
		double fval = 0.0
		np.ndarray[np.float_t, ndim = 1] jacob = np.zeros(optvars.shape[0], dtype = np.float)
	for k in range(logErr.shape[0]):
		for i in range(logErr.shape[1]):
			for l in range(logErr.shape[2]):
				fval = 1 - optvars[nchannels + k * nlevels + l] * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * 1/logErr[k, i, l]
				# Derivatives with respect to each physical noise rate
				jacob[i] = jacob[i] + 2 * fval * (-optvars[nchannels + k * nlevels + l]) * (optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l] - 1) * 1/logErr[k, i, l]
				# Derivatives with respect to each level coefficient
				jacob[nchannels + k * nlevels + l] = jacob[nchannels + k * nlevels + l] + 2 * fval * (-1) * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * 1/logErr[k, i, l]
				# Derivatives with respect to each distance exponent
				jacob[nchannels + ndb * nlevels + k] = jacob[nchannels + ndb * nlevels + k] + 2 * fval * (-optvars[nchannels + k * nlevels + l]) * np.power(optvars[i], optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * np.log(optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * dists[k, l] * 1/logErr[k, i, l]
	return jacob


cpdef double Obj3(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 1] logErr, np.ndarray[np.float_t, ndim = 1] dists):
	# Objective fucntion for the different between the measured logical error rate and ansatz predicted logical error rate, on the log scale.
	# g(p, c) = sum_(i,l,e) [ log (f_(i,l,e)) - log(c_(i,l)) - a_i * d_(i,l) * log(p_e)) ]^2
	# where i is used to label the different databases, a_i 
	#		l for the levels, c_(i,l) is the number of uncorrectable errors in database i at level l.
	# 		e for the different physical channels, p_e is the physical noise rate of the channel e.
	# 		d_(i,l) = fixed value that depends on the distance of the concatenated code.
	cdef:
		int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
		double obj = 0.0, ansatz = 0.0
	for k in range(logErr.shape[0]):
		for i in range(logErr.shape[1]):
			for l in range(logErr.shape[2]):
				ansatz = optvars[nchannels + k * nlevels + l] + optvars[nchannels + ndb * nlevels + k] * dists[k, l] * optvars[i]
				obj = obj + np.power(logErr[k, i, l] - ansatz, 2.0)
	return obj


cpdef np.ndarray[np.float_t, ndim = 1] Jacob3(np.ndarray[np.float_t, ndim = 1] optvars, np.ndarray[np.float_t, ndim = 1] logErr, np.ndarray[np.float_t, ndim = 1] dists):
	# Jacobian for the objective function Obj3.
	cdef:
		int i, k, l, ndb = logErr.shape[0], nchannels = logErr.shape[1], nlevels = logErr.shape[2]
		double ansatz = 0.0, fval = 0.0
		np.ndarray[np.float_t, ndim = 1] jacob = np.zeros(optvars.shape[0], dtype = np.float)
	for k in range(logErr.shape[0]):
		for i in range(logErr.shape[1]):
			for l in range(logErr.shape[2]):
				ansatz = optvars[nchannels + k * nlevels + l] + optvars[nchannels + ndb * nlevels + k] * dists[k, l] * optvars[i]
				fval = logErr[k, i, l] - ansatz
				# Derivatives with respect to each physical noise rate
				jacob[i] = jacob[i] + 2 * fval * (optvars[nchannels + ndb * nlevels + k] * dists[k, l]) * (-1)
				# Derivatives with respect to each level coefficient
				jacob[nchannels + k * nlevels + l] = jacob[nchannels + k * nlevels + l] + 2 * fval * (-1)
				# Derivatives with respect to each distance exponent
				jacob[nchannels + ndb * nlevels + k] = jacob[nchannels + ndb * nlevels + k] + 2 * fval * dists[k, l] * optvars[i] * (-1)
	return None


def ComputePhysicalNoiseRates(physicalmetric, logicalmetric, *dbses):
	# Compute the physical noise rates for all channels in a database.
	# The noise rates are obtained by assuming an ansatz that related the observed logical error rates to the physical noise rate by a polynomial function.
	# We then perform least square fit to obtain the unknown parameters of the ansatz.
	# See Eqs. 3 and 4 of https://arxiv.org/abs/1711.04736 .
	# The input must be one or more databases.
	# If multiple databases are provided, it will be assumed that the underlying physical channels in all of the databases are the same.
	# Optimization variable: {p for every channel} + {c for every level in every database} + {alpha for every database}
	# Create the list of logical error rates to be used by the least squares fit. Create a 3D array L where
	# L[k, i, l] = Logical error rate of the k-th database, for physical channel i and concatenated level l.
	cdef:
		int k, i, l, ndb = len(dbses), nchannels = dbses[0].nchannels, nlevels = dbses[0].nlevels
		double atol = 10E-30
		np.ndarray[np.float_t, ndim = 3] logErr = np.zeros((ndb, nchannels, nlevels), dtype = np.float)
		np.ndarray[np.float_t, ndim = 3] phyErr = np.load(fn.GatheredPhysErrRates(dbses[0], physicalmetric))
	for k in range(len(dbses)):
		logErr[k, :, :] = fn.GatheredLogErrRates(dbses[i], logicalMetric)
		for i in range(nchannels):
			for l in range(levels):
				if ((logErr[k, i, l] < (1 - atol)) and (logErr[k, i, l] > atol)):
					logErr[k, i, l] = np.log(logErr[k, i, l])
				else:
					logErr[k, i, l] = -100
	# For every dataset and concatenation level, store the distance of the code that was used to error correct.
	cdef np.ndarray[np.float_t, ndim = 2] distances = np.zeros((ndb, nlevels), dtype = np.float)
	for i in range(len(dbses)):
		for l in range(dbses[0].levels):
			distances[i, l] = dbses[i].distances[l]
	cdef:
		int var = 0
		np.ndarray[np.float_t, ndim = 1] guess = np.zeros(nchannels + len(dbses) * nlevels + len(dbses), dtype = np.float)
		np.ndarray[np.float_t, ndim = 2] limits = np.zeros((nchannels + len(dbses) * nlevels + len(dbses), 2), dtype = np.float)
	# Bounds and initial guess for physical noise rates
	for i in range(dbses[0].nchannels):
		limits[i, 0] = -5
		limits[i, 1] = 0
		guess[i] = phyErr[i]
	# Bounds and initial guess for the combinatorial factors
	var = dbses[0].nchannels
	for i in range(len(dbses)):
		for l in range(dbses.nlevels):
			limits[var, 0] = 0
			limits[var, 1] = l * np.log(dbses[i].qecc.N * (dbses[i].qecc.N - 1)/np.float(2))
			guess[var] = np.random.randint(0, high = limits[var, 1], dtype = np.float)
			var = var + 1
	# Bounds and initial guess for the exponent constants
	for i in range(len(dbses)):
		limits[var + i, 0] = 0
		limits[var + i, 1] = 2
		guess[var + i] = np.random.rand() * limits[var + i, 1]

	# Objective function and Jacobian
	objective = (lambda optvars: Objective(optvars, logErr, distances))
	jacobian = (lambda optvars: Jacobian(optvars, logErr, distances))

	cdef:
		double start = 0.0, fin = 0.0
		np.ndarray[np.float_t, ndim = 1] pfit = np.zeros(nchannels + len(dbses) * dbses[0].nlevels + len(dbses), dtype = np.float)
	#######
	start = time.time()
	result = opt.minimize(objective, guess, jac = jacobian, bounds = limits, method = 'L-BFGS-B', options = {'disp':True, 'maxiter':5000})
	pfit = result.x
	fin = time.time()
	#######

	if (result.success == True):
		print("\033[92mOptimization completed successfully in %d seconds. Objective function at minima is %.2e.\033[0m" % ((fin - start), result.fun))
	else:
		print("\033[93mOptimization terminated because of\n%s\nin %d seconds. Objective function at minima is %.2e.\033[0m" % (result.message, (fin - start), result.fun))

	# Write the computed physical noise rates into a file.
	np.save(fn.BestPhysicalNoiseRates(dbses[0], logicalmetric), pfit[:nchannels])
	for k in range(ndb):
		np.save(fn.FitWeightEnumerators(dbses[k], logicalmetric), pfit[(dbses[0].nchannels + k * nlevels):(dbses[0].nchannels + (k + 1) * nlevels)])
		np.save(fn.fitNoiseExponent(dbses[k], logicalmetric), pfit[nchannels + ndb * nlevels + k])
	return None


def CompareAnsatzToMetrics(dbs, logicalmetric, stdmetric):
	# Compare the fluctuations in the logical error rates with respect to the physical noise metrics -- obtained from fit and those computed from standard metrics
	logErr = np.load(fn.GatheredLogErrData(dbs, logicalmetric))
	phyErr = np.load(fn.GatheredPhysErrData(dbs, stdmetric))
	fitErr = np.load(fn.BestPhysicalNoiseRates(dbs, logicalmetric))
	weightenums = np.load(fn.FitWeightEnumerators(dbs, logicalmetric))
	expo = np.load(fn.fitNoiseExponent(dbs, logicalmetric))
	plotfname = fn.AnsatzComparePlot(dbs, logicalmetric, stdmetric)
	with PdfPages(plotfname) as pdf:
		fig = plt.figure(gv.canvas_size)
		for l in range(submit.levels):
			plt.plot(phyErr, logErr[:, l + 1], label = metlib.Metrics[stdmetric][1], color = metlib.Metrics[stdmetric][1], marker = metlib.Metrics[stdmetric][2], markersize = gv.marker_size, linestyle = 'None')
			plt.plot(fitErr, logErr[:, l + 1], label = ("$\\epsilon$ where $\\widetilde{\\mathcal{N}_{%d}} = %s \\times \\left[\epsilon(\\mathcal{E})\\right]^{%.2f t}$" % (l + 1, latex_float(weightenums[l + 1]), expo[0])), color = metlib.Metrics[stdmetric][1], marker = metlib.Metrics[stdmetric][2], markersize = gv.marker_size, linestyle = 'None')
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
	pdfInfo['Title'] = ("Comparing fit obtained p to physical %s at levels %s, by studying fluctuations of output %s for %d channels." % (metlib.Metrics[physicalmetric][0], ", ".map(str, join(range(1, 1 + dbs.levels))), metlib.Metrics[logicalmetric][0], dbs.nchannels))
	pdfInfo['Author'] = "Pavithran Iyer"
	pdfInfo['ModDate'] = dt.datetime.today()
	return None


def latex_float(f):
	# Function taken from: https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
	float_str = "{0:.2g}".format(f)
	if "e" in float_str:
		base, exponent = float_str.split("e")
		return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
	else:
		return float_str


