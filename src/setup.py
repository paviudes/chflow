def CheckDependencies():
	# Check if all the requires packages exist
	# if not, create a requirements text file.
	packages = [["adjustText", "1.0", "Fancy annotations", "Mild"],
				["timeit", "3.5", "Time operations", "Critical"],
				["readline", "3.5", "Console input", "Mild"],
				["scipy", "0.18", "Numerical operations", "Critical"],
				["numpy", "0.12", "Numerical operations", "Critical"],
				["picos", "1.0", "semi-definite programs", "Mild"],
				["cvxopt", "1.0", "semi-definite programs", "Mild"],
				["cvxpy", "1.1", "Integer and non-linear programs", "Mild"],
				["multiprocessing", "1.0", "parallel computations", "Critical"],
				["sklearn", "1.0", "machine learning", "Mild"],
				["matplotlib", "1.0", "plotting", "Critical"],
				["tqdm", "4.31.1", "progress bar", "Critical"],
				["pqdm", "0.2.0", "parallel progress bar", "Critical"],
				["ncon", "2.0.0", "Tensor contraction using Matlab style ncon", "Critical"],
				["cotengra", "0.5.6", "Tensor contraction using Cotengra", "Mild"],
				["psutil", "5.8.0", "get system information", "Critical"],
				["ctypes", "3.3", "importing C functions", "Critical"],
				["datetime", "3.2", "Fetching date and time", "Mild"],
				["PyPDF2", "1.26.0", "Manipulating PDF files", "Mild"],
				["tensorflow", "2.4.0", "Tensor contraction", "Mild"]]
	missing = []
	for i in range(len(packages)):
		try:
			exec("import %s" % (packages[i][0]))
		except:
			missing.append(packages[i])
	
	if (len(missing) > 0):
		print("\033[33m", end="")
		print("Missing or outdated packages might affect certain functionalities.")
		print("{:<10} | {:<10} | {:<30} | {:<10}".format("Package", "Version", "Affected functionality", "Impact"))
		print("{:<50}".format("---------------------------------------------------------------------"))
		
		is_critical = 0
		for i in range(len(missing)):
			if ("Critical" in missing[i][3]):
				is_critical = 1
				print("{:<10} | {:<10} | {:<30} | \033[7;31m{:<10}\033[0;33m".format(missing[i][0], missing[i][1], missing[i][2], missing[i][3]))
			else:
				print("{:<10} | {:<10} | {:<30} | \033[32m{:<10}\033[33m".format(missing[i][0], missing[i][1], missing[i][2], missing[i][3]))
		print("xxxxxx")
		print("\033[0m", end="")
		
		with open("./../requirements.txt", "w") as fp:
			fp.write("# Install the missing packages using pip install -r requirements.txt\n")
			for i in range(len(missing)):
				fp.write("%s>=%s\n" % (missing[i][0], missing[i][1]))
		print("\033[7;32mTo install all missing packages, run \"pip install -r requirements.txt\".\033[0m")

		if (is_critical):
			print("\033[0;31mExiting due to missing critical packages ...\033[0m")
			exit(0)
	return None