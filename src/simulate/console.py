def ConsoleInput(submit):
	# Accept all the simulation inputs from a console if a input file is not provided
	print("\033[93m1. Channel Type\033[0m")
	print(">>"),
	submit.Update("channel", raw_input().strip("\n").strip(" "))
	
	print("\033[93m2. Representation of the quantum channel. (Available options: \"krauss\", \"process\", \"choi\", \"chi\", \"stine\")\033[0m")
	print(">>"),
	submit.Update("repr", raw_input().strip("\n").strip(" "))

	print("\033[93m3. Noise Range (<from>, <to>, <number of points>)\n\033[2m(The noise rates are scanned on a log-scale.)\033[0m")
	print(">>"),
	submit.Update("noiserange", raw_input().strip("\n").strip(" "))
	
	print("\033[93m4. Number of samples\033[0m")
	print(">>"),
	submit.Update("samps", raw_input().strip("\n").strip(" "))
	
	print("\033[93m5. Job name (leave blank for no simulation)\033[0m")
	print(">>"),
	submit.Update("job", raw_input().strip("\n").strip(" "))
	
	if (not (submit.job == "")):	
		print("\033[93m6. Error correcting code (Available options: \"Steane\", \"FiveQubit\", \"FourQubit\", \"Cat\")\033[0m")
		print(">>"),
		submit.Update("ecc", raw_input().strip("\n").strip(" "))
		
		print("\033[93m7. Decoder to be used -- 0 for soft decoding and 1 for Hard decoding.\033[0m")
		print(">>"),
		submit.Update("decoder", raw_input().strip("\n").strip(" "))

		print("\033[93m8. Logical frame for error correction(Available options: \"[P] Pauli\", \"[C] Clifford\", \"[PC] Pauli + Logical Clifford\")\033[0m")
		print(">>"),
		submit.Update("frame", raw_input().strip("\n").strip(" "))
		
		print("\033[93m9. Number of concatenation levels\033[0m")
		print(">>"),
		submit.Update("levels", raw_input().strip("\n").strip(" "))
		
		print("\033[93m10. Number of syndromes to be sampled at top level\033[0m")
		print(">>"),
		submit.Update("stats", raw_input().strip("\n").strip(" "))
		
		print("\033[93m11. Importance sampling methods (Available options: [\"N\"] None, [\"A\"] Power law sampling, [\"B\"] Noisy channel)\033[0m")
		print(">>"),
		submit.Update("importance", raw_input().strip("\n").strip(" "))
		
		print("\033[93m12. Metrics to be computed on the effective channels at every level. (Available options: \"dnorm\", \"frb\", \"fidelity\", \"unitarity\", \"trn\")\033[0m")
		print(">>"),
		submit.Update("metrics", raw_input().strip("\n").strip(" "))
		
		print("\033[93m13. Load distribution.\033[0m")
		print(">>"),
		submit.Update("cores", raw_input().strip("\n").strip(" "))

		print("\033[93m14. Name of the host computer.\033[0m")
		print(">>"),
		submit.Update("host", raw_input().strip("\n").strip(" "))

		print("\033[93m15. Walltime in hours\033[0m")
		print(">>"),
		submit.Update("wall", raw_input().strip("\n").strip(" "))
		
		print("\033[93m16. Submission queue (Available options: see goo.gl/pTdqbV)\033[0m")
		print(">>"),
		submit.Update("queue", raw_input().strip("\n").strip(" "))
	return submit