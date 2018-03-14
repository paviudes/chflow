import numpy as np
from QuantumChannels import ShortestPath

if __name__ == '__main__':
	## Test the Shortest path algorithm.
	reprs = ["krauss", "choi", "chi", "process", "stine"]
	# Mapping functions:
	# 1. Choi to process
	# 2. process to choi
	# 3. stine to krauss
	# 4. krauss to stine
	# 5. krauss to process
	# 6. krauss to choi
	# 7. choi to krauss
	# 8. process to chi
	# 9. chi to process
	# 			Krauss 	Choi 	Chi 	Process 	Stine
	# Krauss 	 0 		 6 		-1 		 5 			 4
	# Choi 	 	 7		 0 		-1 		 1 			-1
	# Chi 	 	-1		-1		 0 		 9 			-1
	# Process  	-1		 2		 8		 0 			-1
	# Stine 	 3		-1		-1		-1			 0

	mappings = np.array([[0, 6, -1, 5, 4],
						 [7, 0, -1, 1, -1],
						 [-1, -1, 0, 9, -1],
						 [-1, 2, 8, 0, -1],
						 [3, -1, -1, -1, 0]], dtype = np.int8)
	costs = np.array([[0, 1, -1, 1, 5],
					  [1, 0, -1, 1, -1],
					  [-1, -1, 0, 1, -1],
					  [-1, 1, 1, 0, -1],
					  [5, -1, -1, -1, 0]], dtype = np.int8)

	initial = "process"
	final = "krauss"
	
	map_process = ShortestPath(costs, initial, final, reprs)
	print("Mapping procedure: %s" % (" -> ".join(map_process)))

	