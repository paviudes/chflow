import numpy as np
import globalvars as gv

def HermitianConjugate(mat):
	# Return the Hermitian conjugate of a matrix
	return np.conjugate(np.transpose(mat))


def Dot(matrices):
	# perform a dot product of matrices in a list, from left to right.	
	if (matrices.shape[0] == 1):
		return matrices[0]
	else:
		return np.dot(matrices[0], Dot(matrices[1:]))


def ShortestPath(adjacency, source, target, labels):
	# Find the shortest path in a graph from a source node to a target node, given its adjacency matrix.
	# Return the path as a list of vertex labels.
	# See: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Pseudocode
	srcnode = labels.index(source)
	tarnode = labels.index(target)
	visited = np.zeros(adjacency.shape[0], dtype = np.int8)
	distances = 2 * (np.max(adjacency) * adjacency.shape[0]) * np.ones(adjacency.shape[0], dtype = np.int8)
	trace = (-1) * np.ones(adjacency.shape[0], dtype = np.int8)
	distances[srcnode] = 0
	while (np.prod(visited, dtype = np.int8) == 0):
		# select the unvisited node with the minimum distance from source
		nextvert = 0
		mindist = 2 * (np.max(adjacency) * adjacency.shape[0])
		for i in range(adjacency.shape[0]):
			if (visited[i] == 0):
				if (distances[i] < mindist):
					mindist = distances[i]
					nextvert = i
		visited[nextvert] = 1
		if (nextvert == tarnode):
			break
		# For each unvisited neighbour of nextvert, update its distance from source as mindist + dist(vertex, nextvert)
		for i in range(adjacency.shape[0]):
			if (adjacency[nextvert, i] > 0):
				if (visited[i] == 0):
					alternate = distances[nextvert] + adjacency[nextvert, i]
					if (alternate < distances[i]):
						distances[i] = alternate
						trace[i] = nextvert
	# Back trace to find out the path
	sequence = []
	vert = tarnode
	while (trace[vert] > -1):
		sequence.append(labels[vert])
		vert = trace[vert]
	sequence.append(source)
	sequence = sequence[::-1]
	return sequence


def ConvertRepresentations(channel, initial, final):
	# Convert between different representations of a quantum channel
	gv.Pauli = np.array([[[1, 0], [0, 1]],
					  [[0, 1], [1, 0]],
					  [[0, -1j], [1j, 0]],
					  [[1, 0], [0, -1]]], dtype = np.complex128)

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

	map_process = ShortestPath(costs, initial, final, reprs)
	# print("Input\n%s" % (np.array_str(channel, max_line_width = 150, precision = 3)))
	# print("\033[2mMapping procedure: %s\033[0m" % (" -> ".join(map_process)))
	outrep = np.copy(channel)
	
	for i in range(len(map_process) - 1):
		initial = map_process[i]
		final = map_process[i + 1]
		inprep = np.copy(outrep)
	
		if (initial == 'choi' and final == 'process'):
			# Convert from the Choi matrix to the process matrix, of a quantum channel
			# CHI[a,b] = Trace( Choi * (Pb \otimes Pa^T) )
			process = np.zeros((4,4), dtype = np.longdouble)
			for pa in range(4):
				for pb in range(4):
					process[pa, pb] = np.real(np.trace(np.dot(inprep, np.kron(gv.Pauli[pb, :, :], np.transpose(gv.Pauli[pa, :, :])))))
			outrep = np.copy(process)

		elif (initial == 'process' and final == 'choi'):
			# Convert from the process matrix representation to the Choi matrix represenation, of a quantum channel
			choi = np.zeros((4, 4), dtype = np.complex128)
			for ri in range(4):
				for ci in range(4):
					choi = choi + inprep[ri, ci] * np.kron(gv.Pauli[ci, :, :], np.transpose(gv.Pauli[ri, :, :]))
			choi = choi/np.complex128(4)
			outrep = np.copy(choi)

		elif (initial == 'stine' and final == 'krauss'):
			# Compute the Krauss operators for the input quantum channel, which is represented in the Stinespring dialation
			# The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0> , where {|e_i>} is a basis for the environment and |a>, |b> are basis vectors of the system
			environment = np.zeros((4, 4, 1), dtype = int)
			for bi in range(4):
				environment[bi, :, :] = np.eye(4)[:, bi, np.newaxis]
			system = np.zeros((2, 2, 1), dtype = int)
			for bi in range(2):
				system[bi, :, :] = np.eye(2)[:, bi, np.newaxis]
			krauss = np.zeros((4, 2, 2), dtype = np.complex128)
			for ki in range(4):
				## The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0>.
				for ri in range(2):
					for ci in range(2):
						leftProduct = HermitianConjugate(np.dot(inprep, np.kron(system[ri, :, :], environment[ki, :, :])))
						krauss[ki, ri, ci] = np.dot(leftProduct, np.kron(system[ci, :, :], environment[0, :, :]))[0, 0]
			outrep = np.copy(krauss)

		elif (initial == 'krauss' and final == 'stine'):
			# Compute the Stinespring dialation of the input Krauss operators.
			# The Stinespring dialation is defined only up to a fixed choice of the initial state of the environment.
			# We will consider the size of the environment to be 2 qubits. Hence there must be 4 Krauss operartors.
			# If there are less than 4, we will pad additional Krauss operators with zeros.
			# U[phi, j1, j2][psi, 0, 0] = <phi|K_(j1,j2)|psi>
			stineU = np.zeros((8, 8), dtype = np.complex128)
			for phi in range(2):
				for j1 in range(2):
					for j2 in range(2):
						for psi in range(2):
							stineU[phi * 2**2 + j1 * 2**1 + j2, psi * 2**2] = inprep[j1 * 2 + j2, phi, psi]
			outrep = np.copy(stineU)


		elif (initial == 'krauss' and final == 'process'):
			# Convert from the Krauss representation to the Process matrix representation
			## In particular, Process[i,j] = 1/2 * trace( E(Pi) Pj)	
			process = np.zeros((4, 4), dtype = np.longdouble)
			for pi in range(4):
				for pj in range(4):
					element = 0 + 0 * 1j
					for ki in range(inprep.shape[0]):
						element = element + np.trace(np.dot(np.dot(np.dot(inprep[ki, :, :], gv.Pauli[pi, :, :]), HermitianConjugate(inprep[ki, :, :])), gv.Pauli[pj, :, :]))
					process[pi, pj] = 1/np.longdouble(2) * np.real(element)
			# forcing the channel to be trace preserving.
			outrep = np.copy(process/process[0, 0])


		elif (initial == 'krauss' and final == 'choi'):
			# Convert from the Krauss operator representation to the Choi matrix of a quantum channel
			choi = np.zeros((4, 4), dtype = np.complex128)
			for k in range(inprep.shape[0]):
				choi = choi + np.dot(np.kron(inprep[k, :, :], np.eye(2)), np.dot(gv.bell[0, :, :], HermitianConjugate(np.kron(inprep[k, :, :], np.eye(2)))))
			outrep = np.copy(choi)


		elif (initial == 'choi' and final == 'krauss'):
			# Convert from the Choi matrix to the Krauss representation of a quantum channel.
			# Compute the eigenvalues and the eigen vectors of the Choi matrix. The eigen vectors operators are vectorized forms of the Krauss operators.
			(eigvals, eigvecs) = np.linalg.eig(inprep.astype(np.complex128))
			krauss = np.zeros((4, 2, 2), dtype = np.complex128)
			for i in range(4):
				krauss[i, :, :] = np.sqrt(2 * eigvals[i]) * np.reshape(eigvecs[:, i], [2, 2], order = 'F')
			outrep = np.copy(krauss)
		
		elif (initial == 'process' and final == 'chi'):
			# Convert from the process matrix representation to the Chi matrix representation
			# The process matrix is the action of the channel on the Pauli basis whereas the Chi matrix describes the amplitude of applying a pair of gv.Pauli operators (on the left and right) to the input state
			# X_ij = \sum_(k,l) W_ijkl * Lambda_(k,l)
			# where W_ijkl = Trace(P_k P_j P_l P_i)
			basis = np.zeros((4, 4, 4, 4), dtype = np.complex128)
			for i in range(4):
				for j in range(4):
					for k in range(4):
						for l in range(4):
							basis[i, j, k, l] = 0.5 * np.trace(Dot(gv.Pauli[[k, j, l, i], :, :]))
			chi = np.reshape(np.dot(np.linalg.inv(np.reshape(basis, [16, 16])), np.reshape(inprep, [16, 1])), [4, 4])
			outrep = np.copy(chi)

		elif (initial == 'chi' and final == 'process'):
			# Convert from the process matrix to the Chi matrix
			# The process matrix is the action of the channel on the Pauli basis whereas the Chi matrix describes the amplitude of applying a pair of Pauli operators (on the left and right) to the input state
			# X_ij = \sum_(k,l) W_ijkl * Lambda_(k,l)
			# where W_ijkl = Trace(P_k P_j P_l P_i)
			basis = np.zeros((4, 4, 4, 4), dtype = np.complex128)
			for i in range(4):
				for j in range(4):
					for k in range(4):
						for l in range(4):
							basis[i, j, k, l] = 0.5 * np.trace(Dot(gv.Pauli[[k, j, l, i], :, :]))
			process = np.reshape(np.dot(np.reshape(basis, [16, 16]), np.reshape(inprep, [16, 1])), [4, 4])
			outrep = np.copy(process)
		else:
			sys.stderr.write("\033[91mUnknown conversion task.\n\033[0m")
		
	return outrep
