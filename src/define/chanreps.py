try:
    from tqdm import tqdm
except:
    pass
try:
    import numpy as np
    import scipy as sp
except:
    pass
import ctypes as ct
import multiprocessing as mp
from define import fnames as fn
from define import globalvars as gv
from define import qcode as qc


def HermitianConjugate(mat):
    # Return the Hermitian conjugate of a matrix
    return np.conjugate(np.transpose(mat))


def ShortestPath(adjacency, source, target, labels):
    # Find the shortest path in a graph from a source node to a target node, given its adjacency matrix.
    # Return the path as a list of vertex labels.
    # See: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Pseudocode
    srcnode = labels.index(source)
    tarnode = labels.index(target)
    visited = np.zeros(adjacency.shape[0], dtype=np.int8)
    distances = (
        2
        * (np.max(adjacency) * adjacency.shape[0])
        * np.ones(adjacency.shape[0], dtype=np.int8)
    )
    trace = (-1) * np.ones(adjacency.shape[0], dtype=np.int8)
    distances[srcnode] = 0
    while np.prod(visited, dtype=np.int8) == 0:
        # select the unvisited node with the minimum distance from source
        nextvert = 0
        mindist = 2 * (np.max(adjacency) * adjacency.shape[0])
        for i in range(adjacency.shape[0]):
            if visited[i] == 0:
                if distances[i] < mindist:
                    mindist = distances[i]
                    nextvert = i
        visited[nextvert] = 1
        if nextvert == tarnode:
            break
        # For each unvisited neighbour of nextvert, update its distance from source as mindist + dist(vertex, nextvert)
        for i in range(adjacency.shape[0]):
            if adjacency[nextvert, i] > 0:
                if visited[i] == 0:
                    alternate = distances[nextvert] + adjacency[nextvert, i]
                    if alternate < distances[i]:
                        distances[i] = alternate
                        trace[i] = nextvert
    # Back trace to find out the path
    sequence = []
    vert = tarnode
    while trace[vert] > -1:
        sequence.append(labels[vert])
        vert = trace[vert]
    sequence.append(source)
    sequence = sequence[::-1]
    return sequence


def ConvertRepresentations(channel, initial, final):
    # Convert between different representations of a quantum channel
    gv.Pauli = np.array(
        [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=np.complex128,
    )

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
    # 10. choi to chi
    # 			Krauss 	Choi 	Chi 	Process 	Stine
    # Krauss 	 0 		 6 		-1 		 5 			 4
    # Choi 	 	 7		 0 		 10 	 1 			-1
    # Chi 	 	-1		-1		 0 		 9 			-1
    # Process  	-1		 2		 8		 0 			-1
    # Stine 	 3		-1		-1		-1			 0

    mappings = np.array(
        [
            [0, 6, -1, 5, 4],
            [7, 0, 10, 1, -1],
            [-1, -1, 0, 9, -1],
            [-1, 2, 8, 0, -1],
            [3, -1, -1, -1, 0],
        ],
        dtype=np.int8,
    )
    costs = np.array(
        [
            [0, 1, -1, 1, 5],
            [1, 0, 1, 1, -1],
            [-1, -1, 0, 1, -1],
            [-1, 1, 1, 0, -1],
            [5, -1, -1, -1, 0],
        ],
        dtype=np.int8,
    )

    map_process = ShortestPath(costs, initial, final, reprs)
    outrep = np.copy(channel)

    for i in range(len(map_process) - 1):
        initial = map_process[i]
        final = map_process[i + 1]
        inprep = np.copy(outrep)

        if initial == "choi" and final == "process":
            # Convert from the Choi matrix to the process matrix, of a quantum channel
            # CHI[a,b] = Trace( Choi * (Pb \otimes Pa^T) )
            process = np.zeros((4, 4), dtype=np.longdouble)
            for pa in range(4):
                for pb in range(4):
                    process[pa, pb] = np.real(
                        np.trace(
                            np.dot(
                                inprep,
                                np.kron(
                                    gv.Pauli[pb, :, :], np.transpose(gv.Pauli[pa, :, :])
                                ),
                            )
                        )
                    )
            outrep = np.copy(process)

        elif initial == "process" and final == "choi":
            # Convert from the process matrix representation to the Choi matrix represenation, of a quantum channel
            choi = np.zeros((4, 4), dtype=np.complex128)
            for ri in range(4):
                for ci in range(4):
                    choi = choi + inprep[ri, ci] * np.kron(
                        gv.Pauli[ci, :, :], np.transpose(gv.Pauli[ri, :, :])
                    )
            choi = choi / np.complex128(4)
            outrep = np.copy(choi)

        elif initial == "stine" and final == "krauss":
            # Compute the Krauss operators for the input quantum channel, which is represented in the Stinespring dialation
            # The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0> , where {|e_i>} is a basis for the environment and |a>, |b> are basis vectors of the system
            environment = np.zeros((4, 4, 1), dtype=int)
            for bi in range(4):
                environment[bi, :, :] = np.eye(4)[:, bi, np.newaxis]
            system = np.zeros((2, 2, 1), dtype=int)
            for bi in range(2):
                system[bi, :, :] = np.eye(2)[:, bi, np.newaxis]
            krauss = np.zeros((4, 2, 2), dtype=np.complex128)
            for ki in range(4):
                ## The Krauss operator T_k is given by: <a|T_k|b> = <a e_k|U|b e_0>.
                for ri in range(2):
                    for ci in range(2):
                        leftProduct = HermitianConjugate(
                            np.dot(
                                inprep, np.kron(system[ri, :, :], environment[ki, :, :])
                            )
                        )
                        krauss[ki, ri, ci] = np.dot(
                            leftProduct, np.kron(system[ci, :, :], environment[0, :, :])
                        )[0, 0]
            outrep = np.copy(krauss)

        elif initial == "krauss" and final == "stine":
            # Compute the Stinespring dialation of the input Krauss operators.
            # The Stinespring dialation is defined only up to a fixed choice of the initial state of the environment.
            # We will consider the size of the environment to be 2 qubits. Hence there must be 4 Krauss operartors.
            # If there are less than 4, we will pad additional Krauss operators with zeros.
            # U[phi, j1, j2][psi, 0, 0] = <phi|K_(j1,j2)|psi>
            stineU = np.zeros((8, 8), dtype=np.complex128)
            for phi in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        for psi in range(2):
                            stineU[
                                phi * 2 ** 2 + j1 * 2 ** 1 + j2, psi * 2 ** 2
                            ] = inprep[j1 * 2 + j2, phi, psi]
            outrep = np.copy(stineU)

        elif initial == "krauss" and final == "process":
            # Convert from the Krauss representation to the Process matrix representation
            ## In particular, Process[i,j] = 1/2 * trace( E(Pi) Pj)
            process = np.zeros((4, 4), dtype=np.longdouble)
            for pi in range(4):
                for pj in range(4):
                    element = 0 + 0 * 1j
                    for ki in range(inprep.shape[0]):
                        element = element + np.trace(
                            np.dot(
                                np.dot(
                                    np.dot(inprep[ki, :, :], gv.Pauli[pi, :, :]),
                                    HermitianConjugate(inprep[ki, :, :]),
                                ),
                                gv.Pauli[pj, :, :],
                            )
                        )
                    process[pi, pj] = 1 / np.longdouble(2) * np.real(element)
            # forcing the channel to be trace preserving.
            outrep = np.copy(process / process[0, 0])

        elif initial == "krauss" and final == "choi":
            # Convert from the Krauss operator representation to the Choi matrix of a quantum channel
            choi = np.zeros((4, 4), dtype=np.complex128)
            for k in range(inprep.shape[0]):
                choi = choi + np.dot(
                    np.kron(inprep[k, :, :], np.eye(2)),
                    np.dot(
                        gv.bell[0, :, :],
                        HermitianConjugate(np.kron(inprep[k, :, :], np.eye(2))),
                    ),
                )
            outrep = np.copy(choi)

        elif initial == "choi" and final == "krauss":
            # Convert from the Choi matrix to the Krauss representation of a quantum channel.
            # Compute the eigenvalues and the eigen vectors of the Choi matrix. The eigen vectors operators are vectorized forms of the Krauss operators.
            (eigvals, eigvecs) = np.linalg.eig(inprep.astype(np.complex128))
            krauss = np.zeros((4, 2, 2), dtype=np.complex128)
            for i in range(4):
                krauss[i, :, :] = np.sqrt(2 * eigvals[i]) * np.reshape(
                    eigvecs[:, i], [2, 2], order="F"
                )
            outrep = np.copy(krauss)

        elif initial == "process" and final == "chi":
            # Convert from the process matrix to the chi matrix
            # The process matrix is the action of the channel on the Pauli basis whereas the Chi matrix describes the amplitude of applying a pair of gv.Pauli operators (on the left and right) to the input state
            # Lambda_ij = \sum_(k,l) W_ijkl * Chi_(k,l)
            # where W_ijkl = Trace(P_k P_i P_l P_j)
            chi = np.reshape(
                np.dot(gv.process_to_chi, np.reshape(inprep, [16, 1])), [4, 4]
            )
            outrep = np.copy(chi)

        elif initial == "chi" and final == "process":
            # Convert from the chi matrix to the process matrix
            # The process matrix is the action of the channel on the Pauli basis whereas the Chi matrix describes the amplitude of applying a pair of Pauli operators (on the left and right) to the input state
            # Lambda_ij = \sum_(k,l) W_ijkl * Chi_(k,l)
            # where W_ijkl = Trace(P_k P_i P_l P_j)
            process = np.reshape(
                np.dot(
                    np.reshape(gv.chi_to_process, [16, 16]), np.reshape(inprep, [16, 1])
                ),
                [4, 4],
            )
            outrep = np.copy(process)

        elif initial == "choi" and final == "chi":
            # Convert from the Choi matrix to the Chi matrix
            # Tr(J(E) . (Pa o Pb^T)) = 1/2 \sum_(ij) X_(ij) W_((ij)(ab))
            # where W_(ijab) = 1/2 * Tr(Pi Pb Pj Pa).
            # Let v_(ab) = Tr(J(E) . (Pa o Pb^T)), then we have
            # v_(ab) = X_(ij) W_((ij)(ab)). which is the relation: <v| = <x|W.
            # We can rewrite this as: <x| = <v|W^{-1}.
            choivec = np.zeros((1, 16), dtype=np.complex128)
            for a in range(4):
                for b in range(4):
                    choivec[0, a * 4 + b] = np.trace(
                        np.dot(
                            inprep,
                            np.kron(gv.Pauli[a, :, :], np.transpose(gv.Pauli[b, :, :])),
                        )
                    )
            ####
            # A.CHIVEC = CHOIVEC
            ####
            chi = np.reshape(np.dot(choivec, gv.choi_to_chi), [4, 4])
            outrep = np.copy(chi)

        else:
            sys.stderr.write("\033[91mUnknown conversion task.\n\033[0m")

    return outrep


def PauliConvertToTransfer(pauliprobs, qcode):
    r"""
    Convert from a representation of a Pauli channel as a distribution over various Pauli errors, to that of a Pauli transfer matrix.
    The ordering of errors in the distribution is assumed to be TLS.
    """
    # print("probs = {}".format(pauliprobs.shape))
    nstabs = 2 ** (qcode.N - qcode.K)
    nlogs = 4 ** qcode.K
    # ptm = np.zeros(nstabs * nlogs, dtype=np.double)
    ptm = mp.Array(ct.c_double, np.zeros(nstabs * nlogs, dtype=np.double))
    # processes = []
    ##################
    for l in range(nlogs):
        # processes.append(
        #     mp.Process(
        #         target=GetTransferMatrixElements, args=(l, pauliprobs, qcode, ptm)
        #     )
        # )
        GetTransferMatrixElements(l, pauliprobs, qcode, ptm)
    # for l in range(nlogs):
    #     processes[l].start()
    # for l in range(nlogs):
    #     processes[l].join()
    return np.array(ptm, dtype=np.double)


def GetTransferMatrixElements(logidx, pauliprobs, qcode, ptm):
    r"""
    Compute the Pauli transfer matrix entries of operators L.S where S runs over all stabilizers and L is a given logical operator.
    """
    ordering = np.array([[0, 3], [1, 2]], dtype=np.int8)
    nstabs = 2 ** (qcode.N - qcode.K)
    log_select = np.array(
        list(map(np.int8, np.binary_repr(logidx, width=2 * qcode.K))), dtype=np.int8
    )
    if logidx == 0:
        log_op = np.zeros(qcode.N, dtype=np.int)
    else:
        (log_op, __) = qc.PauliProduct(*qcode.L[np.nonzero(log_select)])
    # for s in tqdm(range(nstabs), ascii=True, desc="\033[2mGoing over Stabilizers:"):
    for s in range(nstabs):
        if s == 0:
            stab_op = np.zeros(qcode.N, dtype=np.int)
        else:
            stab_select = np.array(
                list(map(np.int8, np.binary_repr(s, width=qcode.N - qcode.K))),
                dtype=np.int8,
            )
            (stab_op, __) = qc.PauliProduct(*qcode.S[np.nonzero(stab_select)])
        indices = qc.GetCommuting(log_op, stab_op, qcode.L, qcode.S, qcode.T)
        # print(
        #     "commuting indices: {}\nanti commuting: {}".format(
        #         indices["commuting"], indices["anticommuting"]
        #     )
        # )
        if len(indices["commuting"]) > 0:
            commuting_sum = np.sum(pauliprobs[indices["commuting"]])
        else:
            commuting_sum = 0
        if len(indices["anticommuting"]) > 0:
            anticommuting_sum = np.sum(pauliprobs[indices["anticommuting"]])
        else:
            anticommuting_sum = 0

        ptm[ordering[log_select[0], log_select[1]] * nstabs + s] = (
            commuting_sum - anticommuting_sum
        )
    # print("\033[0m")
    return None


def CreatePauliDistChannels(submit):
    """
    Create the Chi matrix for all channels in the database.
    """
    nlogs = 4 ** (submit.eccs[0].K)
    chan_dim = nlogs * nlogs
    if submit.iscorr == 0:
        nparams = chan_dim
    elif submit.iscorr == 1:
        print("Twirling is not set up for fully correlated channels.")
        return None
    else:
        nparams = submit.eccs[0].N * chan_dim
    rawchans = np.zeros(
        (submit.noiserates.shape[0], submit.samps, nparams), dtype=np.double
    )
    for i in range(submit.noiserates.shape[0]):
        chans = np.load(fn.PhysicalChannel(submit, submit.noiserates[i, :]))
        for j in range(submit.samps):
            if submit.iscorr == 0:
                submit.rawchans[i, j, :] = np.real(
                    ConvertRepresentations(
                        (chans[j, :] * np.eye(nlogs, dtype=np.int).ravel()).reshape(
                            [nlogs, nlogs]
                        ),
                        "process",
                        "chi",
                    )
                ).ravel()
            elif submit.iscorr == 2:
                for q in range(submit.eccs[0].N):
                    chan_qubit = chans[j, q * chan_dim : (q + 1) * chan_dim]
                    submit.rawchans[i, j, q * chan_dim : (q + 1) * chan_dim] = np.real(
                        ConvertRepresentations(
                            (chan_qubit * np.eye(nlogs, dtype=np.int).ravel()).reshape(
                                [nlogs, nlogs]
                            ),
                            "process",
                            "chi",
                        )
                    ).ravel()
            else:
                pass
        np.save(
            fn.RawPhysicalChannel(submit, submit.noiserates[i, :]),
            submit.rawchans[i, :, :],
        )
    return None


def TwirlChannels(submit):
    r"""
    Twirl all the channels in a database, in place.
    """
    if submit.iscorr == 0:
        nparams = 4 ** (2 * submit.eccs[0].K)
    elif submit.iscorr == 1:
        print("Twirling is not set up for fully correlated channels.")
        return None
    else:
        nparams = submit.eccs[0].N * 4 ** (2 * submit.eccs[0].K)

    submit.phychans = np.zeros(
        (submit.noiserates.shape[0], submit.samps, nparams), dtype=np.double
    )
    for i in range(submit.noiserates.shape[0]):
        chans = np.load(fn.PhysicalChannel(submit, submit.noiserates[i, :]))
        for j in range(submit.samps):
            if submit.iscorr == 0:
                submit.phychans[i, j, :] = (
                    chans[j, :] * np.eye(4 ** submit.eccs[0].K, dtype=np.int).ravel()
                )
            else:
                submit.phychans[i, j, :] = (
                    chans[j, :]
                    * np.tile(
                        np.eye(4 ** submit.eccs[0].K, dtype=np.int),
                        (submit.eccs[0].N, 1, 1),
                    ).ravel()
                )
    submit.misc = "This is the Twirl of %s" % (submit.timestamp)
    submit.plotsettings["name"] = "With Randomized compiling"
    return None
