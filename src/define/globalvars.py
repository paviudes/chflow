import numpy as np

# Computational basis
kets = np.array([[[1],[0]], [[0],[1]]])
bras = np.array([[[1, 0]], [[0,1]]])

# Pauli matrices
Pauli = np.array([[[1, 0], [0, 1]],
				  [[0, 1], [1, 0]],
				  [[0, -1j], [1j, 0]],
				  [[1, 0], [0, -1]]], dtype = np.complex128)

# Bell states
bell = np.zeros([4,4,4], dtype = np.float)
# Bell state |00> + |11>
bell[0, :, :] = np.dot((np.kron(kets[0], kets[0]) + np.kron(kets[1], kets[1])), (np.kron(bras[0], bras[0]) + np.kron(bras[1], bras[1])))/np.float(2)
# Bell state |01> + |10>
bell[1, :, :] = np.dot((np.kron(kets[0], kets[1]) + np.kron(kets[1], kets[0])), (np.kron(bras[0], bras[1]) + np.kron(bras[1], bras[0])))/np.float(2)
# Bell state i(|10> - |01>)
bell[2, :, :] = np.dot((np.kron(kets[0], kets[1]) - np.kron(kets[1], kets[0])), (np.kron(bras[0], bras[1]) - np.kron(bras[1], bras[0])))/np.float(2)
# Bell state |00> - |11>
bell[3, :, :] = np.dot((np.kron(kets[0], kets[0]) - np.kron(kets[1], kets[1])), (np.kron(bras[0], bras[0]) - np.kron(bras[1], bras[1])))/np.float(2)

# Hadamard gate
hadamard = np.array([[1/np.sqrt(np.longdouble(2)), 1/np.sqrt(np.longdouble(2))], [1/np.sqrt(np.longdouble(2)), -1/np.sqrt(np.longdouble(2))]], dtype = np.complex128)

## Plot settings
# Frame
title_fontsize = 48
canvas_size = (32, 22)
# Axes
axes_labels_fontsize = 48
ticks_fontsize = 42
ticks_length = 12
ticks_width = 4
ticks_pad = 40
# Curve
line_width = 5
marker_size = 15
# Legend
legend_fontsize = 42
legend_marker_scale = 2
# Contour plot
contour_nlevs = 10
contour_linewidth = 2
contour_linestyle = "solid"
# Color bar
colorbar_fontsize = 48