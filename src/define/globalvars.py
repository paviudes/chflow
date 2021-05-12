import numpy as np
import itertools as it


def Dot(matrices):
    # perform a dot product of matrices in a list, from left to right.
    if matrices.shape[0] == 1:
        return matrices[0]
    else:
        return np.dot(matrices[0], Dot(matrices[1:]))


# Computational basis
kets = np.array([[[1], [0]], [[0], [1]]])
bras = np.array([[[1, 0]], [[0, 1]]])

# Pauli matrices
Pauli = np.array(
    [[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
    dtype=np.complex128,
)

paulibasis = np.zeros((np.power(4, 3, dtype=int), 8, 8), dtype=np.complex128)
combinations = it.product(range(4), repeat=3)
for (i, comb) in enumerate(combinations):
    paulibasis[i, :, :] = np.kron(
        np.kron(Pauli[comb[0]], Pauli[comb[1]]), Pauli[comb[2]]
    )

# Basis transformations for converting between channel representations
# Converting from the Chi matrix to the Choi matrix
chi_to_choi = np.zeros((4, 4, 4, 4), dtype=np.complex128)
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                chi_to_choi[i, j, k, l] = 0.5 * np.trace(Dot(Pauli[[i, l, j, k], :, :]))
choi_to_chi = np.linalg.inv(np.reshape(chi_to_choi, [16, 16]))

# Converting from the Chi matrix to the Pauli Liouville matrix
chi_to_process = np.zeros((4, 4, 4, 4), dtype=np.complex128)
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                chi_to_process[i, j, k, l] = 0.5 * np.trace(
                    Dot(Pauli[[k, i, l, j], :, :])
                )
process_to_chi = np.linalg.inv(np.reshape(chi_to_process, [16, 16]))

# Bell states
bell = np.zeros([4, 4, 4], dtype=np.float)
# Bell state |00> + |11>
bell[0, :, :] = np.dot(
    (np.kron(kets[0], kets[0]) + np.kron(kets[1], kets[1])),
    (np.kron(bras[0], bras[0]) + np.kron(bras[1], bras[1])),
) / np.float(2)
# Bell state |01> + |10>
bell[1, :, :] = np.dot(
    (np.kron(kets[0], kets[1]) + np.kron(kets[1], kets[0])),
    (np.kron(bras[0], bras[1]) + np.kron(bras[1], bras[0])),
) / np.float(2)
# Bell state i(|10> - |01>)
bell[2, :, :] = np.dot(
    (np.kron(kets[0], kets[1]) - np.kron(kets[1], kets[0])),
    (np.kron(bras[0], bras[1]) - np.kron(bras[1], bras[0])),
) / np.float(2)
# Bell state |00> - |11>
bell[3, :, :] = np.dot(
    (np.kron(kets[0], kets[0]) - np.kron(kets[1], kets[1])),
    (np.kron(bras[0], bras[0]) - np.kron(bras[1], bras[1])),
) / np.float(2)

# Hadamard gate
hadamard = np.array(
    [
        [1 / np.sqrt(np.longdouble(2)), 1 / np.sqrt(np.longdouble(2))],
        [1 / np.sqrt(np.longdouble(2)), -1 / np.sqrt(np.longdouble(2))],
    ],
    dtype=np.complex128,
)

## Plot settings
# linestyles from https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html
line_styles = [(0, ()), (0, (5, 5)), (0, (1, 1)), (0, (3, 5, 1, 5, 1, 5))]
n_line_styles = len(line_styles)
# Markers
# https://matplotlib.org/2.0.2/api/markers_api.html
Markers = [
    "o",  # Without RC, choose o for predictability plots
    "o",  # With RC, choose o for predictability plots
    "+",
    "*",  # Diamond norm (predictability)
    "^",  # Infidelity (predictability)
    "v",
    "<",
    ">",
    "8",
    "s",
    "p",
    "h",
    "H",
    "D",
    "d",
    "P",
    "X",
]
n_Markers = len(Markers)
# Colors
# https://stackoverflow.com/questions/16006572/plotting-different-colors-in-matplotlib
QB_GREEN = "#42b863"
QB_BLUE = "#2697d0"
Colors = [
    "blue",  # Without RC, choose 0.65 for predictability plots
    "red", # With RC, choose red for predictability plots
    "blue",
    "brown",
    "QB_BLUE",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
n_Colors = len(Colors)
# Frame
title_fontsize = 48
canvas_size = (32, 22)
# Axes
axes_labels_fontsize = 58
ticks_fontsize = 42
ticks_length = 12
ticks_width = 4
ticks_pad = 40
axes_labelpad = 20
# Curve
line_width = 5
marker_size = 20
# Legend
legend_fontsize = 42
legend_marker_scale = 2
# Contour plot
contour_nlevs = 10
contour_linewidth = 2
contour_linestyle = "solid"
# Color bar
colorbar_fontsize = 48

# Cluster hosts
cluster_info = {"graham": 32, "cedar": 48, "niagara": 24, "beluga": 40}
