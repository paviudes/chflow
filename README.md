# chflow

[![DOI](https://zenodo.org/badge/125156309.svg)](https://zenodo.org/badge/latestdoi/125156309)

This package is intended to be used for studying the response of different quantum error correction schemes on various types of quantum noise processes. In order to gauge the response of a scheme, we perform numerical simulations of a quantum (stabilizer) error correcting scheme over a noise model and study different measures of noise strength for a qubit. As an example, the following plot produced by `chflow` describes the [fidelity](https://github.com/paviudes/chflow/wiki/Measures-of-noise-strength) of a logical qubit [error corrected](https://github.com/paviudes/chflow/wiki/Running-Simulations#running-simulations) for [coherent errors](https://github.com/paviudes/chflow/wiki/Quantum-channels#predefined-channels) \(rotations about Z-axis\) as a function of the angle of the over rotation error.
![rotz](https://github.com/paviudes/chflow/blob/master/docs/rotz.jpg)

For any stabilizer error correction scheme, `chflow` tracks the _flow of a quantum channel_ -- evolution from a physical noise process to the _effective quantum channel_ that affects the underlying logical information encompassing quantum error correction steps.

More about quantum error correction with various noise processes can be learnt from [arXiv:1711.04736](https://arxiv.org/abs/1711.04736) and [Phys. Rev. A 95, 042332](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.042332).

In addition to its main purpose, this package can also be used to perform simple but useful operations on quantum error correcting codes as well as quantum channels. For instance, derivation of the [canonical basis](https://github.com/paviudes/chflow/wiki/Quantum-error-correction#complete-description-of-a-stabilizer-code) \(Stabilizers, Logicals and Pure errors\) for an error correcting code, converting between various [representations of a quantum channel](https://github.com/paviudes/chflow/wiki/Quantum-channels#representations-for-quantum-channels), computing [Pauli approximations](https://github.com/paviudes/chflow/wiki/Quantum-channels#approximations-to-a-pauli-channel) of a quantum channel and so on.

## Downloading and installing
The latest version of `chflow` can be obtained by either [cloning this github repository](https://help.github.com/articles/cloning-a-repository/) or directly downloading the source zip file by following the ![clone](https://github.com/paviudes/chflow/blob/master/docs/clone.jpg) link in the home page.

The following dependencies, along with their recommended versions are desirable for the smooth compiling and execution of `chflow`.

| Software     	| Recommended version 	                               |
|--------------	|------------------------------------------------------|
| Python       	| [2.7](https://www.python.org/downloads/)             |
| NumPy, SciPy 	| [1.1.0](https://www.scipy.org/install.html)          |
| PICOS        	| [1.1.2](http://picos.zib.de/intro.html#installation) |
| CVXOPT       	| [1.1.9](http://cvxopt.org/install/index.html)        |
| Cython       	| [0.25.2](https://docs.anaconda.com/anaconda/install/)|

