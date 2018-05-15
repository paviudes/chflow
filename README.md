# chflow -- Quantum error correction for realistic noise

[![DOI](https://zenodo.org/badge/125156309.svg)](https://zenodo.org/badge/latestdoi/125156309)

This package is intended to be used for studying the response of different quantum error correction schemes on various types of quantum noise processes. Here, numerical simulations can be peformed for any quantum (stabilizer) error correcting scheme over any noise model, thereby revealing how different measures of noise strength for a logical qubit depend on physical noise parameters.

## Example
As an example, the following plot produced by `chflow` describes the [fidelity](https://github.com/paviudes/chflow/wiki/Measures-of-noise-strength) of a logical qubit [error corrected](https://github.com/paviudes/chflow/wiki/Running-Simulations#running-simulations) for [coherent errors](https://github.com/paviudes/chflow/wiki/Quantum-channels#predefined-channels) \(rotations about Z-axis\) as a function of the angle of the over rotation error.
![rotz](https://github.com/paviudes/chflow/blob/master/docs/rotz.jpg)

The following command set is required to generate the above plot.

```python
Pavithrans-MacBook-Pro:chflow pavithran$ ./chflow.sh
>> sbload 11_04_2018_13_52_37
Simulation data is available for 0% of the channels.
Preparing physical channels... 26 (100%) done.        
 >> ecc
Simulation data is available for 0% of the channels.
Please wait ...
100% done, approximately 0 seconds remaining ...   
done, in 870 seconds.
>> metrics fidelity 
>> collect
Simulation data is available for 100% of the channels.
>> lplot 0 fidelity
>> 
```

For any stabilizer error correction scheme, `chflow` tracks the _flow of a quantum channel_ -- evolution from a physical noise process to the _effective quantum channel_ that affects the underlying logical information encompassing quantum error correction steps. To learn more about such techniques, see [arXiv:1711.04736](https://arxiv.org/abs/1711.04736) and [Phys. Rev. A 95, 042332](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.042332).

In addition to its main purpose, this package can also be used to perform simple but useful operations on quantum error correcting codes as well as quantum channels. For instance, derivation of the [canonical basis](https://github.com/paviudes/chflow/wiki/Quantum-error-correction#complete-description-of-a-stabilizer-code) \(Stabilizers, Logicals and Pure errors\) for an error correcting code, converting between various [representations of a quantum channel](https://github.com/paviudes/chflow/wiki/Quantum-channels#representations-for-quantum-channels), computing [Pauli approximations](https://github.com/paviudes/chflow/wiki/Quantum-channels#approximations-to-a-pauli-channel) of a quantum channel and so on.

## Downloading and installing
The latest version of `chflow` can be obtained by [cloning this repository](https://help.github.com/articles/cloning-a-repository/). All of the interface and plotting tools require [Python](https://www.python.org/downloads/) (any version higher than) 2.7 and standard packages such as [Numpy](https://www.scipy.org/install.html), [Scipy](https://www.scipy.org/install.html) and [Matplotlib](https://matplotlib.org/users/installing.html). On the other hand, most of the error correction simulations that run in the backend are implemented as Cython extensions, with the associated C files already provided. However, they can also be generated using the `build cython` command in `chflow`, which requires [Cython](http://docs.cython.org/en/latest/src/quickstart/install.html) and [gcc](https://gcc.gnu.org/install/) compilers. Some advanced functions require the [PICOS](http://picos.zib.de/intro.html#installation), [CVXOPT](http://cvxopt.org/install/index.html) and [scikit-learn](http://scikit-learn.org/stable/install.html) packages. When `chflow` is started, a text file called `requirements.txt` will be generated in `chflow/` containing the names of the missing packages. To fulfill all of those requirements, one can simply run `pip install -r requirements.txt` on the shell. MacOS users must prepend this with a `sudo`. Refer to the [Wiki](https://github.com/paviudes/chflow/wiki) for instructions on running `chflow`.

## Funding
This material is based upon work supported by, or in part by, the U.S. Army Research Laboratory and the U.S. Army Research Office  under 
Contract W911NF-14-C-0048.

## Contributing to `chflow`

There are no restrictions on contributing to `chflow`. While updating the GitHub repository, care should be taken to avoid syncing data files and compiler generated files. Two commands, `clean` and `clean git`, in `chflow` remove files unwanted for syncing. The latter option returns the `chflow/` directory to its factory setting. Note that when files are removed, they are not actually _deleted_, but _moved_ into `chflow/.ignore`. It is then safe to delete this folder using `rm -r .ignore/`.

## Citing `chflow`
```text
P. Iyer and D. Poulin. chflow: quantum error correction for realistic noise. https://github.com/paviudes/chflow, DOI: 10.5281/zenodo.1216202, April 2018.
```
```text
@misc{chflow,
  author       = {Pavithran Iyer and David Poulin},
  title        = {chflow: Quantum error correction for realistic noise},
  month        = {April},
  year         = {2018},
  doi          = {10.5281/zenodo.1216202},
  url          = {https://github.com/paviudes/chflow}
}
```
## Comments
Comments and suggestions are welcome, they can be addressed to [pavithran.iyer.sridharan@usherbrooke.ca](mailto:pavithran.iyer.sridharan@usherbrooke.ca?subject="comments on chflow").
